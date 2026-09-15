"""
FASE 3e -- Enrutado por segmento: usar el mejor modelo para cada tipo de cliente.

    python src/router.py

--------------------------------------------------------------------------------------
La idea, y la trampa que tiene
--------------------------------------------------------------------------------------
Hay varios modelos que funcionan y ninguno gana en todo. El Transformer lee trayectorias,
asi que deberia brillar con clientes de historial largo; XGBoost se apoya en el perfil,
asi que deberia aguantar mejor cuando no hay historia. Si eso es cierto, en vez de elegir
UN modelo se puede elegir uno POR TIPO DE CLIENTE.

Pero hay una forma de hacer esto que parece razonable y es hacer trampa:

    mirar el mes de test, ver que modelo gana en cada segmento, y quedarse con ese.

Eso es usar la respuesta para tomar la decision. El resultado sale precioso y no significa
nada, porque en produccion no vas a tener el mes que viene para consultarlo.

Aqui la decision y la medicion viven en meses distintos:

    meses <= t-2   entrenar los modelos
    mes t-1        decidir que modelo atiende a cada segmento
    mes t          aplicar esa decision a ciegas y medir, una sola vez

Si el enrutado sobrevive a eso, es real. Si solo funcionaba mirando el test, aqui se cae.

--------------------------------------------------------------------------------------
Por que puede fallar aunque los modelos sean distintos
--------------------------------------------------------------------------------------
Que un modelo gane en un segmento durante el mes t-1 no garantiza que gane en t. Si la
diferencia entre modelos en ese segmento es pequena comparada con el ruido del mes, la
decision es esencialmente una moneda al aire y el enrutado no aporta -- o resta.

Por eso se reporta tambien cuantos segmentos cambian de ganador entre los dos meses. Es
la comprobacion de estabilidad: si el ganador baila, la regla no vale.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from pyspark.sql import functions as F

from common import (
    DATA_FEATURES,
    MES_VALIDACION,
    NOMBRES_PRODUCTO,
    OUTPUTS,
    crear_sesion,
    cronometro,
    guardar_metricas,
)
from recommend import (
    K,
    average_precision,
    predecir_popularidad,
    predecir_ranker,
    ranking_popularidad,
)

PRODUCTOS = list(NOMBRES_PRODUCTO)
N_PRODUCTOS = len(PRODUCTOS)

# Formas de partir la cartera. Cada una es una hipotesis distinta sobre donde puede
# haber diferencias entre modelos.
SEGMENTACIONES = {
    "historia": (
        "meses_observados",
        [("1 mes", 0, 1), ("2-3", 2, 3), ("4-6", 4, 6), ("7-11", 7, 11),
         ("12+", 12, 999)],
    ),
    "n_productos": (
        "n_prod_prev",
        [("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4-5", 4, 5), ("6+", 6, 99)],
    ),
}


def recoger_mes(spark, mes_idx: int):
    """Clientes evaluables de un mes, con sus altas reales, poseidos y segmentos."""
    feats = spark.read.parquet(str(DATA_FEATURES)).filter(F.col("mes_idx") == mes_idx)
    altas = F.array(*[
        F.when(F.col(f"alta_{p}") == 1, F.lit(p)).otherwise(F.lit(None))
        for p in PRODUCTOS
    ])
    poseidos = F.array(*[
        F.when(F.col(f"prev_{p}") == 1, F.lit(p)).otherwise(F.lit(None))
        for p in PRODUCTOS
    ])
    return (
        feats.select(
            "ncodpers", "meses_observados", "n_prod_prev", "prev_segmento",
            F.array_compact(altas).alias("reales"),
            F.array_compact(poseidos).alias("poseidos"),
        )
        .filter(F.size("reales") > 0)
        .collect()
    )


def ordenes_de_npz(ruta, clientes: list[int]) -> dict[str, dict[int, list[str]]]:
    """Lee las probabilidades guardadas por src/secuencia.py y las pasa a ordenes."""
    if not ruta.exists():
        return {}
    guardado = np.load(ruta)
    posicion = {int(c): i for i, c in enumerate(guardado["ncodpers"])}
    salida = {}
    for nombre in ("gru", "transformer"):
        if nombre not in guardado:
            continue
        probs = guardado[nombre]
        salida[nombre] = {
            cliente: [PRODUCTOS[b] for b in np.argsort(-probs[posicion[cliente]])]
            for cliente in clientes
            if cliente in posicion
        }
    return salida


def map_por_cliente(ordenes: dict[int, list[str]], filas, k: int = K) -> dict[int, float]:
    """AP@k de cada cliente para un modelo, quitando lo que ya tiene."""
    salida = {}
    for fila in filas:
        cliente = fila["ncodpers"]
        orden = ordenes.get(cliente)
        if orden is None:
            salida[cliente] = 0.0
            continue
        poseidos = set(fila["poseidos"])
        recomendados = [p for p in orden if p not in poseidos][:k]
        salida[cliente] = average_precision(recomendados, list(fila["reales"]), k)
    return salida


def evaluar_segmentos(aps: dict[str, dict[int, float]], filas, columna, tramos):
    """MAP de cada modelo dentro de cada segmento."""
    resultado = []
    for nombre, minimo, maximo in tramos:
        clientes = [
            f["ncodpers"] for f in filas
            if f[columna] is not None and minimo <= f[columna] <= maximo
        ]
        if not clientes:
            continue
        fila = {"segmento": nombre, "n": len(clientes)}
        for modelo, por_cliente in aps.items():
            fila[modelo] = sum(por_cliente.get(c, 0.0) for c in clientes) / len(clientes)
        fila["ganador"] = max(
            (m for m in aps), key=lambda m: fila[m]
        )
        resultado.append(fila)
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrutado por segmento entre modelos")
    parser.add_argument("--mes-validacion", default=MES_VALIDACION)
    parser.add_argument("--arboles", type=int, default=80)
    parser.add_argument("--profundidad", type=int, default=8)
    args = parser.parse_args()

    spark = crear_sesion("router")
    spark.sparkContext.setLogLevel("ERROR")
    metricas: dict = {"mes_validacion": args.mes_validacion}

    print("\n=== FASE 3e: enrutado por segmento ===\n")

    feats = spark.read.parquet(str(DATA_FEATURES))
    mes_test = feats.filter(F.col("fecha_dato") == args.mes_validacion).select(
        F.first("mes_idx").alias("idx")
    ).collect()[0]["idx"]
    mes_decision = mes_test - 1
    fechas = {
        int(f["mes_idx"]): str(f["fecha_dato"])
        for f in feats.select("mes_idx", "fecha_dato").distinct().collect()
    }

    entrenamiento = feats.filter(F.col("mes_idx") < mes_decision).cache()
    print(f"  Entrenamiento: meses < {mes_decision} ({entrenamiento.count():,} filas)")
    print(f"  Decision del enrutado: mes {mes_decision} ({fechas[mes_decision]})")
    print(f"  Medicion final:        mes {mes_test} ({fechas[mes_test]})")
    print("  El mes de medicion no participa en la decision.\n")

    ranking = ranking_popularidad(entrenamiento)
    export = DATA_FEATURES.parent / "export"

    aps_por_mes = {}
    filas_por_mes = {}
    for etiqueta, mes in (("decision", mes_decision), ("test", mes_test)):
        datos_mes = feats.filter(F.col("mes_idx") == mes)
        filas = recoger_mes(spark, mes)
        clientes = [f["ncodpers"] for f in filas]
        filas_por_mes[etiqueta] = filas

        ordenes = ordenes_de_npz(export / f"probs_secuencia_{fechas[mes]}.npz", clientes)
        if not ordenes:
            raise SystemExit(
                f"Faltan las predicciones de las redes para {fechas[mes]}.\n"
                f"Ejecuta: python src/secuencia.py --mes-validacion {fechas[mes]}"
            )

        with cronometro(f"xgboost_{etiqueta}", metricas):
            pred = predecir_ranker(
                entrenamiento, datos_mes, algoritmo="xgboost",
                arboles=args.arboles, profundidad=args.profundidad,
            ).select("ncodpers", "recomendados").collect()
        ordenes["xgboost"] = {f["ncodpers"]: list(f["recomendados"]) for f in pred}

        pop = predecir_popularidad(datos_mes, ranking).collect()
        ordenes["popularidad"] = {f["ncodpers"]: list(f["recomendados"]) for f in pop}

        aps_por_mes[etiqueta] = {
            modelo: map_por_cliente(orden, filas)
            for modelo, orden in ordenes.items()
        }

    modelos = list(aps_por_mes["test"])
    globales = {
        m: sum(aps_por_mes["test"][m].values()) / len(filas_por_mes["test"])
        for m in modelos
    }
    mejor_global = max(globales, key=globales.get)
    print("\n  MAP@7 global en el mes de medicion:")
    for m, v in sorted(globales.items(), key=lambda kv: -kv[1]):
        marca = "  <-- mejor modelo suelto" if m == mejor_global else ""
        print(f"    {m:<14} {v:.5f}{marca}")

    # --- Enrutado por cada forma de segmentar -------------------------------------------
    resultados = {}
    for nombre_seg, (columna, tramos) in SEGMENTACIONES.items():
        print(f"\n  --- Segmentacion por {nombre_seg} ---")

        tabla_decision = evaluar_segmentos(
            aps_por_mes["decision"], filas_por_mes["decision"], columna, tramos
        )
        tabla_test = evaluar_segmentos(
            aps_por_mes["test"], filas_por_mes["test"], columna, tramos
        )

        # La regla se fija con el mes de decision. El de test solo se usa para aplicarla.
        regla = {f["segmento"]: f["ganador"] for f in tabla_decision}

        cabecera = f"  {'segmento':<10} {'n':>7}"
        for m in modelos:
            cabecera += f" {m[:11]:>12}"
        print(cabecera + "   decide -> gana de verdad")

        cambios = 0
        for fila_t in tabla_test:
            linea = f"  {fila_t['segmento']:<10} {fila_t['n']:>7,}"
            for m in modelos:
                linea += f" {fila_t[m]:>12.5f}"
            elegido = regla.get(fila_t["segmento"], mejor_global)
            estable = "=" if elegido == fila_t["ganador"] else "!"
            if elegido != fila_t["ganador"]:
                cambios += 1
            print(f"{linea}   {elegido} {estable} {fila_t['ganador']}")

        # MAP aplicando la regla decidida en t-1 sobre el mes de test.
        total, acumulado = 0, 0.0
        for fila_t in tabla_test:
            elegido = regla.get(fila_t["segmento"], mejor_global)
            acumulado += fila_t[elegido] * fila_t["n"]
            total += fila_t["n"]
        map_enrutado = acumulado / total

        resultados[nombre_seg] = {
            "regla": regla,
            "map_enrutado": round(map_enrutado, 5),
            "map_mejor_suelto": round(globales[mejor_global], 5),
            "ganancia_pct": round(
                100 * (map_enrutado / globales[mejor_global] - 1), 3
            ),
            "segmentos_que_cambian_de_ganador": cambios,
            "segmentos_totales": len(tabla_test),
            "tabla_decision": [
                {k: (round(v, 5) if isinstance(v, float) else v) for k, v in f.items()}
                for f in tabla_decision
            ],
            "tabla_test": [
                {k: (round(v, 5) if isinstance(v, float) else v) for k, v in f.items()}
                for f in tabla_test
            ],
        }
        r = resultados[nombre_seg]
        print(f"\n    Enrutado: {map_enrutado:.5f}  |  mejor suelto "
              f"({mejor_global}): {globales[mejor_global]:.5f}  "
              f"->  {r['ganancia_pct']:+.3f}%")
        print(f"    Estabilidad: {cambios} de {len(tabla_test)} segmentos cambian de "
              f"ganador entre el mes de decision y el de medicion.")

    metricas["modelos"] = {m: round(v, 5) for m, v in globales.items()}
    metricas["mejor_global"] = mejor_global
    metricas["segmentaciones"] = resultados

    mejor_seg = max(resultados, key=lambda s: resultados[s]["ganancia_pct"])
    ganancia = resultados[mejor_seg]["ganancia_pct"]
    metricas["mejor_segmentacion"] = mejor_seg
    metricas["ganancia_maxima_pct"] = ganancia

    if ganancia > 0.5:
        veredicto = (
            f"El enrutado por {mejor_seg} aporta {ganancia:+.2f}% sobre usar solo "
            f"{mejor_global}, y la regla se decidio sin mirar el mes de medicion. "
            "Merece la pena mantener varios modelos en produccion."
        )
    elif ganancia > 0:
        veredicto = (
            f"El enrutado aporta {ganancia:+.2f}%, demasiado poco para justificar "
            f"mantener varios modelos, reentrenarlos y decidir cual atiende a quien. "
            f"La decision correcta es quedarse solo con {mejor_global}."
        )
    else:
        veredicto = (
            f"El enrutado NO aporta ({ganancia:+.2f}%). El ganador por segmento no es "
            "estable entre meses: la diferencia entre modelos dentro de cada segmento es "
            "menor que el ruido mensual, asi que elegir por segmento es casi echarlo a "
            f"suertes. Se usa {mejor_global} para todos."
        )
    metricas["veredicto"] = veredicto
    print(f"\n  VEREDICTO: {veredicto}")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    guardar_metricas("router_metrics.json", metricas)
    spark.stop()


if __name__ == "__main__":
    main()
