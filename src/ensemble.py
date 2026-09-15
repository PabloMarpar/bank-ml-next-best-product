"""
FASE 3c -- Mezcla de modelos, hecha sin hacer trampas.

    python src/ensemble.py

--------------------------------------------------------------------------------------
Por que mezclar, y por que casi todo el mundo lo hace mal
--------------------------------------------------------------------------------------
Los tres enfoques que funcionan miran cosas distintas: la popularidad captura la
estructura del mercado, XGBoost las caracteristicas del cliente, y la red secuencial el
orden de su trayectoria. Cuando varios modelos aciertan en sitios distintos, combinarlos
suele dar mas que el mejor de ellos.

El problema es COMO se eligen los pesos de la mezcla. La forma comoda es probar pesos
sobre el mes de validacion y quedarse con los que mejor puntuan ahi. Eso es ajustar
contra el conjunto de test: el numero final sale bonito y no significa nada, porque se ha
usado la respuesta para elegir la pregunta.

Aqui el reparto temporal es de tres tramos, y cada uno tiene un unico trabajo:

    meses <= t-2    entrenar los modelos base
    mes t-1         elegir los pesos de la mezcla
    mes t           medir, una sola vez

El mes t no participa en ninguna decision. Es la unica forma de que la cifra final
signifique algo.

--------------------------------------------------------------------------------------
Como se mezcla: por posicion, no por probabilidad
--------------------------------------------------------------------------------------
Las salidas de los tres modelos no son comparables entre si. La popularidad da un orden
sin numeros, XGBoost da probabilidades comprimidas hacia el centro, y la red da un
softmax con su propia escala. Promediarlas directamente hace que mande el que tenga la
escala mas ancha, que no tiene nada que ver con el que acierte mas.

La solucion es fusion por posicion (reciprocal rank fusion): cada modelo aporta 1/(60+p),
donde p es la posicion en la que coloca cada producto. Al usar solo el ORDEN, la escala
deja de importar. El 60 amortigua las primeras posiciones para que un modelo muy seguro y
muy equivocado no arrastre la mezcla entera.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
from pyspark.sql import functions as F

from common import (
    DATA_FEATURES,
    MES_VALIDACION,
    NOMBRES_PRODUCTO,
    crear_sesion,
    cronometro,
    guardar_metricas,
)
from recommend import (
    K,
    average_precision,
    predecir_ranker,
    ranking_popularidad,
)

PRODUCTOS = list(NOMBRES_PRODUCTO)

# Se rellena en main(): indice de mes -> fecha, para localizar las predicciones de las
# redes secuenciales, que se guardan con la fecha en el nombre.
MESES_POR_INDICE: dict[int, str] = {}
N_PRODUCTOS = len(PRODUCTOS)

# Constante de amortiguacion de la fusion por posicion. 60 es el valor del articulo
# original de Cormack (2009) y funciona bien sin ajustarlo; ajustarlo tambien habria que
# hacerlo sobre t-1, no sobre t.
AMORTIGUACION = 60.0


def matriz_posiciones(orden_por_cliente: dict[int, list[str]], clientes: list[int]):
    """Convierte listas ordenadas de productos en una matriz de puntuaciones 1/(60+p)."""
    indice = {p: i for i, p in enumerate(PRODUCTOS)}
    matriz = np.zeros((len(clientes), N_PRODUCTOS), dtype=np.float32)
    for fila, cliente in enumerate(clientes):
        for posicion, producto in enumerate(orden_por_cliente.get(cliente, [])):
            if producto in indice:
                matriz[fila, indice[producto]] = 1.0 / (AMORTIGUACION + posicion)
    return matriz


def map_de_matriz(puntuaciones, poseidos, reales, k: int = K) -> float:
    """MAP@7 sobre una matriz de puntuaciones, descartando lo que el cliente ya tiene."""
    enmascarado = np.where(poseidos == 1, -np.inf, puntuaciones)
    orden = np.argsort(-enmascarado, axis=1)[:, :k]
    total = 0.0
    for i, indices in enumerate(orden):
        total += average_precision([PRODUCTOS[j] for j in indices], reales[i], k)
    return total / len(reales)


def recoger(spark, mes_idx, columnas_extra=()):
    """Trae de Spark lo necesario para evaluar un mes: poseidos y altas reales."""
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
            "ncodpers",
            F.array_compact(altas).alias("reales"),
            F.array_compact(poseidos).alias("poseidos"),
            *columnas_extra,
        )
        .filter(F.size("reales") > 0)
        .collect()
    )


def buscar_pesos(matrices: dict, poseidos, reales, paso: float = 0.1):
    """Rejilla sobre el simplex de pesos. Se ejecuta SOLO sobre el mes t-1.

    Con tres modelos y paso 0,1 son 66 combinaciones: se puede explorar entero y no hace
    falta nada mas sofisticado. Lo que importa aqui no es el optimizador, es sobre que
    datos se optimiza.
    """
    nombres = list(matrices)
    valores = [round(v * paso, 4) for v in range(int(1 / paso) + 1)]

    mejor = None
    for combinacion in itertools.product(valores, repeat=len(nombres)):
        if abs(sum(combinacion) - 1.0) > 1e-6:
            continue
        mezcla = sum(
            peso * matrices[nombre]
            for peso, nombre in zip(combinacion, nombres)
        )
        puntuacion = map_de_matriz(mezcla, poseidos, reales)
        if mejor is None or puntuacion > mejor[1]:
            mejor = (dict(zip(nombres, combinacion)), puntuacion)
    return mejor


def main() -> None:
    parser = argparse.ArgumentParser(description="Mezcla de modelos con pesos honestos")
    parser.add_argument("--mes-validacion", default=MES_VALIDACION)
    parser.add_argument("--arboles", type=int, default=80)
    parser.add_argument("--profundidad", type=int, default=8)
    args = parser.parse_args()

    spark = crear_sesion("ensemble")
    spark.sparkContext.setLogLevel("ERROR")
    metricas: dict = {"mes_validacion": args.mes_validacion, "amortiguacion": AMORTIGUACION}

    print("\n=== FASE 3c: mezcla de modelos ===\n")

    feats = spark.read.parquet(str(DATA_FEATURES))
    mes_test = feats.filter(F.col("fecha_dato") == args.mes_validacion).select(
        F.first("mes_idx").alias("idx")
    ).collect()[0]["idx"]
    mes_pesos = mes_test - 1

    # Indice de mes -> fecha, para poder localizar el fichero de predicciones de cada
    # red secuencial, que se nombra por fecha.
    global MESES_POR_INDICE
    MESES_POR_INDICE = {
        int(f["mes_idx"]): str(f["fecha_dato"])
        for f in feats.select("mes_idx", "fecha_dato").distinct().collect()
    }

    entrenamiento = feats.filter(F.col("mes_idx") < mes_pesos).cache()
    datos_pesos = feats.filter(F.col("mes_idx") == mes_pesos).cache()
    datos_test = feats.filter(F.col("mes_idx") == mes_test).cache()

    print(f"  Entrenamiento de los modelos base: meses < {mes_pesos} "
          f"({entrenamiento.count():,} filas)")
    print(f"  Eleccion de pesos: mes {mes_pesos}")
    print(f"  Medicion final:    mes {mes_test} ({args.mes_validacion})")
    print("  El mes de medicion no participa en ninguna decision.\n")

    # --- Modelos base, entrenados una vez y aplicados a los dos meses -------------------
    ranking = ranking_popularidad(entrenamiento)

    predicciones = {}
    for etiqueta, datos in (("pesos", datos_pesos), ("test", datos_test)):
        with cronometro(f"xgboost_{etiqueta}", metricas):
            pred = predecir_ranker(
                entrenamiento, datos, algoritmo="xgboost",
                arboles=args.arboles, profundidad=args.profundidad,
            ).select("ncodpers", "recomendados").collect()
        predicciones[f"xgboost_{etiqueta}"] = {
            f["ncodpers"]: list(f["recomendados"]) for f in pred
        }

    # --- Evaluacion por mes -------------------------------------------------------------
    resultados = {}
    pesos_elegidos = None

    for etiqueta, mes in (("pesos", mes_pesos), ("test", mes_test)):
        filas = recoger(spark, mes)
        clientes = [f["ncodpers"] for f in filas]
        reales = [list(f["reales"]) for f in filas]

        indice = {p: i for i, p in enumerate(PRODUCTOS)}
        poseidos = np.zeros((len(filas), N_PRODUCTOS), dtype=np.uint8)
        for i, f in enumerate(filas):
            for p in f["poseidos"]:
                poseidos[i, indice[p]] = 1

        matrices = {
            # La popularidad es la misma lista para todos: se replica.
            "popularidad": matriz_posiciones(
                {c: ranking for c in clientes}, clientes
            ),
            "xgboost": matriz_posiciones(
                predicciones[f"xgboost_{etiqueta}"], clientes
            ),
        }

        # Las redes secuenciales entran si estan disponibles PARA ESE MES.
        #
        # Este era el fallo de la primera version: se cargaba un unico fichero y solo
        # para el mes de medicion. Resultado, los pesos se elegian sin las redes y luego
        # la mezcla no podia usarlas -- la conclusion "mezclar no aporta" era un artefacto
        # del codigo, no un hallazgo sobre los modelos. Hay que generar las predicciones
        # de los dos meses (src/secuencia.py --mes-validacion) para que esto funcione.
        fecha = MESES_POR_INDICE.get(mes)
        ruta_red = (
            DATA_FEATURES.parent / "export" / f"probs_secuencia_{fecha}.npz"
            if fecha else None
        )
        if ruta_red is not None and ruta_red.exists():
            guardado = np.load(ruta_red)
            posicion = {int(c): i for i, c in enumerate(guardado["ncodpers"])}
            for nombre in ("gru", "transformer"):
                if nombre not in guardado:
                    continue
                probs = guardado[nombre]
                ordenes = {}
                for cliente in clientes:
                    j = posicion.get(cliente)
                    if j is None:
                        continue
                    ordenes[cliente] = [
                        PRODUCTOS[b] for b in np.argsort(-probs[j])
                    ]
                matrices[nombre] = matriz_posiciones(ordenes, clientes)

        individuales = {
            nombre: map_de_matriz(matriz, poseidos, reales)
            for nombre, matriz in matrices.items()
        }

        if etiqueta == "pesos":
            print(f"  Mes {mes_pesos} -- modelos individuales:")
            for nombre, valor in sorted(individuales.items(), key=lambda kv: -kv[1]):
                print(f"    {nombre:<16} {valor:.5f}")
            with cronometro("busqueda_pesos", metricas):
                pesos_elegidos, puntuacion = buscar_pesos(matrices, poseidos, reales)
            metricas["pesos_elegidos"] = pesos_elegidos
            metricas["map_mezcla_en_mes_de_pesos"] = round(puntuacion, 5)
            print(f"\n  Pesos elegidos sobre el mes {mes_pesos}: {pesos_elegidos}")
            print(f"  MAP de la mezcla en ese mes: {puntuacion:.5f}  "
                  f"(no es la cifra final: ahi se han elegido los pesos)\n")
        else:
            mezcla = sum(
                pesos_elegidos.get(nombre, 0.0) * matriz
                for nombre, matriz in matrices.items()
            )
            resultados = dict(individuales)
            resultados["mezcla"] = map_de_matriz(mezcla, poseidos, reales)

            print(f"  Mes {mes_test} -- MEDICION FINAL:")
            for nombre, valor in sorted(resultados.items(), key=lambda kv: -kv[1]):
                marca = "  <-- mezcla" if nombre == "mezcla" else ""
                print(f"    {nombre:<16} {valor:.5f}{marca}")

    metricas["map7_final"] = {k: round(v, 5) for k, v in resultados.items()}

    mejor_individual = max(
        (v for k, v in resultados.items() if k != "mezcla"), default=0.0
    )
    aporte = resultados.get("mezcla", 0.0) - mejor_individual
    metricas["aporte_mezcla"] = round(aporte, 5)
    metricas["aporte_mezcla_pct"] = round(100 * aporte / max(mejor_individual, 1e-9), 2)

    veredicto = (
        f"La mezcla aporta {aporte:+.5f} MAP sobre el mejor modelo suelto "
        f"({metricas['aporte_mezcla_pct']:+.1f}%). Merece la pena mantener los tres "
        "modelos en produccion."
        if aporte > 0.002 else
        f"La mezcla NO aporta ({aporte:+.5f} MAP). Los modelos aciertan en los mismos "
        "clientes, asi que combinarlos no anade informacion -- solo complejidad de "
        "mantenimiento. Conviene quedarse con el mejor modelo suelto."
    )
    metricas["veredicto"] = veredicto
    print(f"\n  {veredicto}")

    guardar_metricas("ensemble_metrics.json", metricas)
    spark.stop()


if __name__ == "__main__":
    main()
