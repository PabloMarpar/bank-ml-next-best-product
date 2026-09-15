"""
ANALISIS -- Cada cuanto caduca el modelo.

    python src/deriva.py

--------------------------------------------------------------------------------------
La pregunta que nadie hace hasta que ya es tarde
--------------------------------------------------------------------------------------
Un modelo se entrena una vez y se mide una vez, y con esa cifra se va a produccion. Pero
el mundo del que aprendio sigue cambiando: entran clientes nuevos, el banco saca
productos, cambia la campana comercial, llega el verano. El modelo no se entera de nada
de eso. Sigue contestando con las reglas que aprendio el dia que se entreno.

A eso se le llama deriva, y no es una posibilidad teorica: es lo que pasa siempre. La
unica pregunta abierta es a que velocidad.

Este script la mide. Entrena UNA sola vez con los meses disponibles hasta una fecha, y
despues evalua ese mismo modelo congelado en cada uno de los meses siguientes, uno a uno.
La curva que sale responde a algo muy concreto:

    "Cuantos meses aguanta este modelo antes de que reentrenarlo compense el trabajo?"

--------------------------------------------------------------------------------------
Por que se compara contra un modelo reentrenado
--------------------------------------------------------------------------------------
Que el rendimiento baje no demuestra por si solo que el modelo se haya quedado viejo:
tambien puede ser que ese mes concreto sea mas dificil para cualquiera. En junio se
contratan cosas distintas que en febrero, y eso afecta igual a un modelo nuevo.

Para separar las dos cosas se entrena tambien un modelo "fresco" para cada mes, usando
solo datos anteriores a ese mes. La diferencia entre el congelado y el fresco es lo que
de verdad cuesta no reentrenar. Si los dos bajan a la vez, el mes era dificil. Si solo
baja el congelado, el modelo ha caducado.

Se usa el objetivo de bajas y no el de recomendacion porque es binario: el AUC es una
metrica estable, comparable entre meses, y no depende de cuanta gente contrato ese mes.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.classification import GBTClassifier  # noqa: E402
from pyspark.ml.evaluation import BinaryClassificationEvaluator  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from churn import CATEGORICAS, NUMERICAS, PRODUCTOS, a_formato_largo  # noqa: E402
from common import (  # noqa: E402
    DATA_FEATURES,
    OUTPUTS,
    crear_sesion,
    cronometro,
    etapas_preprocesado,
    guardar_metricas,
    marcar_faltantes,
)


def entrenar(datos, arboles: int, profundidad: int):
    etapas, col_features = etapas_preprocesado(
        categoricas=CATEGORICAS,
        numericas=NUMERICAS,
        columnas_extra=[f"prev_{p}" for p in PRODUCTOS],
    )
    gbt = GBTClassifier(
        featuresCol=col_features, labelCol="label",
        maxIter=arboles, maxDepth=profundidad, maxBins=256, seed=42,
    )
    return Pipeline(stages=[*etapas, gbt]).fit(datos)


def auc_de(modelo, datos) -> float:
    return BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC"
    ).evaluate(modelo.transform(datos))


def dibujar(puntos: list[dict], destino) -> None:
    meses = [p["meses_desde_entrenamiento"] for p in puntos]
    congelado = [p["auc_congelado"] for p in puntos]
    fresco = [p["auc_reentrenado"] for p in puntos]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(meses, congelado, marker="o", linewidth=2.2, color="#b91c1c",
            label="Modelo congelado (entrenado una vez)")
    ax.plot(meses, fresco, marker="s", linewidth=2.2, color="#15803d",
            label="Modelo reentrenado cada mes")
    ax.fill_between(meses, congelado, fresco, alpha=0.15, color="#b91c1c")

    ax.axhline(0.5, linestyle=":", color="#999999", linewidth=1.2)
    ax.text(meses[0], 0.505, "azar", fontsize=9, color="#666666")

    ax.set_xlabel("Meses transcurridos desde que se entreno el modelo")
    ax.set_ylabel("AUC sobre el mes evaluado")
    ax.set_title("Cuanto cuesta no reentrenar")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    print(f"\n  Grafico guardado en {destino}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mide la deriva temporal del modelo")
    parser.add_argument("--meses-entrenamiento", type=int, default=4)
    parser.add_argument("--horizonte", type=int, default=6,
                        help="Cuantos meses hacia delante se evalua el modelo congelado")
    parser.add_argument("--arboles", type=int, default=30)
    parser.add_argument("--profundidad", type=int, default=6)
    args = parser.parse_args()

    spark = crear_sesion("deriva")
    spark.sparkContext.setLogLevel("ERROR")
    metricas: dict = {
        "meses_entrenamiento": args.meses_entrenamiento,
        "horizonte_meses": args.horizonte,
    }

    print("\n=== ANALISIS: deriva temporal ===\n")

    feats = spark.read.parquet(str(DATA_FEATURES))
    largo = marcar_faltantes(a_formato_largo(feats), NUMERICAS).cache()

    meses = sorted(
        r["mes_idx"] for r in largo.select("mes_idx").distinct().collect()
    )
    # El modelo congelado se entrena con la ventana mas antigua posible, para que quede
    # sitio por delante donde medir la degradacion.
    fin_entrenamiento = meses[args.meses_entrenamiento]
    evaluables = [m for m in meses if m > fin_entrenamiento][:args.horizonte]

    entrenamiento = largo.filter(F.col("mes_idx") <= fin_entrenamiento).cache()
    print(f"  Modelo congelado: entrenado con {entrenamiento.count():,} filas "
          f"hasta el mes indice {fin_entrenamiento}")
    print(f"  Se evaluara en los {len(evaluables)} meses siguientes\n")

    with cronometro("entrenar_congelado", metricas):
        modelo_congelado = entrenar(entrenamiento, args.arboles, args.profundidad)

    puntos = []
    print(f"  {'mes':>4} {'congelado':>11} {'reentrenado':>13} {'coste':>9}")
    for distancia, mes in enumerate(evaluables, start=1):
        objetivo = largo.filter(F.col("mes_idx") == mes).cache()
        if objetivo.count() == 0:
            continue

        auc_congelado = auc_de(modelo_congelado, objetivo)

        # Modelo fresco: misma receta y misma cantidad de meses, pero con la ventana
        # deslizada hasta justo antes del mes que se evalua.
        ventana = largo.filter(
            (F.col("mes_idx") < mes) & (F.col("mes_idx") >= mes - args.meses_entrenamiento)
        )
        modelo_fresco = entrenar(ventana, args.arboles, args.profundidad)
        auc_fresco = auc_de(modelo_fresco, objetivo)

        coste = auc_fresco - auc_congelado
        puntos.append({
            "mes_idx": int(mes),
            "meses_desde_entrenamiento": distancia,
            "auc_congelado": round(auc_congelado, 4),
            "auc_reentrenado": round(auc_fresco, 4),
            "coste_de_no_reentrenar": round(coste, 4),
        })
        print(f"  {distancia:>4} {auc_congelado:>11.4f} {auc_fresco:>13.4f} "
              f"{coste:>+9.4f}")
        objetivo.unpersist()

    metricas["curva"] = puntos

    # --- Interpretacion -----------------------------------------------------------------
    if puntos:
        primero, ultimo = puntos[0], puntos[-1]
        caida_total = primero["auc_congelado"] - ultimo["auc_congelado"]
        caida_por_mes = caida_total / max(ultimo["meses_desde_entrenamiento"] - 1, 1)
        coste_medio = sum(p["coste_de_no_reentrenar"] for p in puntos) / len(puntos)

        metricas.update({
            "caida_auc_total": round(caida_total, 4),
            "caida_auc_por_mes": round(caida_por_mes, 5),
            "coste_medio_de_no_reentrenar": round(coste_medio, 5),
        })

        print(f"\n  El modelo congelado pierde {caida_total:+.4f} de AUC en "
              f"{ultimo['meses_desde_entrenamiento']} meses "
              f"({caida_por_mes:+.5f} al mes).")
        print(f"  Reentrenar cada mes vale, de media, {coste_medio:+.4f} de AUC.")

        # Regla practica: se considera que compensa reentrenar cuando la diferencia
        # acumulada supera 0,01 de AUC, que es el orden de magnitud a partir del cual
        # la diferencia se nota en el ranking de una campana real.
        umbral = 0.01
        mes_caducidad = next(
            (p["meses_desde_entrenamiento"] for p in puntos
             if p["coste_de_no_reentrenar"] >= umbral),
            None,
        )
        if mes_caducidad:
            recomendacion = (
                f"Reentrenar cada {mes_caducidad} meses. A partir de ahi, no hacerlo "
                f"cuesta mas de {umbral} de AUC, que ya se nota en el orden de la lista "
                "de llamadas."
            )
        else:
            recomendacion = (
                f"En el horizonte medido ({len(puntos)} meses) el modelo congelado "
                f"aguanta: la diferencia contra reentrenar nunca llega a {umbral} de AUC. "
                "Con un reentrenamiento trimestral iria sobrado. Conviene repetir esta "
                "medicion con mas meses antes de darlo por bueno a largo plazo."
            )
        metricas["recomendacion"] = recomendacion
        print(f"\n  RECOMENDACION: {recomendacion}")

        OUTPUTS.mkdir(parents=True, exist_ok=True)
        dibujar(puntos, OUTPUTS / "deriva_temporal.png")

    guardar_metricas("deriva_metrics.json", metricas)
    spark.stop()


if __name__ == "__main__":
    main()
