"""
ANALISIS -- La renta que falta a uno de cada cinco clientes.

    python src/analisis_renta.py

Al 20,5% de las filas les falta la renta. La reaccion habitual es imputar la mediana y
seguir adelante. Este script hace las dos preguntas que esa reaccion se salta.

--------------------------------------------------------------------------------------
PREGUNTA 1: falta al azar, o falta por algun motivo?
--------------------------------------------------------------------------------------
No es lo mismo. Si falta al azar, el hueco es ruido y la imputacion solo repara un dato
perdido. Si falta por algun motivo -- clientes de cierto canal, de cierta antiguedad, de
cierta provincia -- entonces el hueco EN SI MISMO es informacion, y tirarlo es perder
senal.

La forma de saberlo es directa: entrenar un modelo que prediga si la renta va a faltar,
usando el resto de variables. Si acierta poco (AUC cerca de 0,5) la ausencia es
aleatoria. Si acierta mucho, no lo es, y el indicador de ausencia se gana su sitio como
variable del modelo principal.

En la literatura estadistica esto es la distincion entre MAR (missing at random) y MNAR
(missing not at random). El nombre importa menos que la consecuencia practica.

--------------------------------------------------------------------------------------
PREGUNTA 2: mediana o modelo?
--------------------------------------------------------------------------------------
Si hay que rellenar el hueco, se puede poner la mediana de todos, o se puede predecir la
renta de ese cliente concreto a partir de su provincia, edad, segmento y canal.

La segunda suena mejor, pero no siempre lo es: un valor imputado por un modelo arrastra
el error del modelo y puede inventar estructura que no existe. Asi que no se decide por
intuicion. Se mide: se cogen las filas donde la renta SI se conoce, se esconde a
proposito, y se compara que estrategia se acerca mas al valor real.
"""

from __future__ import annotations

import argparse

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import functions as F

from common import (
    DATA_PARQUET,
    crear_sesion,
    cronometro,
    guardar_metricas,
)

# Variables disponibles para explicar la ausencia. No se incluye ninguna que dependa de
# la propia renta, obviamente.
CATEGORICAS = ["segmento", "canal_entrada", "sexo", "tiprel_1mes", "nomprov",
               "ind_empleado", "indresi", "pais_residencia"]
NUMERICAS = ["age", "antiguedad", "ind_actividad_cliente", "ind_nuevo", "n_productos"]


def preparar(spark):
    """Carga una muestra del dataset con las variables del analisis."""
    df = spark.read.parquet(str(DATA_PARQUET))
    productos = [c for c in df.columns if c.startswith("ind_") and c.endswith("_ult1")]

    return df.select(
        *CATEGORICAS,
        *[c for c in NUMERICAS if c != "n_productos"],
        F.col("renta"),
        sum(F.coalesce(F.col(p), F.lit(0)) for p in productos).alias("n_productos"),
    )


def _etapas(categoricas, numericas):
    """Etapas de preprocesado. Sin imputar la renta, evidentemente."""
    indexadores = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in categoricas
    ]
    codificadores = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe", handleInvalid="keep")
        for c in categoricas
    ]
    ensamblador = VectorAssembler(
        inputCols=[f"{c}_ohe" for c in categoricas] + numericas,
        outputCol="features",
        handleInvalid="skip",  # aqui se descartan filas incompletas en vez de imputar
    )
    return [*indexadores, *codificadores, ensamblador]


# --------------------------------------------------------------------------------------
# Pregunta 1: es predecible que falte?
# --------------------------------------------------------------------------------------
def diagnosticar_ausencia(datos, metricas: dict, arboles: int) -> None:
    print("\n--- PREGUNTA 1: la ausencia de renta, es aleatoria? ---\n")

    etiquetado = datos.withColumn("label", F.col("renta").isNull().cast("double"))
    tasa = etiquetado.agg(F.avg("label")).collect()[0][0]
    metricas["pct_renta_ausente"] = round(100 * tasa, 2)
    print(f"  A la renta le falta el {100 * tasa:.2f}% de las filas.")

    # Se descarta la propia renta de las variables, claro.
    train, test = etiquetado.randomSplit([0.8, 0.2], seed=42)

    gbt = GBTClassifier(
        featuresCol="features", labelCol="label",
        maxIter=arboles, maxDepth=5, maxBins=256, seed=42,
    )
    modelo = Pipeline(stages=[*_etapas(CATEGORICAS, NUMERICAS), gbt]).fit(train)

    auc = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC"
    ).evaluate(modelo.transform(test))

    metricas["auc_prediccion_ausencia"] = round(auc, 4)
    print(f"  AUC prediciendo SI FALTA la renta: {auc:.4f}")

    # Que variables lo explican. Se agregan las importancias de cada bloque one-hot,
    # porque una provincia suelta no dice nada; el bloque "provincia" si.
    importancias = modelo.stages[-1].featureImportances.toArray()
    ensamblador = next(e for e in modelo.stages if isinstance(e, VectorAssembler))
    por_variable: dict[str, float] = {}
    posicion = 0
    for nombre in ensamblador.getInputCols():
        atributos = [
            a for a in modelo.transform(test).schema["features"].metadata.get("ml_attr", {})
            .get("attrs", {}).get("binary", [])
            if a["name"].startswith(nombre)
        ] if nombre.endswith("_ohe") else []
        ancho = max(len(atributos), 1)
        limpio = nombre.replace("_ohe", "")
        por_variable[limpio] = float(importancias[posicion:posicion + ancho].sum())
        posicion += ancho

    top = sorted(por_variable.items(), key=lambda kv: -kv[1])[:6]
    metricas["variables_que_explican_la_ausencia"] = {k: round(v, 4) for k, v in top}
    print("\n  Que variables explican que falte:")
    for nombre, peso in top:
        print(f"    {nombre:<26} {peso:.3f}")

    if auc > 0.70:
        veredicto = (
            f"NO es aleatoria (AUC {auc:.3f}). Que a un cliente le falte la renta se "
            "puede predecir a partir del resto de su ficha, asi que el hueco es un dato "
            "en si mismo. Esto justifica guardar el indicador de ausencia como variable "
            "del modelo principal, en vez de taparlo imputando y ya esta."
        )
    elif auc > 0.60:
        veredicto = (
            f"Parcialmente predecible (AUC {auc:.3f}). Hay algo de estructura en quien "
            "tiene la renta sin informar, pero debil. El indicador de ausencia aporta "
            "poco, aunque no estorba."
        )
    else:
        veredicto = (
            f"Compatible con aleatoria (AUC {auc:.3f}). La ausencia no se explica con el "
            "resto de variables, asi que el indicador de ausencia probablemente no aporte "
            "nada y bastaria con imputar."
        )
    metricas["veredicto_ausencia"] = veredicto
    print(f"\n  {veredicto}")


# --------------------------------------------------------------------------------------
# Pregunta 2: mediana o modelo?
# --------------------------------------------------------------------------------------
def comparar_imputaciones(datos, metricas: dict, arboles: int) -> None:
    print("\n--- PREGUNTA 2: imputar con la mediana o con un modelo? ---\n")

    # Solo las filas donde la renta se conoce: son las unicas donde se puede comprobar
    # si el valor imputado habria acertado.
    conocidas = datos.filter(F.col("renta").isNotNull())
    train, test = conocidas.randomSplit([0.8, 0.2], seed=42)
    train.cache()
    test.cache()
    print(f"  Filas con renta conocida: {conocidas.count():,}")

    # --- Estrategia A: la mediana de todos ---------------------------------------------
    # approxQuantile con precision 0,001: sobre millones de filas, calcular la mediana
    # exacta obligaria a un orden global y no compensa para esta decision.
    mediana = train.approxQuantile("renta", [0.5], 0.001)[0]
    evaluador = RegressionEvaluator(labelCol="renta", predictionCol="prediccion")

    rmse_mediana = evaluador.evaluate(
        test.withColumn("prediccion", F.lit(mediana)),
        {evaluador.metricName: "rmse"},
    )
    mae_mediana = evaluador.evaluate(
        test.withColumn("prediccion", F.lit(mediana)),
        {evaluador.metricName: "mae"},
    )
    print(f"  A) Mediana global ({mediana:,.0f} EUR)")
    print(f"       RMSE {rmse_mediana:>12,.0f}   MAE {mae_mediana:>12,.0f}")

    # --- Estrategia B: predecir la renta de cada cliente --------------------------------
    regresor = GBTRegressor(
        featuresCol="features", labelCol="renta",
        maxIter=arboles, maxDepth=6, maxBins=256, seed=42,
    )
    modelo = Pipeline(
        stages=[*_etapas(CATEGORICAS, [n for n in NUMERICAS]), regresor]
    ).fit(train)

    predicho = modelo.transform(test).withColumnRenamed("prediction", "prediccion")
    rmse_modelo = evaluador.evaluate(predicho, {evaluador.metricName: "rmse"})
    mae_modelo = evaluador.evaluate(predicho, {evaluador.metricName: "mae"})
    print(f"  B) Modelo (GBT sobre provincia, edad, segmento, canal...)")
    print(f"       RMSE {rmse_modelo:>12,.0f}   MAE {mae_modelo:>12,.0f}")

    metricas["imputacion"] = {
        "mediana_eur": round(mediana, 2),
        "rmse_mediana": round(rmse_mediana, 2),
        "mae_mediana": round(mae_mediana, 2),
        "rmse_modelo": round(rmse_modelo, 2),
        "mae_modelo": round(mae_modelo, 2),
        "mejora_rmse_pct": round(100 * (1 - rmse_modelo / rmse_mediana), 2),
        "mejora_mae_pct": round(100 * (1 - mae_modelo / mae_mediana), 2),
    }

    mejora = metricas["imputacion"]["mejora_mae_pct"]
    print(f"\n  El modelo reduce el error absoluto medio un {mejora:.1f}%.")

    if mejora > 15:
        conclusion = (
            "Merece la pena imputar con modelo: la renta de un cliente es bastante "
            "predecible a partir de su provincia y su perfil, y la mediana global se "
            "queda muy lejos para los extremos."
        )
    elif mejora > 5:
        conclusion = (
            "El modelo gana, pero poco. Hay que sopesar si compensa la complejidad de "
            "mantener un segundo modelo solo para tapar un hueco."
        )
    else:
        conclusion = (
            "No compensa. El modelo apenas mejora a la mediana, asi que la mediana con "
            "su indicador de ausencia es la opcion correcta: mas simple, mas robusta y "
            "sin un modelo extra que mantener y que se puede desajustar sin avisar."
        )
    metricas["conclusion_imputacion"] = conclusion
    print(f"  {conclusion}")

    # Un aviso que conviene dejar por escrito: el error es enorme en terminos absolutos
    # comparado con la propia renta. Imputar renta es tapar un hueco, no adivinar el dato.
    renta_media = train.agg(F.avg("renta")).collect()[0][0]
    metricas["imputacion"]["mae_modelo_sobre_renta_media_pct"] = round(
        100 * mae_modelo / renta_media, 1
    )
    print(f"\n  Aviso: incluso el mejor error medio es el "
          f"{metricas['imputacion']['mae_modelo_sobre_renta_media_pct']:.0f}% de la renta "
          f"media ({renta_media:,.0f} EUR).")
    print("  Imputar es tapar un hueco de forma razonable, no adivinar el dato real.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analisis de la renta ausente")
    parser.add_argument("--arboles", type=int, default=30)
    parser.add_argument("--muestra", type=float, default=0.15,
                        help="Fraccion del dataset a usar. 0.15 de 13,6M son ~2M de "
                             "filas, de sobra para estimar un AUC y un RMSE.")
    args = parser.parse_args()

    spark = crear_sesion("analisis-renta")
    spark.sparkContext.setLogLevel("ERROR")
    metricas: dict = {"muestra": args.muestra}

    print("\n=== ANALISIS: la renta ausente ===")

    datos = preparar(spark)
    if args.muestra < 1.0:
        datos = datos.sample(fraction=args.muestra, seed=42)
    datos = datos.cache()
    metricas["filas_analizadas"] = datos.count()
    print(f"\n  Filas analizadas: {metricas['filas_analizadas']:,}")

    with cronometro("diagnostico_ausencia", metricas):
        diagnosticar_ausencia(datos, metricas, args.arboles)

    with cronometro("comparar_imputaciones", metricas):
        comparar_imputaciones(datos, metricas, args.arboles)

    guardar_metricas("analisis_renta.json", metricas)
    spark.stop()


if __name__ == "__main__":
    main()
