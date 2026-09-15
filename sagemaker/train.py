"""
Script de entrenamiento empaquetado con el contrato de Amazon SageMaker.

IMPORTANTE, Y DICHO SIN ADORNOS: este script esta escrito siguiendo el contrato que espera
SageMaker, pero NO se ha ejecutado en AWS. No hay cuenta, no hay factura y no hay run que
ensenar. Lo que demuestra es que el pipeline esta empaquetado de forma que podria lanzarse
en la nube sin reescribirlo, no que se haya lanzado.

--------------------------------------------------------------------------------------
Que es "el contrato de SageMaker"
--------------------------------------------------------------------------------------
SageMaker no importa tu codigo como una libreria: arranca un contenedor, copia los datos
en unas rutas fijas, y ejecuta tu script como un programa normal. El acuerdo es:

  * los datos de entrada aparecen en la ruta que indica la variable de entorno
    SM_CHANNEL_<nombre del canal>,
  * el modelo entrenado hay que dejarlo en SM_MODEL_DIR, y SageMaker lo sube solo a S3
    cuando el script termina,
  * cualquier otro artefacto (metricas, graficos) va en SM_OUTPUT_DATA_DIR,
  * los hiperparametros llegan como argumentos de linea de comandos.

Respetar eso es todo lo que hace falta para que un script local sea lanzable en la nube.
Por eso las rutas se leen de variables de entorno con un valor por defecto local: el mismo
fichero corre en el portatil y correria en SageMaker.

--------------------------------------------------------------------------------------
Como se lanzaria
--------------------------------------------------------------------------------------
Ver sagemaker/README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# El codigo del pipeline viaja junto al script en el mismo paquete de codigo fuente.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.classification import GBTClassifier  # noqa: E402
from pyspark.ml.evaluation import BinaryClassificationEvaluator  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from churn import CATEGORICAS, NUMERICAS, PRODUCTOS, a_formato_largo  # noqa: E402
from common import crear_sesion, etapas_preprocesado, marcar_faltantes  # noqa: E402


def rutas_sagemaker() -> dict[str, Path]:
    """Lee el contrato de rutas, con valores por defecto para ejecutar en local."""
    raiz = Path(__file__).resolve().parent.parent
    return {
        "entrada": Path(os.environ.get("SM_CHANNEL_TRAIN", raiz / "data" / "features")),
        "modelo": Path(os.environ.get("SM_MODEL_DIR", raiz / "outputs" / "modelo")),
        "salida": Path(os.environ.get("SM_OUTPUT_DATA_DIR", raiz / "outputs")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    # SageMaker pasa los hiperparametros como --nombre valor. Los nombres van en ingles
    # porque es lo que espera quien lea la configuracion del job en la consola de AWS.
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--validation-month", default="2016-05-28")
    args = parser.parse_args()

    rutas = rutas_sagemaker()
    rutas["modelo"].mkdir(parents=True, exist_ok=True)
    rutas["salida"].mkdir(parents=True, exist_ok=True)

    print(f"Entrada: {rutas['entrada']}")
    print(f"Modelo:  {rutas['modelo']}")
    print(f"Hiperparametros: {vars(args)}")

    spark = crear_sesion("sagemaker-entrenamiento")
    spark.sparkContext.setLogLevel("ERROR")

    feats = spark.read.parquet(str(rutas["entrada"]))
    corte = feats.filter(F.col("fecha_dato") == args.validation_month).select(
        F.first("mes_idx").alias("idx")
    ).collect()[0]["idx"]

    largo = marcar_faltantes(a_formato_largo(feats), NUMERICAS)
    entrenamiento = largo.filter(
        (F.col("mes_idx") < corte) & (F.col("mes_idx") >= corte - args.train_months)
    )
    validacion = largo.filter(F.col("mes_idx") == corte)

    etapas, col_features = etapas_preprocesado(
        categoricas=CATEGORICAS,
        numericas=NUMERICAS,
        columnas_extra=[f"prev_{p}" for p in PRODUCTOS],
    )
    gbt = GBTClassifier(
        featuresCol=col_features, labelCol="label",
        maxIter=args.max_iter, maxDepth=args.max_depth,
        stepSize=args.step_size, maxBins=256, seed=42,
    )
    modelo = Pipeline(stages=[*etapas, gbt]).fit(entrenamiento)

    predicciones = modelo.transform(validacion)
    metricas = {
        "auc": BinaryClassificationEvaluator(
            labelCol="label", metricName="areaUnderROC"
        ).evaluate(predicciones),
        "average_precision": BinaryClassificationEvaluator(
            labelCol="label", metricName="areaUnderPR"
        ).evaluate(predicciones),
        "base_rate": predicciones.agg(F.avg("label")).collect()[0][0],
    }

    # Formato de metricas que SageMaker sabe leer para comparar jobs entre si.
    for nombre, valor in metricas.items():
        print(f"{nombre}: {valor:.6f};")

    modelo.write().overwrite().save(str(rutas["modelo"] / "pipeline"))
    (rutas["salida"] / "metrics.json").write_text(
        json.dumps(metricas, indent=2), encoding="utf-8"
    )
    print(f"\nModelo guardado en {rutas['modelo'] / 'pipeline'}")

    spark.stop()


if __name__ == "__main__":
    main()
