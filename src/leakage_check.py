"""
Control de fuga de datos: entrenar haciendo trampas a proposito, para medir la trampa.

    python src/leakage_check.py

--------------------------------------------------------------------------------------
Que es una fuga de datos, sin jerga
--------------------------------------------------------------------------------------
Es cuando el modelo tiene acceso, sin que nadie se de cuenta, a informacion que en la
vida real no tendria en el momento de decidir. Es el equivalente a estudiar un examen
con las respuestas delante: se saca un diez, y no significa nada.

En este dataset la fuga esta servida en bandeja. Cada fila trae las 24 columnas de
producto del mes en curso. Si se usan como variables para predecir que contrata el
cliente ESE mes, la columna de la tarjeta de credito ya contiene si contrato la tarjeta.
El modelo no aprende a predecir: aprende a copiar.

--------------------------------------------------------------------------------------
Que hace este script
--------------------------------------------------------------------------------------
Entrena el mismo modelo dos veces sobre exactamente los mismos datos:

  LIMPIO    solo con informacion del mes anterior, que es como se entrena de verdad.
  CON FUGA  anadiendo la columna del producto del mes en curso.

Si la separacion de la Fase 2 esta bien hecha, el modelo con fuga tiene que salir
disparado hacia un AUC cercano a 1. Ese contraste es la prueba: demuestra que la fuga
existe, que seria facilisimo caer en ella, y que el pipeline de verdad no la tiene.

Si el modelo con fuga NO mejora, no hay que celebrarlo: significa que la fuga no se ha
inyectado donde se creia, y que el control no esta midiendo nada.
"""

from __future__ import annotations

import argparse

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F

from common import (
    DATA_FEATURES,
    DATA_PARQUET,
    MES_VALIDACION,
    NOMBRES_PRODUCTO,
    crear_sesion,
    cronometro,
    etapas_preprocesado,
    guardar_metricas,
    marcar_faltantes,
)

PRODUCTOS = list(NOMBRES_PRODUCTO)

# Se prueba sobre un producto concreto y frecuente, para que el experimento sea rapido
# y la interpretacion directa: "predecir si este cliente contrata una cuenta nomina".
PRODUCTO_PRUEBA = "ind_cno_fin_ult1"

CATEGORICAS = ["prev_segmento", "prev_canal_entrada", "prev_sexo", "prev_tiprel_1mes"]
NUMERICAS = ["prev_age", "prev_antiguedad", "prev_renta", "prev_ind_actividad_cliente",
             "n_prod_prev", "altas_3m", "bajas_3m", "meses_observados"]


def entrenar_y_evaluar(entrenamiento, validacion, extra: list[str], etiqueta: str,
                       arboles: int) -> float:
    """Entrena el GBT con las columnas indicadas y devuelve el AUC en validacion."""
    etapas, col_features = etapas_preprocesado(
        categoricas=CATEGORICAS,
        numericas=NUMERICAS,
        columnas_extra=[f"prev_{p}" for p in PRODUCTOS] + extra,
    )
    gbt = GBTClassifier(
        featuresCol=col_features, labelCol="label",
        maxIter=arboles, maxDepth=5, maxBins=256, seed=42,
    )
    modelo = Pipeline(stages=[*etapas, gbt]).fit(entrenamiento)
    auc = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC"
    ).evaluate(modelo.transform(validacion))
    print(f"  {etiqueta:<34} AUC = {auc:.4f}")
    return auc


def main() -> None:
    parser = argparse.ArgumentParser(description="Control de fuga de datos")
    parser.add_argument("--producto", default=PRODUCTO_PRUEBA)
    parser.add_argument("--arboles", type=int, default=20)
    parser.add_argument("--meses", type=int, default=4,
                        help="Meses de entrenamiento; pocos bastan para el contraste")
    args = parser.parse_args()

    spark = crear_sesion("control_fuga")
    spark.sparkContext.setLogLevel("ERROR")

    print("\n=== CONTROL DE FUGA DE DATOS ===\n")
    print(f"  Producto de prueba: {args.producto} "
          f"({NOMBRES_PRODUCTO.get(args.producto, '?')})\n")

    feats = spark.read.parquet(str(DATA_FEATURES))

    # La columna "tramposa": el estado del producto en el propio mes que se predice.
    # Se recupera de la tabla cruda de la Fase 1 y se pega por (cliente, mes).
    crudo = spark.read.parquet(str(DATA_PARQUET)).select(
        "ncodpers", "fecha_dato",
        F.coalesce(F.col(args.producto), F.lit(0)).alias("FUGA_estado_mes_actual"),
    )

    datos = (
        feats.join(crudo, ["ncodpers", "fecha_dato"])
        .withColumn("label", F.col(f"alta_{args.producto}").cast("double"))
    )
    datos = marcar_faltantes(datos, NUMERICAS)

    corte = datos.filter(F.col("fecha_dato") == MES_VALIDACION).select(
        F.first("mes_idx").alias("idx")
    ).collect()[0]["idx"]

    entrenamiento = datos.filter(
        (F.col("mes_idx") < corte) & (F.col("mes_idx") >= corte - args.meses)
    ).cache()
    validacion = datos.filter(F.col("mes_idx") == corte).cache()

    tasa = validacion.agg(F.avg("label")).collect()[0][0]
    print(f"  Filas de entrenamiento: {entrenamiento.count():,}")
    print(f"  Tasa base de altas:     {100 * tasa:.2f}%\n")

    resultados = {}
    with cronometro("modelo_limpio"):
        resultados["auc_limpio"] = entrenar_y_evaluar(
            entrenamiento, validacion, [], "LIMPIO (solo mes anterior)", args.arboles
        )
    with cronometro("modelo_con_fuga"):
        resultados["auc_con_fuga"] = entrenar_y_evaluar(
            entrenamiento, validacion, ["FUGA_estado_mes_actual"],
            "CON FUGA (mes actual incluido)", args.arboles
        )

    salto = resultados["auc_con_fuga"] - resultados["auc_limpio"]
    resultados["salto"] = round(salto, 4)
    resultados["producto"] = args.producto
    resultados["tasa_base"] = round(tasa, 5)

    print(f"\n  Salto de AUC por la fuga: +{salto:.4f}")

    if resultados["auc_con_fuga"] > 0.97 and salto > 0.10:
        veredicto = (
            "CORRECTO. El modelo con fuga se dispara a un AUC casi perfecto, que es "
            "justo lo que se esperaba: la columna del mes actual contiene la respuesta. "
            "Que el modelo limpio se quede muy por debajo confirma que el pipeline real "
            "no esta usando informacion del futuro."
        )
    else:
        veredicto = (
            "REVISAR. El modelo con fuga no se ha disparado como deberia. O la columna "
            "tramposa no se esta inyectando bien, o la definicion del objetivo no es la "
            "que se cree. Hay que mirarlo antes de fiarse de las metricas de las Fases 3 y 4."
        )
    print(f"\n  {veredicto}\n")
    resultados["veredicto"] = veredicto

    guardar_metricas("leakage_check.json", resultados)
    spark.stop()


if __name__ == "__main__":
    main()
