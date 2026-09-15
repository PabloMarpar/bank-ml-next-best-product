"""
FASE 3d -- Learning to rank: entrenar exactamente lo que se mide.

    python src/ranker.py

--------------------------------------------------------------------------------------
El desajuste que arrastraba el proyecto
--------------------------------------------------------------------------------------
Lo que queremos es una LISTA ORDENADA de productos para cada cliente. Pero hasta aqui el
mejor modelo se entrena con una perdida de CLASIFICACION: "de los 24 productos, cual
contrata este cliente". Es una pregunta de una sola respuesta correcta, como un test.

La lista sale ordenada, si -- el modelo devuelve una probabilidad por producto y las
ordenamos nosotros. Pero el modelo nunca fue entrenado para ordenar: fue entrenado para
acertar una casilla.

La diferencia se ve en que le da igual el error. Para la clasificacion, dejar el producto
correcto en la posicion 2 o en la 7 es exactamente el mismo fallo: en los dos casos no ha
marcado la casilla buena. Para MAP@7 son cosas muy distintas, porque premia acertar
arriba. Estamos optimizando una cosa y midiendo otra, y confiar en que la primera arrastre
a la segunda es una suposicion, no un diseno.

Y en el problema real manda la lista: el comercial va a mencionar dos o tres productos en
la llamada, no veinticuatro. Solo importa que los buenos esten arriba.

--------------------------------------------------------------------------------------
Learning to rank: se aprende por grupos, no por ejemplos sueltos
--------------------------------------------------------------------------------------
Es la familia de metodos que optimiza el orden directamente. XGBoost trae varios
objetivos de este tipo; aqui se usa rank:ndcg.

La diferencia clave esta en la unidad de aprendizaje:

    CLASIFICADOR   ve un ejemplo:  (cliente, producto que contrato)
    RANKER         ve un grupo:    (cliente, TODOS sus productos candidatos a la vez)

El ranker aprende comparando productos ENTRE SI dentro del mismo cliente: "para este
cliente, el producto A deberia ir por delante del B". Esa es literalmente la forma del
problema que tenemos.

El grupo se declara con qid_col (query id). El nombre viene de la busqueda web, donde cada
consulta trae su lista de documentos que hay que ordenar. Aqui la consulta es un
cliente-mes y los documentos son los productos.

--------------------------------------------------------------------------------------
Por que hay que muestrear negativos
--------------------------------------------------------------------------------------
Un clasificador multiclase necesita una fila por alta: 561.710 filas. Un ranker necesita
una fila por cada producto CANDIDATO de cada cliente-mes, porque tiene que aprender a
ordenarlos entre si. Con unos 22 productos no poseidos por cliente, eso pasa de 12
millones de filas.

La practica estandar es quedarse con todos los positivos y una muestra de negativos. No es
solo por coste: con 22 negativos por positivo, la inmensa mayoria de las comparaciones son
entre dos productos que el cliente no contrato, y de ese par el modelo no aprende nada.
"""

from __future__ import annotations

import argparse
import json

from pyspark.ml import Pipeline
from pyspark.sql import Window
from pyspark.sql import functions as F

from common import (
    MES_VALIDACION,
    NOMBRES_PRODUCTO,
    OUTPUTS,
    crear_sesion,
    cronometro,
    etapas_preprocesado,
    guardar_metricas,
    marcar_faltantes,
)
from recommend import (
    CATEGORICAS,
    K,
    NUMERICAS,
    altas_reales,
    evaluar,
    preparar,
    ya_tiene,
)

PRODUCTOS = list(NOMBRES_PRODUCTO)


def a_formato_ranking(df, negativos_por_grupo: int, semilla: int = 42):
    """Una fila por (cliente-mes, producto candidato), con etiqueta 1 si lo contrato.

    Solo entran clientes-mes con al menos un alta: un grupo donde ningun producto es
    positivo no ensena nada a un ranker, porque no hay ningun par que ordenar.
    """
    estructuras = F.array(*[
        F.struct(
            F.lit(p).alias("producto"),
            F.col(f"alta_{p}").cast("double").alias("label"),
            F.col(f"prev_{p}").alias("tenia"),
        )
        for p in PRODUCTOS
    ])

    comunes = [c for c in df.columns if not c.startswith(("alta_", "baja_"))]
    largo = (
        df.filter(F.col("n_altas") > 0)
        .select(*comunes, F.explode(estructuras).alias("e"))
        # Un producto que el cliente ya tiene no es candidato: no se le puede dar de alta.
        .filter(F.col("e.tenia") == 0)
        .withColumn("producto", F.col("e.producto"))
        .withColumn("label", F.col("e.label"))
        .drop("e")
    )

    ventana = Window.partitionBy("ncodpers", "mes_idx").orderBy("_aleatorio")
    positivos = largo.filter(F.col("label") == 1)
    negativos = (
        largo.filter(F.col("label") == 0)
        .withColumn("_aleatorio", F.rand(semilla))
        .withColumn("_orden", F.row_number().over(ventana))
        .filter(F.col("_orden") <= negativos_por_grupo)
        .drop("_aleatorio", "_orden")
    )
    return positivos.unionByName(negativos)


def predecir_con_ranker(entrenamiento, validacion, arboles: int, profundidad: int,
                        negativos: int):
    """Entrena SparkXGBRanker con rank:ndcg y devuelve la lista recomendada."""
    from xgboost.spark import SparkXGBRanker

    train = marcar_faltantes(a_formato_ranking(entrenamiento, negativos), NUMERICAS)

    # El identificador de grupo tiene que ser un entero. Ademas XGBoost exige que las
    # filas de un mismo grupo lleguen JUNTAS al ejecutor, de ahi el repartition por qid
    # mas el orden dentro de cada particion.
    train = train.withColumn(
        "qid", (F.col("ncodpers").cast("long") * 100 + F.col("mes_idx")).cast("long")
    )
    train = train.repartition("qid").sortWithinPartitions("qid")

    # El producto entra como variable: es lo que permite al modelo aprender que ciertos
    # productos van antes que otros segun el perfil del cliente.
    categoricas = list(CATEGORICAS) + ["producto"]
    etapas, col_features = etapas_preprocesado(
        categoricas=categoricas,
        numericas=NUMERICAS,
        columnas_extra=[f"prev_{p}" for p in PRODUCTOS],
    )
    ranker = SparkXGBRanker(
        features_col=col_features, label_col="label", qid_col="qid",
        num_workers=1, device="cpu",
        objective="rank:ndcg",
        n_estimators=arboles, max_depth=profundidad,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        random_state=42, missing=float("nan"),
    )
    modelo = Pipeline(stages=[*etapas, ranker]).fit(train)

    # En validacion se puntua cada uno de los 24 productos para cada cliente.
    val_largo = validacion.withColumn(
        "producto", F.explode(F.array(*[F.lit(p) for p in PRODUCTOS]))
    )
    val_largo = marcar_faltantes(val_largo, NUMERICAS).withColumn(
        "qid", (F.col("ncodpers").cast("long") * 100 + F.col("mes_idx")).cast("long")
    )
    puntuado = modelo.transform(val_largo)

    # Se agrupa por cliente y se ordena por la puntuacion del ranker. array_sort con
    # comparador va dentro de Spark; hacerlo con una UDF de Python seria mucho mas lento
    # sobre 24 filas por cada uno de los ~900.000 clientes.
    ordenados = (
        puntuado.groupBy("ncodpers")
        .agg(F.collect_list(F.struct("prediction", "producto")).alias("pares"))
        .withColumn(
            "ordenados",
            F.expr(
                "transform("
                "  array_sort(pares, (a, b) -> "
                "    CASE WHEN a.prediction > b.prediction THEN -1 "
                "         WHEN a.prediction < b.prediction THEN 1 ELSE 0 END), "
                "  x -> x.producto)"
            ),
        )
        .select("ncodpers", "ordenados")
    )

    return (
        ordenados.join(ya_tiene(validacion), "ncodpers")
        .withColumn("recomendados", F.slice(F.array_except("ordenados", "poseidos"), 1, K))
        .select("ncodpers", "recomendados")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Learning to rank con XGBoost")
    parser.add_argument("--mes-validacion", default=MES_VALIDACION)
    parser.add_argument("--arboles", type=int, default=100)
    parser.add_argument("--profundidad", type=int, default=8)
    parser.add_argument("--negativos", type=int, default=6,
                        help="Negativos muestreados por cliente-mes")
    args = parser.parse_args()

    spark = crear_sesion("ranker")
    spark.sparkContext.setLogLevel("ERROR")
    metricas: dict = {
        "mes_validacion": args.mes_validacion,
        "objetivo": "rank:ndcg",
        "negativos_por_grupo": args.negativos,
        "arboles": args.arboles,
        "profundidad": args.profundidad,
    }

    print("\n=== FASE 3d: learning to rank ===\n")
    entrenamiento, validacion = preparar(spark, args.mes_validacion)
    entrenamiento.cache()
    validacion.cache()
    reales = altas_reales(validacion).cache()
    print(f"  Clientes evaluables: {reales.count():,}\n")

    with cronometro("ranker_ndcg", metricas):
        pred = predecir_con_ranker(
            entrenamiento, validacion, args.arboles, args.profundidad, args.negativos
        )
        puntuacion = evaluar(pred.join(reales, "ncodpers"))

    metricas["map7_ranker"] = round(puntuacion, 5)
    print(f"  XGBoost ranker (rank:ndcg)   MAP@7 = {puntuacion:.5f}")

    # Comparacion contra el clasificador de la Fase 3, que es la pregunta del experimento.
    ruta = OUTPUTS / "recommend_metrics.json"
    if ruta.exists():
        previas = json.loads(ruta.read_text(encoding="utf-8")).get("map7", {})
        if "xgboost" in previas:
            base = previas["xgboost"]
            diferencia = puntuacion - base
            metricas["map7_clasificador"] = base
            metricas["mejora"] = round(diferencia, 5)
            metricas["mejora_pct"] = round(100 * diferencia / base, 2)
            print(f"  XGBoost clasificador         MAP@7 = {base:.5f}")
            print(f"\n  Diferencia: {diferencia:+.5f} ({metricas['mejora_pct']:+.2f}%)")
            metricas["veredicto"] = (
                "Optimizar el orden directamente SI mejora sobre entrenar como "
                "clasificacion. El desajuste entre la perdida y la metrica era real y "
                "tenia coste."
                if diferencia > 0 else
                "Optimizar el orden directamente NO mejora aqui. El clasificador "
                "multiclase ya ordenaba bien al ordenar por probabilidad, y el ranker "
                "paga dos peajes: solo puede entrenar con grupos que tienen algun "
                "positivo, y el muestreo de negativos le quita parte del contexto."
            )
            print(f"  {metricas['veredicto']}")

    guardar_metricas("ranker_metrics.json", metricas)
    spark.stop()


if __name__ == "__main__":
    main()
