"""
FASE 3 -- Recomendacion: que producto ofrecer a cada cliente el mes que viene.

Se comparan tres enfoques sobre el mismo split temporal y con la misma metrica, porque
la pregunta interesante no es "cuanto puntua mi modelo" sino "cuanto mejora respecto a
lo que ya se podria hacer sin modelo":

  1. POPULARIDAD  -- recomienda a todo el mundo los productos que mas se contratan,
                     quitando los que el cliente ya tiene. Sin personalizar nada.
  2. ALS          -- filtrado colaborativo clasico: "clientes parecidos a ti contrataron
                     esto". Es el recomendador de libro de texto.
  3. BOSQUE       -- clasificador multiclase sobre el comportamiento del cliente
                     (antiguedad, renta, productos que ya tiene, actividad reciente).

    python src/recommend.py

--------------------------------------------------------------------------------------
Por que un bosque aleatorio y no gradient boosting
--------------------------------------------------------------------------------------
El GBTClassifier de Spark solo resuelve problemas binarios. Aqui hay 24 productos que
compiten entre si, asi que hace falta un clasificador multiclase, y el bosque aleatorio
es el equivalente mas cercano dentro de Spark ML. En la Fase 4, donde el problema si es
binario (se da de baja o no), si se usa gradient boosting.

--------------------------------------------------------------------------------------
Metrica: MAP@7
--------------------------------------------------------------------------------------
La misma de la competicion original, para poder situar el resultado. Premia acertar y
premia mas acertar arriba en la lista: si el equipo comercial solo va a mencionar dos o
tres productos en la llamada, importa que los buenos vayan primero.

Solo entran en la media los clientes que contrataron algo ese mes. A quien no contrato
nada no se le puede medir el acierto, y meterlo como cero hundiria la metrica de forma
uniforme para los tres enfoques sin aportar informacion.
"""

from __future__ import annotations

import argparse

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, StringType

from common import (
    DATA_EXPORT,
    DATA_FEATURES,
    MES_VALIDACION,
    NOMBRES_PRODUCTO,
    crear_sesion,
    cronometro,
    etapas_preprocesado,
    guardar_metricas,
    marcar_faltantes,
)

K = 7  # longitud de la lista recomendada

CATEGORICAS = ["prev_segmento", "prev_canal_entrada", "prev_sexo", "prev_tiprel_1mes",
               "prev_nomprov", "prev_ind_empleado", "prev_indresi"]
NUMERICAS = ["prev_age", "prev_antiguedad", "prev_renta", "prev_ind_actividad_cliente",
             "prev_ind_nuevo", "n_prod_prev", "altas_3m", "bajas_3m", "meses_observados",
             "mes_del_ano"]

PRODUCTOS = list(NOMBRES_PRODUCTO)


# --------------------------------------------------------------------------------------
# Metrica
# --------------------------------------------------------------------------------------
def average_precision(predichos: list[str], reales: list[str], k: int = K) -> float:
    """AP@k de un solo cliente.

    Recorre la lista recomendada de arriba abajo; cada acierto suma la precision
    acumulada hasta esa posicion. Se divide entre el minimo de aciertos posibles, para
    no penalizar a quien contrato mas de k productos.
    """
    if not reales:
        return 0.0
    conjunto_real = set(reales)
    aciertos = 0
    suma = 0.0
    for posicion, producto in enumerate(predichos[:k], start=1):
        if producto in conjunto_real:
            aciertos += 1
            suma += aciertos / posicion
    return suma / min(len(conjunto_real), k)


def _verificar_metrica() -> None:
    """Tres casos calculados a mano. Si la metrica se rompe, se rompe aqui y no en silencio."""
    # Acierto en la primera posicion, un solo producto real -> AP = 1/1 / 1 = 1.0
    assert abs(average_precision(["a", "b", "c"], ["a"]) - 1.0) < 1e-9
    # Acierto solo en la tercera posicion -> AP = (1/3) / 1 = 0.333...
    assert abs(average_precision(["x", "y", "a"], ["a"]) - 1 / 3) < 1e-9
    # Dos reales, aciertos en posiciones 1 y 3 -> ((1/1) + (2/3)) / 2 = 0.8333...
    assert abs(average_precision(["a", "x", "b"], ["a", "b"]) - (1 + 2 / 3) / 2) < 1e-9
    # Ningun acierto -> 0
    assert average_precision(["x", "y"], ["a"]) == 0.0


def evaluar(df_predicciones) -> float:
    """MAP@7 sobre un DataFrame con columnas 'recomendados' y 'reales' (arrays)."""
    udf_ap = F.udf(average_precision, DoubleType())
    return (
        df_predicciones.withColumn("ap", udf_ap("recomendados", "reales"))
        .agg(F.avg("ap").alias("map"))
        .collect()[0]["map"]
    )


# --------------------------------------------------------------------------------------
# Preparacion del split
# --------------------------------------------------------------------------------------
def preparar(spark, mes_validacion: str):
    """Divide en entrenamiento (meses anteriores) y validacion (ultimo mes)."""
    feats = spark.read.parquet(str(DATA_FEATURES))
    corte = feats.filter(F.col("fecha_dato") == mes_validacion).select(
        F.first("mes_idx").alias("idx")
    ).collect()[0]["idx"]

    entrenamiento = feats.filter(F.col("mes_idx") < corte)
    validacion = feats.filter(F.col("mes_idx") == corte)
    return entrenamiento, validacion


def altas_reales(df):
    """Una fila por cliente con la lista de productos que contrato de verdad."""
    columnas = F.array(*[
        F.when(F.col(f"alta_{p}") == 1, F.lit(p)).otherwise(F.lit(None))
        for p in PRODUCTOS
    ])
    return (
        df.select("ncodpers", F.array_compact(columnas).alias("reales"))
        .filter(F.size("reales") > 0)
    )


def ya_tiene(df):
    """Una fila por cliente con la lista de productos que ya tenia el mes anterior."""
    columnas = F.array(*[
        F.when(F.col(f"prev_{p}") == 1, F.lit(p)).otherwise(F.lit(None))
        for p in PRODUCTOS
    ])
    return df.select("ncodpers", F.array_compact(columnas).alias("poseidos"))


# --------------------------------------------------------------------------------------
# Enfoque 1: popularidad
# --------------------------------------------------------------------------------------
def ranking_popularidad(entrenamiento) -> list[str]:
    """Productos ordenados por numero de altas en el historico de entrenamiento."""
    totales = entrenamiento.agg(
        *[F.sum(f"alta_{p}").alias(p) for p in PRODUCTOS]
    ).collect()[0].asDict()
    return [p for p, _ in sorted(totales.items(), key=lambda kv: -(kv[1] or 0))]


def predecir_popularidad(validacion, ranking: list[str]):
    """Misma lista para todos, quitando lo que cada cliente ya tiene."""
    orden = F.array(*[F.lit(p) for p in ranking])
    return (
        ya_tiene(validacion)
        .withColumn("recomendados", F.slice(F.array_except(orden, F.col("poseidos")), 1, K))
        .select("ncodpers", "recomendados")
    )


# --------------------------------------------------------------------------------------
# Enfoque 2: ALS (filtrado colaborativo)
# --------------------------------------------------------------------------------------
def predecir_als(spark, entrenamiento, validacion, semilla: int = 42):
    """Entrena ALS sobre el historico de altas y recomienda al mes de validacion.

    Se usa feedback implicito: no hay valoraciones, solo el hecho de que un cliente
    contrato un producto. implicitPrefs=True le dice a ALS que interprete la ausencia
    como "no observado" y no como "no le gusta".
    """
    indice = {p: i for i, p in enumerate(PRODUCTOS)}

    # Historico de altas en formato largo: (cliente, producto, 1)
    tripletas = None
    for p in PRODUCTOS:
        parcial = (
            entrenamiento.filter(F.col(f"alta_{p}") == 1)
            .select("ncodpers", F.lit(indice[p]).alias("item"))
        )
        tripletas = parcial if tripletas is None else tripletas.union(parcial)

    tripletas = (
        tripletas.groupBy("ncodpers", "item")
        .agg(F.count("*").cast("double").alias("rating"))
        .cache()
    )

    als = ALS(
        userCol="ncodpers", itemCol="item", ratingCol="rating",
        implicitPrefs=True, rank=16, maxIter=10, regParam=0.1, alpha=20.0,
        coldStartStrategy="drop", seed=semilla, nonnegative=True,
    )
    modelo = als.fit(tripletas)

    usuarios = validacion.select("ncodpers").distinct()
    recomendaciones = modelo.recommendForUserSubset(usuarios, len(PRODUCTOS))

    nombres = F.udf(lambda items: [PRODUCTOS[i] for i in items], ArrayType(StringType()))
    recomendaciones = recomendaciones.select(
        "ncodpers",
        nombres(F.col("recommendations.item")).alias("ordenados"),
    )

    salida = (
        ya_tiene(validacion)
        .join(recomendaciones, "ncodpers", "left")
        .withColumn(
            "recomendados",
            F.slice(F.array_except(F.coalesce("ordenados", F.array()), F.col("poseidos")), 1, K),
        )
        .select("ncodpers", "recomendados")
    )
    tripletas.unpersist()
    return salida


# --------------------------------------------------------------------------------------
# Enfoque 3: bosque aleatorio multiclase
# --------------------------------------------------------------------------------------
def _matriz_a_largo(df):
    """Explota cada fila con altas en una fila por producto contratado.

    Es la forma de convertir "este cliente contrato tarjeta y recibos" en dos ejemplos de
    entrenamiento, uno por clase.
    """
    columnas = F.array(*[
        F.when(F.col(f"alta_{p}") == 1, F.lit(p)).otherwise(F.lit(None))
        for p in PRODUCTOS
    ])
    return (
        df.withColumn("contratados", F.array_compact(columnas))
        .filter(F.size("contratados") > 0)
        .withColumn("producto", F.explode("contratados"))
        .drop("contratados")
    )


def predecir_bosque(entrenamiento, validacion, arboles: int = 60, profundidad: int = 10):
    """Clasificador multiclase: dado el cliente, que producto es mas probable que contrate."""
    train_largo = marcar_faltantes(_matriz_a_largo(entrenamiento), NUMERICAS)
    validacion = marcar_faltantes(validacion, NUMERICAS)

    etapas, col_features = etapas_preprocesado(
        categoricas=CATEGORICAS,
        numericas=NUMERICAS,
        # Que productos ya tiene el cliente: son 0/1, no necesitan imputacion.
        columnas_extra=[f"prev_{p}" for p in PRODUCTOS],
    )
    indexador_objetivo = StringIndexer(
        inputCol="producto", outputCol="label", handleInvalid="keep"
    )
    bosque = RandomForestClassifier(
        featuresCol=col_features, labelCol="label",
        numTrees=arboles, maxDepth=profundidad,
        # maxBins debe superar la cardinalidad de la categorica mas grande
        # (canal_entrada ronda los 160 valores distintos).
        maxBins=256, seed=42, subsamplingRate=0.7,
    )

    pipeline = Pipeline(stages=[indexador_objetivo, *etapas, bosque])
    modelo = pipeline.fit(train_largo)

    # Las etiquetas que aprendio el indexador, en el orden de la clase 0, 1, 2...
    etiquetas = next(
        e.labels for e in modelo.stages if hasattr(e, "labels") and e.getOutputCol() == "label"
    )

    predicciones = modelo.transform(validacion)

    def ordenar(probabilidades) -> list[str]:
        pares = sorted(zip(etiquetas, probabilidades.toArray()), key=lambda kv: -kv[1])
        return [p for p, _ in pares]

    udf_ordenar = F.udf(ordenar, ArrayType(StringType()))

    # La propension de compra del cliente es la del mejor producto que todavia no tiene:
    # es lo que decide si merece la pena llamarle, independientemente de cual sea.
    udf_maximo = F.udf(lambda v: float(max(v.toArray())), DoubleType())

    return (
        predicciones.withColumn("ordenados", udf_ordenar("probability"))
        .withColumn("prob_compra", udf_maximo("probability"))
        .join(ya_tiene(validacion), "ncodpers")
        .withColumn(
            "recomendados",
            F.slice(F.array_except("ordenados", "poseidos"), 1, K),
        )
        .select("ncodpers", "recomendados", "prob_compra")
    ), modelo


# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Compara tres enfoques de recomendacion")
    parser.add_argument("--mes-validacion", default=MES_VALIDACION)
    parser.add_argument("--arboles", type=int, default=60)
    parser.add_argument("--saltar-als", action="store_true")
    parser.add_argument("--exportar", action="store_true",
                        help="Guarda las probabilidades de compra para la Fase 5 y la demo")
    args = parser.parse_args()

    _verificar_metrica()
    print("Metrica MAP@7 verificada contra casos calculados a mano.")

    spark = crear_sesion("recomendacion")
    spark.sparkContext.setLogLevel("ERROR")
    metricas: dict = {"mes_validacion": args.mes_validacion, "k": K}

    print("\n=== FASE 3: recomendacion ===\n")
    entrenamiento, validacion = preparar(spark, args.mes_validacion)
    entrenamiento.cache()
    validacion.cache()

    reales = altas_reales(validacion).cache()
    n_evaluables = reales.count()
    n_validacion = validacion.count()
    metricas["clientes_validacion"] = n_validacion
    metricas["clientes_con_alta"] = n_evaluables
    metricas["pct_clientes_con_alta"] = round(100 * n_evaluables / n_validacion, 2)
    print(f"  Clientes en {args.mes_validacion}: {n_validacion:,}")
    print(f"  De ellos contratan algo: {n_evaluables:,} "
          f"({metricas['pct_clientes_con_alta']}%)\n")

    resultados: dict[str, float] = {}

    # --- 1. Popularidad ----------------------------------------------------------------
    with cronometro("popularidad", metricas):
        ranking = ranking_popularidad(entrenamiento)
        pred = predecir_popularidad(validacion, ranking)
        resultados["popularidad"] = evaluar(pred.join(reales, "ncodpers"))
    metricas["ranking_popularidad"] = ranking[:K]
    print(f"  Popularidad      MAP@7 = {resultados['popularidad']:.5f}")

    # --- 2. ALS ------------------------------------------------------------------------
    if not args.saltar_als:
        with cronometro("als", metricas):
            pred = predecir_als(spark, entrenamiento, validacion)
            resultados["als"] = evaluar(pred.join(reales, "ncodpers"))
        print(f"  ALS              MAP@7 = {resultados['als']:.5f}")

    # --- 3. Bosque ---------------------------------------------------------------------
    with cronometro("bosque", metricas):
        pred, modelo = predecir_bosque(entrenamiento, validacion, arboles=args.arboles)
        pred.cache()
        resultados["bosque"] = evaluar(pred.join(reales, "ncodpers"))
    print(f"  Bosque aleatorio MAP@7 = {resultados['bosque']:.5f}")

    # --- Exportacion para la Fase 5 y la demo ------------------------------------------
    if args.exportar:
        DATA_EXPORT.mkdir(parents=True, exist_ok=True)
        # contrato = 1 si el cliente contrato algo de verdad ese mes. Es la etiqueta que
        # necesita la curva de captura para saber si la llamada habria acertado.
        etiquetas = validacion.select(
            "ncodpers", (F.col("n_altas") > 0).cast("int").alias("contrato")
        )
        (
            pred.join(etiquetas, "ncodpers")
            .select("ncodpers", "prob_compra", "contrato", "recomendados")
            .write.mode("overwrite")
            .parquet(str(DATA_EXPORT / "prob_compra"))
        )
        print(f"  Exportado a {DATA_EXPORT / 'prob_compra'}")

    # --- Resumen -----------------------------------------------------------------------
    metricas["map7"] = {k: round(v, 5) for k, v in resultados.items()}
    mejor = max(resultados, key=resultados.get)
    metricas["mejor"] = mejor
    metricas["mejora_sobre_popularidad"] = round(
        resultados[mejor] / resultados["popularidad"], 2
    ) if resultados["popularidad"] else None

    print("\n--- Comparativa ---")
    for nombre, valor in sorted(resultados.items(), key=lambda kv: -kv[1]):
        marca = "  <-- mejor" if nombre == mejor else ""
        print(f"  {nombre:<18} {valor:.5f}{marca}")
    if metricas["mejora_sobre_popularidad"]:
        print(f"\n  El mejor enfoque multiplica por "
              f"{metricas['mejora_sobre_popularidad']} el MAP de recomendar "
              f"siempre lo mas popular.")

    guardar_metricas("recommend_metrics.json", metricas)
    spark.stop()


if __name__ == "__main__":
    main()
