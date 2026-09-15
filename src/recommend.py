"""
FASE 3 -- Recomendacion: que producto ofrecer a cada cliente el mes que viene.

No es "entreno un modelo y reporto su metrica". Son cinco enfoques de complejidad
creciente sobre el mismo split temporal y con la misma metrica, porque la pregunta que
importa no es cuanto puntua un modelo, sino cuanto aporta cada pieza que se le anade:

  1. POPULARIDAD  Recomendar a todo el mundo lo que mas se contrata, quitando lo que ya
                  tiene. Sin personalizar. Es el suelo: cualquier modelo que no lo bata
                  no merece existir.

  2. ALS          Filtrado colaborativo: "clientes parecidos a ti contrataron esto". No
                  mira ni una sola caracteristica del cliente, solo el patron de quien
                  contrata que.

  3. BOSQUE       Bosque aleatorio sobre el comportamiento: antiguedad, renta, segmento,
                  que productos tiene ya, movimiento reciente. Lo contrario del anterior:
                  todo caracteristicas, ninguna senal colaborativa.

  4. XGBOOST      El mismo problema con gradient boosting distribuido. Es el estandar de
                  facto en datos tabulares y aqui sirve para separar dos preguntas que
                  suelen confundirse: cuanto gano por cambiar de algoritmo, y cuanto por
                  cambiar de informacion.

  5. HIBRIDO      XGBoost + los vectores latentes que aprendio el ALS como variables de
                  entrada. Es el que contesta a la pregunta buena.

    python src/recommend.py

--------------------------------------------------------------------------------------
Por que el hibrido, y por que es la parte interesante
--------------------------------------------------------------------------------------
Comparar ALS contra el bosque y quedarse con el ganador es una falsa disyuntiva: no
capturan lo mismo. El ALS sabe algo que las caracteristicas del cliente no dicen -- que
ciertos productos se contratan juntos, y que hay grupos de clientes que se comportan
igual sin parecerse en la ficha. El bosque sabe algo que el ALS ignora -- que un
universitario de 22 anos y un autonomo de 50 no quieren lo mismo.

Asi que en vez de elegir, se encadenan: el ALS comprime a cada cliente en 16 numeros que
resumen su posicion en el espacio de "quien contrata que", y esos 16 numeros entran como
variables mas en el ranker supervisado.

Esto no es un invento: es la arquitectura estandar de los sistemas de recomendacion
grandes (generacion de candidatos + reordenacion), y es la razon por la que en produccion
casi nunca se usa filtrado colaborativo a secas.

Efecto secundario util: los clientes que el ALS nunca vio quedan con esos 16 valores
ausentes, y el preprocesado los marca con su indicador de "faltaba". El modelo aprende
solo a desconfiar de la senal colaborativa justo en los clientes donde no existe.

--------------------------------------------------------------------------------------
Por que XGBoost y no el GBTClassifier de Spark
--------------------------------------------------------------------------------------
El GBTClassifier de Spark solo resuelve problemas binarios. Aqui hay 24 productos que
compiten entre si, o sea multiclase, asi que queda descartado de entrada. Las opciones
dentro de Spark ML eran el bosque aleatorio (enfoque 3) o envolver 24 clasificadores
binarios con OneVsRest, que multiplica por 24 el tiempo de entrenamiento.

XGBoost soporta multiclase de forma nativa con objective="multi:softprob", y su paquete
trae un estimador que distribuye el entrenamiento sobre Spark. En la Fase 4, donde el
problema si es binario, se usa el GBTClassifier nativo.

--------------------------------------------------------------------------------------
Metrica: MAP@7
--------------------------------------------------------------------------------------
La misma de la competicion original, para poder situar el resultado. Premia acertar, y
premia mas acertar arriba: si el comercial solo va a mencionar dos o tres productos en la
llamada, importa que los buenos vayan primero.

Solo entran en la media los clientes que contrataron algo ese mes. A quien no contrato
nada no se le puede medir el acierto, y meterlo como cero hundiria la metrica por igual
en los cinco enfoques sin aportar informacion.
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

K = 7        # longitud de la lista recomendada
RANK_ALS = 16  # dimension del espacio latente del ALS

CATEGORICAS = ["prev_segmento", "prev_canal_entrada", "prev_sexo", "prev_tiprel_1mes",
               "prev_nomprov", "prev_ind_empleado", "prev_indresi"]
NUMERICAS = ["prev_age", "prev_antiguedad", "prev_renta", "prev_ind_actividad_cliente",
             "prev_ind_nuevo", "n_prod_prev", "altas_3m", "bajas_3m", "meses_observados",
             "mes_del_ano"]

PRODUCTOS = list(NOMBRES_PRODUCTO)
COLS_ALS = [f"als_f{i}" for i in range(RANK_ALS)]


# --------------------------------------------------------------------------------------
# Metrica
# --------------------------------------------------------------------------------------
def average_precision(predichos: list[str], reales: list[str], k: int = K) -> float:
    """AP@k de un solo cliente.

    Recorre la lista recomendada de arriba abajo; cada acierto suma la precision
    acumulada hasta esa posicion. Se divide entre el minimo de aciertos posibles, para no
    penalizar a quien contrato mas de k productos.
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
    """Casos calculados a mano. Si la metrica se rompe, se rompe aqui y no en silencio."""
    # Acierto en la primera posicion, un solo producto real -> AP = (1/1) / 1 = 1.0
    assert abs(average_precision(["a", "b", "c"], ["a"]) - 1.0) < 1e-9
    # Acierto solo en la tercera posicion -> AP = (1/3) / 1 = 0.333...
    assert abs(average_precision(["x", "y", "a"], ["a"]) - 1 / 3) < 1e-9
    # Dos reales, aciertos en las posiciones 1 y 3 -> ((1/1) + (2/3)) / 2 = 0.8333...
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
    return feats.filter(F.col("mes_idx") < corte), feats.filter(F.col("mes_idx") == corte)


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
# 1. Popularidad
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
# 2. ALS (filtrado colaborativo)
# --------------------------------------------------------------------------------------
def entrenar_als(entrenamiento, rank: int = RANK_ALS, semilla: int = 42):
    """Entrena ALS sobre el historico de altas y devuelve el modelo.

    Feedback implicito: no hay valoraciones, solo el hecho de que alguien contrato algo.
    implicitPrefs=True le dice a ALS que interprete la ausencia como "no observado" y no
    como "no le gusta", que es la lectura correcta -- que un cliente no tenga hipoteca no
    significa que la rechazara.
    """
    indice = {p: i for i, p in enumerate(PRODUCTOS)}
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
    )

    als = ALS(
        userCol="ncodpers", itemCol="item", ratingCol="rating",
        implicitPrefs=True, rank=rank, maxIter=10, regParam=0.1, alpha=20.0,
        coldStartStrategy="drop", seed=semilla, nonnegative=True,
    )
    return als.fit(tripletas)


def predecir_als(modelo, validacion):
    """Lista recomendada usando solo el filtrado colaborativo."""
    usuarios = validacion.select("ncodpers").distinct()
    recomendaciones = modelo.recommendForUserSubset(usuarios, len(PRODUCTOS))

    nombres = F.udf(lambda items: [PRODUCTOS[i] for i in items], ArrayType(StringType()))
    recomendaciones = recomendaciones.select(
        "ncodpers", nombres(F.col("recommendations.item")).alias("ordenados")
    )
    return (
        ya_tiene(validacion)
        .join(recomendaciones, "ncodpers", "left")
        .withColumn(
            "recomendados",
            F.slice(
                F.array_except(F.coalesce("ordenados", F.array()), F.col("poseidos")), 1, K
            ),
        )
        .select("ncodpers", "recomendados")
    )


def factores_usuario(modelo, rank: int = RANK_ALS):
    """Saca los vectores latentes de cliente del ALS como columnas sueltas.

    Cada cliente queda resumido en `rank` numeros que codifican su posicion en el espacio
    de "quien contrata que". Son los que se inyectan en el ranker del enfoque hibrido.
    """
    return modelo.userFactors.select(
        F.col("id").alias("ncodpers"),
        *[F.col("features")[i].alias(f"als_f{i}") for i in range(rank)],
    )


# --------------------------------------------------------------------------------------
# 3, 4 y 5. Rankers supervisados
# --------------------------------------------------------------------------------------
def _a_formato_largo(df):
    """Explota cada fila con altas en una fila por producto contratado.

    Es lo que convierte "este cliente contrato tarjeta y recibos" en dos ejemplos de
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


def _clasificador(algoritmo: str, n_clases: int, arboles: int, profundidad: int):
    """Devuelve el estimador multiclase pedido, ya configurado."""
    if algoritmo == "bosque":
        return RandomForestClassifier(
            featuresCol="features", labelCol="label",
            numTrees=arboles, maxDepth=profundidad,
            # maxBins debe superar la cardinalidad de la categorica mas grande;
            # canal_entrada ronda los 160 valores distintos.
            maxBins=256, seed=42, subsamplingRate=0.7,
        )
    if algoritmo == "xgboost":
        from xgboost.spark import SparkXGBClassifier

        return SparkXGBClassifier(
            features_col="features", label_col="label",
            # num_workers=1 porque Spark corre en local[*]: hay un unico ejecutor. En un
            # cluster de verdad se sube al numero de ejecutores y XGBoost reparte el
            # entrenamiento entre ellos.
            num_workers=1, device="cpu",
            # Nota: SparkXGBClassifier NO deja fijar objective ni num_class a mano --
            # lanza un ValueError. Los deduce del numero de clases distintas que ve en
            # la columna de etiqueta, y con 24 productos elige multi:softprob solo.
            n_estimators=arboles, max_depth=profundidad,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=42, missing=float("nan"),
        )
    raise ValueError(f"Algoritmo desconocido: {algoritmo}")


def predecir_ranker(entrenamiento, validacion, algoritmo: str = "xgboost",
                    factores=None, arboles: int = 80, profundidad: int = 8):
    """Entrena un clasificador multiclase y devuelve la lista recomendada por cliente.

    Si se pasan `factores` (los vectores latentes del ALS), se unen a los datos y se
    tratan como variables numericas mas: eso es el enfoque hibrido. Los clientes que el
    ALS no llego a ver quedan con esas columnas nulas, y el preprocesado comun las imputa
    marcandolas como ausentes -- el modelo aprende solo a desconfiar de la senal
    colaborativa donde no la hay.
    """
    numericas = list(NUMERICAS)
    if factores is not None:
        entrenamiento = entrenamiento.join(factores, "ncodpers", "left")
        validacion = validacion.join(factores, "ncodpers", "left")
        numericas += COLS_ALS

    train_largo = marcar_faltantes(_a_formato_largo(entrenamiento), numericas)
    validacion = marcar_faltantes(validacion, numericas)

    etapas, col_features = etapas_preprocesado(
        categoricas=CATEGORICAS,
        numericas=numericas,
        # Que productos tiene ya: son 0/1, no necesitan imputacion.
        columnas_extra=[f"prev_{p}" for p in PRODUCTOS],
    )
    indexador_objetivo = StringIndexer(
        inputCol="producto", outputCol="label", handleInvalid="keep"
    )

    # El numero de clases se lee de los datos: no todos los productos tienen por que
    # aparecer como alta en la ventana de entrenamiento.
    n_clases = train_largo.select("producto").distinct().count()

    modelo = Pipeline(stages=[
        indexador_objetivo,
        *etapas,
        _clasificador(algoritmo, n_clases, arboles, profundidad),
    ]).fit(train_largo)

    etiquetas = next(
        e.labels for e in modelo.stages
        if hasattr(e, "labels") and e.getOutputCol() == "label"
    )

    def ordenar(probabilidades) -> list[str]:
        pares = sorted(zip(etiquetas, probabilidades.toArray()), key=lambda kv: -kv[1])
        return [p for p, _ in pares]

    udf_ordenar = F.udf(ordenar, ArrayType(StringType()))
    # La propension de compra del cliente es la del mejor producto que aun no tiene: es
    # lo que decide si merece la pena llamarle, independientemente de cual sea.
    udf_maximo = F.udf(lambda v: float(max(v.toArray())), DoubleType())

    predicciones = modelo.transform(validacion)
    return (
        predicciones.withColumn("ordenados", udf_ordenar("probability"))
        .withColumn("prob_compra", udf_maximo("probability"))
        .join(ya_tiene(validacion), "ncodpers")
        .withColumn("recomendados", F.slice(F.array_except("ordenados", "poseidos"), 1, K))
        .select("ncodpers", "recomendados", "prob_compra")
    )


# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Compara cinco enfoques de recomendacion")
    parser.add_argument("--mes-validacion", default=MES_VALIDACION)
    parser.add_argument("--arboles", type=int, default=80)
    parser.add_argument("--profundidad", type=int, default=8)
    parser.add_argument("--enfoques", default="todos",
                        help="Lista separada por comas, o 'todos'. Utiles: "
                             "popularidad,als,bosque,xgboost,hibrido")
    parser.add_argument("--exportar", action="store_true",
                        help="Guarda las probabilidades del mejor enfoque para la Fase 5")
    args = parser.parse_args()

    pedidos = (
        ["popularidad", "als", "bosque", "xgboost", "hibrido"]
        if args.enfoques == "todos"
        else [e.strip() for e in args.enfoques.split(",")]
    )

    _verificar_metrica()
    print("Metrica MAP@7 verificada contra casos calculados a mano.")

    spark = crear_sesion("recomendacion")
    spark.sparkContext.setLogLevel("ERROR")
    metricas: dict = {"mes_validacion": args.mes_validacion, "k": K,
                      "rank_als": RANK_ALS, "enfoques": pedidos}

    print("\n=== FASE 3: recomendacion ===\n")
    entrenamiento, validacion = preparar(spark, args.mes_validacion)
    entrenamiento.cache()
    validacion.cache()

    reales = altas_reales(validacion).cache()
    n_evaluables, n_validacion = reales.count(), validacion.count()
    metricas.update({
        "clientes_validacion": n_validacion,
        "clientes_con_alta": n_evaluables,
        "pct_clientes_con_alta": round(100 * n_evaluables / n_validacion, 2),
    })
    print(f"  Clientes en {args.mes_validacion}: {n_validacion:,}")
    print(f"  De ellos contratan algo: {n_evaluables:,} "
          f"({metricas['pct_clientes_con_alta']}%)\n")

    resultados: dict[str, float] = {}
    predicciones_por_enfoque: dict = {}
    modelo_als = None
    factores = None

    # --- 1. Popularidad ----------------------------------------------------------------
    if "popularidad" in pedidos:
        with cronometro("popularidad", metricas):
            ranking = ranking_popularidad(entrenamiento)
            pred = predecir_popularidad(validacion, ranking)
            resultados["popularidad"] = evaluar(pred.join(reales, "ncodpers"))
        metricas["ranking_popularidad"] = ranking[:K]
        print(f"  1. Popularidad        MAP@7 = {resultados['popularidad']:.5f}")

    # --- 2. ALS ------------------------------------------------------------------------
    # Se entrena si se pide el enfoque ALS o si hace falta para el hibrido.
    if "als" in pedidos or "hibrido" in pedidos:
        with cronometro("als_entrenamiento", metricas):
            modelo_als = entrenar_als(entrenamiento)
            factores = factores_usuario(modelo_als).cache()
            n_factores = factores.count()
        metricas["clientes_con_factores_als"] = n_factores
        print(f"     (ALS entrenado: {n_factores:,} clientes con vector latente)")

    if "als" in pedidos:
        with cronometro("als_evaluacion", metricas):
            pred = predecir_als(modelo_als, validacion)
            resultados["als"] = evaluar(pred.join(reales, "ncodpers"))
        print(f"  2. ALS                MAP@7 = {resultados['als']:.5f}")

    # --- 3, 4, 5. Rankers supervisados -------------------------------------------------
    configuraciones = [
        ("bosque", "3. Bosque aleatorio", "bosque", None),
        ("xgboost", "4. XGBoost", "xgboost", None),
        ("hibrido", "5. XGBoost + ALS", "xgboost", "factores"),
    ]
    for clave, titulo, algoritmo, usa_factores in configuraciones:
        if clave not in pedidos:
            continue
        with cronometro(clave, metricas):
            pred = predecir_ranker(
                entrenamiento, validacion, algoritmo=algoritmo,
                factores=factores if usa_factores else None,
                arboles=args.arboles, profundidad=args.profundidad,
            ).cache()
            resultados[clave] = evaluar(pred.join(reales, "ncodpers"))
            predicciones_por_enfoque[clave] = pred
        print(f"  {titulo:<21} MAP@7 = {resultados[clave]:.5f}")

    # --- Resumen -----------------------------------------------------------------------
    metricas["map7"] = {k: round(v, 5) for k, v in resultados.items()}
    mejor = max(resultados, key=resultados.get)
    metricas["mejor"] = mejor

    print("\n--- Comparativa ---")
    base = resultados.get("popularidad")
    for nombre, valor in sorted(resultados.items(), key=lambda kv: -kv[1]):
        relativo = f"  ({valor / base:.2f}x sobre popularidad)" if base else ""
        marca = "   <-- mejor" if nombre == mejor else ""
        print(f"  {nombre:<14} {valor:.5f}{relativo}{marca}")

    # Lo que aporta la senal colaborativa, aislada: misma arquitectura, mismos
    # hiperparametros, la unica diferencia son los 16 numeros del ALS.
    if "xgboost" in resultados and "hibrido" in resultados:
        aporte = resultados["hibrido"] - resultados["xgboost"]
        metricas["aporte_factores_als"] = round(aporte, 5)
        metricas["aporte_factores_als_pct"] = round(
            100 * aporte / resultados["xgboost"], 2
        )
        signo = "aporta" if aporte > 0 else "NO aporta"
        print(f"\n  Anadir los vectores del ALS a XGBoost {signo}: "
              f"{aporte:+.5f} MAP ({metricas['aporte_factores_als_pct']:+.1f}%)")

    # --- Exportacion para la Fase 5 y la demo ------------------------------------------
    if args.exportar and predicciones_por_enfoque:
        exportable = predicciones_por_enfoque.get(mejor) or next(
            iter(predicciones_por_enfoque.values())
        )
        DATA_EXPORT.mkdir(parents=True, exist_ok=True)
        # contrato = 1 si el cliente contrato algo de verdad ese mes. Es la etiqueta que
        # necesita la curva de captura para saber si la llamada habria acertado.
        etiquetas = validacion.select(
            "ncodpers", (F.col("n_altas") > 0).cast("int").alias("contrato")
        )
        (
            exportable.join(etiquetas, "ncodpers")
            .select("ncodpers", "prob_compra", "contrato", "recomendados")
            .write.mode("overwrite")
            .parquet(str(DATA_EXPORT / "prob_compra"))
        )
        metricas["enfoque_exportado"] = mejor
        print(f"\n  Exportado el enfoque '{mejor}' a {DATA_EXPORT / 'prob_compra'}")

    guardar_metricas("recommend_metrics.json", metricas)
    spark.stop()


if __name__ == "__main__":
    main()
