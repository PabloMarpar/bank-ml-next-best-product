"""
FASE 4 -- Caida de negocio: que cliente esta a punto de dejar un producto.

Mismo dato y mismas variables que la Fase 3, pero la pregunta se invierte: en vez de
"que va a contratar", aqui es "que va a cancelar". Cada fila es un producto que el
cliente SI tenia el mes anterior, y la etiqueta es si lo ha dejado este mes.

    python src/churn.py

--------------------------------------------------------------------------------------
Por que el split es temporal y nunca aleatorio
--------------------------------------------------------------------------------------
Un split aleatorio repartiria al mismo cliente entre entrenamiento y validacion, y
mezclaria meses: el modelo veria junio al entrenar y luego se le pediria predecir marzo.
Daria una metrica bonita e inservible, porque en produccion nadie tiene datos del futuro.
Aqui se entrena con todos los meses anteriores y se valida sobre el ultimo, que es
exactamente la situacion real.

--------------------------------------------------------------------------------------
Por que la precision media se reporta siempre junto a la tasa base
--------------------------------------------------------------------------------------
Las bajas son raras. Con un 2% de bajas, un modelo que no sirva para nada puede sacar
una exactitud del 98% diciendo siempre "no se da de baja". La cifra que informa es
cuanto mejora la precision media respecto a esa tasa base: eso es el multiplicador real
sobre elegir clientes al azar.

--------------------------------------------------------------------------------------
Por que hace falta calibrar, y por que casi nadie lo hace
--------------------------------------------------------------------------------------
El umbral de la campana se elige maximizando el valor esperado: se compara lo que cuesta
llamar con lo que se gana reteniendo. Ese calculo multiplica por una PROBABILIDAD, asi
que solo tiene sentido si el numero que devuelve el modelo es una probabilidad de verdad
-- si el modelo dice 0,30, de cada cien clientes con ese 0,30 tienen que darse de baja
unos treinta.

Los ensembles de arboles no cumplen eso. Ordenan de maravilla, pero sus salidas estan
sistematicamente comprimidas hacia el centro: promediar muchos arboles hace que sea casi
imposible que el conjunto diga 0,01 o 0,99. Sirven para rankear y mienten como magnitud.

La solucion es una regresion isotonica: un tercer tramo de datos, posterior al
entrenamiento y anterior a la validacion, sirve para aprender la funcion monotona que
traduce la puntuacion cruda en probabilidad observada.

Lo que cambia al calibrar es el umbral economico y, con el, a cuanta gente se llama y
cuanto rinde la campana. Por eso se reporta el Brier score, que si mide el error como
magnitud, y se comparan los dos umbrales. El AUC casi no se mueve, porque solo mira el
orden y la calibracion es creciente.

Casi, no nada, y el matiz importa: la regresion isotonica es creciente pero NO
estrictamente creciente. Es una funcion escalonada, asi que muchas puntuaciones crudas
distintas acaban en el mismo escalon y quedan empatadas entre si. Esos empates mueven el
AUC ligeramente. Si hiciera falta preservarlo exacto habria que usar escalado de Platt --
ajustar una logistica sobre la puntuacion--, que si es estrictamente creciente; a cambio
impone una forma sigmoidea que se ajusta peor cuando la distorsion no tiene esa forma.
Aqui interesa mas acertar la magnitud que conservar el AUC hasta el cuarto decimal, asi
que la isotonica es la eleccion correcta -- pero el script imprime la deriva del AUC y el
numero de escalones para que se vea, en lugar de afirmar que no cambia nada.
"""

from __future__ import annotations

import argparse

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.regression import IsotonicRegression
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from common import (
    DATA_EXPORT,
    agrupar_categorias_raras,
    DATA_FEATURES,
    MES_VALIDACION,
    NOMBRES_PRODUCTO,
    crear_sesion,
    cronometro,
    etapas_preprocesado,
    guardar_metricas,
    marcar_faltantes,
)

PRODUCTOS = list(NOMBRES_PRODUCTO)

CATEGORICAS = ["prev_segmento", "prev_canal_entrada", "prev_sexo", "prev_tiprel_1mes",
               "prev_nomprov", "producto"]
NUMERICAS = ["prev_age", "prev_antiguedad", "prev_renta", "prev_ind_actividad_cliente",
             "prev_ind_nuevo", "n_prod_prev", "altas_3m", "bajas_3m", "meses_observados",
             "mes_del_ano",
             # Meses (de los ultimos 6) que el cliente lleva con ESTE producto. La anade
             # a_formato_largo al apilar, sacandola del struct de cada producto.
             "tenencia_producto"]

# Supuestos del caso de negocio, explicitos para poder discutirlos.
# No son cifras reales de ningun banco: son un orden de magnitud razonable que sirve
# para elegir un umbral con criterio en vez de dejarlo en 0,5 por inercia.
COSTE_CONTACTO = 5.0      # lo que cuesta una llamada de retencion
VALOR_RETENCION = 120.0   # margen anual que se salva si la retencion funciona
TASA_EXITO_RETENCION = 0.30  # solo una de cada tres llamadas evita la baja


def a_formato_largo(df):
    """Una fila por (cliente, mes, producto que tenia), con la etiqueta de baja.

    Se parte de la tabla ancha y se apila: es lo que convierte 24 columnas de objetivo
    en un unico problema binario con el producto como variable mas.

    Detalle de rendimiento que no es cosmetico: se proyectan PRIMERO solo las columnas
    que el modelo va a usar, y se explota despues. Al reves -- explotar la tabla entera y
    quedarse luego con lo que interesa -- se multiplican por 24 unas 130 columnas de las
    que el modelo usa una fraccion, y el resultado no cabe en memoria. La primera version
    de esto cacheaba la tabla completa y la JVM se ahogaba: 86% del contenedor ocupado y
    la CPU desplomada recolectando basura.
    """
    # Lo unico que hace falta aguas abajo: identificadores, el corte temporal, las
    # variables del modelo y las 24 columnas prev_ de producto.
    columnas_modelo = (
        ["ncodpers", "fecha_dato", "mes_idx"]
        + [c for c in CATEGORICAS if c != "producto"]
        + NUMERICAS
        + [f"prev_{p}" for p in PRODUCTOS]
    )
    presentes = [c for c in dict.fromkeys(columnas_modelo) if c in df.columns]

    estructuras = F.array(*[
        F.struct(
            F.lit(p).alias("producto"),
            F.col(f"baja_{p}").cast("double").alias("label"),
            F.col(f"prev_{p}").alias("tenia"),
            # La antiguedad viaja dentro del struct para que, al apilar, cada fila se
            # quede con la de SU producto. Es la variable mas informativa del modelo:
            # una tarjeta de hace un mes y una de hace dos anos se cancelan a ritmos
            # completamente distintos.
            F.coalesce(F.col(f"tenencia_{p}"), F.lit(0)).cast("double").alias("tenencia"),
        )
        for p in PRODUCTOS
    ])

    return (
        df.select(*presentes, F.explode(estructuras).alias("e"))
        .filter(F.col("e.tenia") == 1)          # solo se puede dar de baja lo que se tiene
        .withColumn("producto", F.col("e.producto"))
        .withColumn("label", F.col("e.label"))
        .withColumn("tenencia_producto", F.col("e.tenencia"))
        .drop("e")
    )


def calibrar(predicciones_calibracion, predicciones_validacion):
    """Ajusta una regresion isotonica y la aplica a las predicciones de validacion.

    La isotonica aprende una funcion escalonada creciente que mapea puntuacion cruda ->
    probabilidad observada. Creciente es la clave: no puede reordenar a nadie, solo
    reescalar. Por eso conserva el AUC intacto y solo arregla la magnitud.

    Se ajusta sobre un mes que el modelo NO vio al entrenar. Calibrar sobre los datos de
    entrenamiento daria una curva optimista, porque ahi el modelo ya acerto de mas.
    """
    extraer = F.udf(lambda v: float(v[1]), DoubleType())

    calibracion = predicciones_calibracion.select(
        extraer("probability").alias("p_cruda"), F.col("label")
    )
    isotonica = IsotonicRegression(
        featuresCol="p_cruda", labelCol="label", predictionCol="p_calibrada"
    ).fit(calibracion.withColumn("p_cruda", F.col("p_cruda").cast("double")))

    validacion = predicciones_validacion.withColumn(
        "p_cruda", extraer("probability").cast("double")
    )
    return isotonica.transform(validacion)


def brier(df, columna_prob: str) -> float:
    """Error cuadratico medio entre probabilidad predicha y resultado real.

    A diferencia del AUC, el Brier score si castiga decir 0,9 cuando la respuesta es 0.
    Es la metrica que detecta una mala calibracion; el AUC es ciego a ella.
    """
    return df.select(
        F.avg(F.pow(F.col(columna_prob) - F.col("label"), 2)).alias("brier")
    ).collect()[0]["brier"]


def tabla_fiabilidad(df, columna_prob: str, tramos: int = 10) -> list[dict]:
    """Compara, por tramos de probabilidad, lo prometido contra lo ocurrido.

    Es la forma honesta de mirar una calibracion: en el tramo "el modelo dice 30%",
    cuantos se dieron de baja de verdad.
    """
    filas = (
        df.withColumn("tramo", F.least(F.floor(F.col(columna_prob) * tramos),
                                       F.lit(tramos - 1)))
        .groupBy("tramo")
        .agg(
            F.count("*").alias("n"),
            F.avg(columna_prob).alias("prometido"),
            F.avg("label").alias("observado"),
        )
        .orderBy("tramo")
        .collect()
    )
    return [
        {
            "tramo": f"{int(f['tramo']) / tramos:.1f}-{(int(f['tramo']) + 1) / tramos:.1f}",
            "n": int(f["n"]),
            "prometido": round(float(f["prometido"]), 4),
            "observado": round(float(f["observado"]), 4),
        }
        for f in filas
    ]


def construir_pipeline(arboles: int, profundidad: int) -> Pipeline:
    etapas, col_features = etapas_preprocesado(
        categoricas=CATEGORICAS,
        numericas=NUMERICAS,
        columnas_extra=[f"prev_{p}" for p in PRODUCTOS],
    )
    gbt = GBTClassifier(
        featuresCol=col_features, labelCol="label",
        # maxBins 40 y no 256: tras recortar la cola de canal_entrada, la categorica mas
        # grande tiene 31 valores. El coste de construir los histogramas de cada nodo
        # crece con maxBins, asi que ponerlo justo por encima de lo necesario es una
        # de las palancas mas directas para acelerar un arbol en Spark.
        maxIter=arboles, maxDepth=profundidad, maxBins=40,
        stepSize=0.1, subsamplingRate=0.8, seed=42,
    )
    return Pipeline(stages=[*etapas, gbt])


def umbral_optimo(predicciones, columna_prob: str = "p_calibrada",
                  coste=COSTE_CONTACTO, valor=VALOR_RETENCION,
                  exito=TASA_EXITO_RETENCION):
    """Elige el umbral que maximiza el valor esperado de la campana de retencion.

    Por cada cliente al que se llama se paga el coste del contacto. Si de verdad se iba
    a dar de baja y la llamada funciona, se salva el margen. El optimo no es 0,5: con
    una llamada barata y un margen alto, compensa llamar a bastante gente dudosa.

    Como se calcula, y por que asi:

    La version directa es un bucle que prueba 99 umbrales y, para cada uno, cuenta cuanta
    gente queda por encima. El problema es que cada cuenta es una accion de Spark, o sea
    99 trabajos completos recorriendo la tabla entera. Medido sobre un dataset de juguete
    ya tardaba casi dos minutos; sobre los 13,6M de filas reales seria inviable.

    Aqui se recorre la tabla UNA vez: se agrupa por la probabilidad redondeada a dos
    decimales, lo que deja como mucho 101 filas, y el acumulado se calcula en el driver
    sobre esas 101 filas. Es el patron de siempre -- agregar en el cluster, rematar en
    local -- y convierte 99 trabajos en uno.
    """
    # Un unico recorrido de la tabla: histograma de probabilidad contra etiqueta.
    histograma = (
        predicciones.select(F.col(columna_prob).alias("p"), F.col("label"))
        .withColumn("bucket", F.round(F.col("p"), 2))
        .groupBy("bucket")
        .agg(F.count("*").alias("n"), F.sum("label").alias("bajas"))
        .collect()
    )

    # A partir de aqui son ~101 filas en memoria del driver: calculo trivial.
    por_bucket = {
        round(float(f["bucket"]), 2): (int(f["n"]), int(f["bajas"] or 0))
        for f in histograma
    }

    resultados = []
    for umbral in [i / 100 for i in range(1, 100)]:
        # Contactados = todos los buckets con probabilidad >= umbral.
        contactados = sum(n for b, (n, _) in por_bucket.items() if b >= umbral)
        aciertos = sum(bajas for b, (_, bajas) in por_bucket.items() if b >= umbral)
        beneficio = aciertos * exito * valor - contactados * coste
        resultados.append({
            "umbral": umbral, "contactados": int(contactados),
            "bajas_capturadas": int(aciertos), "beneficio_estimado": round(beneficio, 2),
        })

    mejor = max(resultados, key=lambda r: r["beneficio_estimado"])
    return mejor, resultados


def main() -> None:
    parser = argparse.ArgumentParser(description="Modelo de baja de producto")
    parser.add_argument("--mes-validacion", default=MES_VALIDACION)
    parser.add_argument("--meses-entrenamiento", type=int, default=6,
                        help="Cuantos meses previos usar. Los mas recientes se parecen "
                             "mas al mes a predecir y el entrenamiento es mas rapido.")
    parser.add_argument("--arboles", type=int, default=40)
    parser.add_argument("--profundidad", type=int, default=6)
    parser.add_argument("--exportar", action="store_true",
                        help="Guarda las probabilidades del mes de validacion para la demo")
    args = parser.parse_args()

    spark = crear_sesion("bajas")
    spark.sparkContext.setLogLevel("ERROR")
    metricas: dict = {
        "mes_validacion": args.mes_validacion,
        "supuestos_negocio": {
            "coste_contacto_eur": COSTE_CONTACTO,
            "valor_retencion_eur": VALOR_RETENCION,
            "tasa_exito_retencion": TASA_EXITO_RETENCION,
        },
    }

    print("\n=== FASE 4: caida de negocio ===\n")

    feats = spark.read.parquet(str(DATA_FEATURES))
    corte = feats.filter(F.col("fecha_dato") == args.mes_validacion).select(
        F.first("mes_idx").alias("idx")
    ).collect()[0]["idx"]

    # Recorte de la cola larga antes de nada. prev_canal_entrada trae 161 valores
    # distintos; los que aparecen un punado de veces no dan senal y obligan a subir
    # maxBins, que es justo lo que ralentiza el entrenamiento.
    for columna, tope in (("prev_canal_entrada", 30), ("prev_nomprov", 30)):
        feats = agrupar_categorias_raras(feats, columna, maximo=tope)

    largo = marcar_faltantes(a_formato_largo(feats), NUMERICAS)

    # Split temporal a TRES tramos, no dos. El mes inmediatamente anterior al de
    # validacion se reserva para calibrar: el modelo no lo ve al entrenar, asi que sus
    # predicciones ahi son honestas y sirven para aprender la correccion. Calibrar con
    # los datos de entrenamiento daria una curva optimista, porque ahi el modelo ya
    # acerto de mas.
    mes_calibracion = corte - 1
    entrenamiento = largo.filter(
        (F.col("mes_idx") < mes_calibracion)
        & (F.col("mes_idx") >= mes_calibracion - args.meses_entrenamiento)
    ).cache()
    calibracion = largo.filter(F.col("mes_idx") == mes_calibracion).cache()
    validacion = largo.filter(F.col("mes_idx") == corte).cache()

    n_train, n_cal, n_val = entrenamiento.count(), calibracion.count(), validacion.count()
    tasa_base = validacion.agg(F.avg("label")).collect()[0][0]
    metricas.update({
        "filas_entrenamiento": n_train,
        "filas_calibracion": n_cal,
        "filas_validacion": n_val,
        "tasa_base_bajas": round(tasa_base, 5),
        "meses_entrenamiento": args.meses_entrenamiento,
    })
    print(f"  Entrenamiento: {n_train:,} pares cliente-producto "
          f"({args.meses_entrenamiento} meses)")
    print(f"  Calibracion:   {n_cal:,} pares (el mes anterior al de validacion)")
    print(f"  Validacion:    {n_val:,} pares en {args.mes_validacion}")
    print(f"  Tasa base de bajas: {100 * tasa_base:.2f}%\n")

    with cronometro("entrenamiento_gbt", metricas):
        modelo = construir_pipeline(args.arboles, args.profundidad).fit(entrenamiento)

    with cronometro("prediccion", metricas):
        predicciones = modelo.transform(validacion).cache()
        auc = BinaryClassificationEvaluator(
            labelCol="label", metricName="areaUnderROC"
        ).evaluate(predicciones)
        ap = BinaryClassificationEvaluator(
            labelCol="label", metricName="areaUnderPR"
        ).evaluate(predicciones)

    metricas["auc"] = round(auc, 4)
    metricas["precision_media"] = round(ap, 4)
    metricas["mejora_sobre_tasa_base"] = round(ap / tasa_base, 1)

    print(f"  AUC:                      {auc:.4f}")
    print(f"  Precision media (AP):     {ap:.4f}")
    print(f"  Tasa base:                {tasa_base:.4f}")
    print(f"  Mejora sobre el azar:     {metricas['mejora_sobre_tasa_base']}x\n")

    # --- Calibracion -------------------------------------------------------------------
    with cronometro("calibracion", metricas):
        pred_calibracion = modelo.transform(calibracion)
        predicciones = calibrar(pred_calibracion, predicciones).cache()

        brier_crudo = brier(predicciones, "p_cruda")
        brier_calibrado = brier(predicciones, "p_calibrada")

        # El AUC se recalcula sobre la probabilidad calibrada para comprobar que apenas
        # se mueve. Apenas, no nada: ver la nota sobre los empates mas abajo.
        auc_calibrado = BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="p_calibrada", metricName="areaUnderROC"
        ).evaluate(predicciones)

    metricas["calibracion"] = {
        "brier_crudo": round(brier_crudo, 6),
        "brier_calibrado": round(brier_calibrado, 6),
        "mejora_brier_pct": round(100 * (1 - brier_calibrado / brier_crudo), 2),
        "auc_antes": round(auc, 4),
        "auc_despues": round(auc_calibrado, 4),
        "fiabilidad_cruda": tabla_fiabilidad(predicciones, "p_cruda"),
        "fiabilidad_calibrada": tabla_fiabilidad(predicciones, "p_calibrada"),
    }

    # Cuantos valores distintos deja la isotonica: cada escalon agrupa muchas
    # puntuaciones crudas en una sola probabilidad, y ahi es donde nacen los empates.
    escalones = predicciones.select("p_calibrada").distinct().count()
    metricas["calibracion"]["escalones_isotonica"] = escalones
    metricas["calibracion"]["deriva_auc"] = round(auc_calibrado - auc, 4)

    print(f"  Brier antes de calibrar:  {brier_crudo:.6f}")
    print(f"  Brier despues:            {brier_calibrado:.6f} "
          f"({metricas['calibracion']['mejora_brier_pct']:+.1f}%)")
    print(f"  AUC antes / despues:      {auc:.4f} / {auc_calibrado:.4f} "
          f"({auc_calibrado - auc:+.4f})")
    print(f"  La isotonica deja {escalones} valores distintos: es una funcion escalonada,")
    print(f"  y cada escalon empata entre si a los clientes que cae dentro. Por eso el AUC")
    print(f"  se mueve un poco en vez de quedarse clavado.\n")

    # --- Umbral: con y sin calibrar ----------------------------------------------------
    with cronometro("busqueda_umbral", metricas):
        mejor, curva = umbral_optimo(predicciones, "p_calibrada")
        mejor_crudo, _ = umbral_optimo(predicciones, "p_cruda")

    metricas["umbral_sin_calibrar"] = mejor_crudo
    print(f"  Umbral sobre la probabilidad SIN calibrar: {mejor_crudo['umbral']:.2f} "
          f"-> {mejor_crudo['contactados']:,} contactos")

    metricas["umbral_elegido"] = mejor
    metricas["curva_umbral"] = curva[::5]  # una de cada cinco, para no inflar el JSON
    if mejor["contactados"] == 0:
        # Caso degenerado, y merece decirse en voz alta: si el optimo es no llamar a
        # nadie, el modelo no distingue lo suficiente como para que la campana pague el
        # coste de las llamadas. Es un resultado valido y hay que reportarlo, no
        # esconderlo bajando el umbral a mano hasta que salga un numero bonito.
        metricas["campana_rentable"] = False
        print("  El umbral optimo es no contactar a nadie: con estos supuestos de coste,")
        print("  el modelo no separa lo bastante como para que la campana se pague.")
    else:
        metricas["campana_rentable"] = True
        print(f"  Umbral optimo por valor de negocio: {mejor['umbral']:.2f}")
        print(f"    contactar a {mejor['contactados']:,} pares cliente-producto")
        print(f"    captura {mejor['bajas_capturadas']:,} de las bajas reales "
              f"({100 * mejor['bajas_capturadas'] / max(n_val * tasa_base, 1):.1f}% del total)")
        print(f"    valor esperado: {mejor['beneficio_estimado']:,.0f} EUR")
        por_inercia = next(r for r in curva if r["umbral"] >= 0.5)
        print(f"    frente al umbral 0,50 por inercia: contactaria a "
              f"{por_inercia['contactados']:,} y rendiria "
              f"{por_inercia['beneficio_estimado']:,.0f} EUR")

    if args.exportar:
        DATA_EXPORT.mkdir(parents=True, exist_ok=True)
        # Se exporta la CALIBRADA: aguas abajo, la Fase 5 la usa para cruzar riesgo con
        # propension y la demo la muestra como porcentaje. En los dos sitios el numero se
        # lee como una probabilidad, asi que tiene que serlo de verdad.
        (
            predicciones.select(
                "ncodpers", "producto",
                F.col("p_calibrada").alias("prob_baja"),
                F.col("p_cruda").alias("prob_baja_sin_calibrar"),
            )
            .write.mode("overwrite")
            .parquet(str(DATA_EXPORT / "prob_baja"))
        )
        print(f"\n  Probabilidades exportadas a {DATA_EXPORT / 'prob_baja'}")

    guardar_metricas("churn_metrics.json", metricas)
    spark.stop()


if __name__ == "__main__":
    main()
