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
"""

from __future__ import annotations

import argparse

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

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

PRODUCTOS = list(NOMBRES_PRODUCTO)

CATEGORICAS = ["prev_segmento", "prev_canal_entrada", "prev_sexo", "prev_tiprel_1mes",
               "prev_nomprov", "producto"]
NUMERICAS = ["prev_age", "prev_antiguedad", "prev_renta", "prev_ind_actividad_cliente",
             "prev_ind_nuevo", "n_prod_prev", "altas_3m", "bajas_3m", "meses_observados",
             "mes_del_ano"]

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
    """
    estructuras = F.array(*[
        F.struct(
            F.lit(p).alias("producto"),
            F.col(f"baja_{p}").cast("double").alias("label"),
            F.col(f"prev_{p}").alias("tenia"),
        )
        for p in PRODUCTOS
    ])
    comunes = [c for c in df.columns
               if not c.startswith(("alta_", "baja_"))]
    return (
        df.select(*comunes, F.explode(estructuras).alias("e"))
        .filter(F.col("e.tenia") == 1)          # solo se puede dar de baja lo que se tiene
        .withColumn("producto", F.col("e.producto"))
        .withColumn("label", F.col("e.label"))
        .drop("e")
    )


def construir_pipeline(arboles: int, profundidad: int) -> Pipeline:
    etapas, col_features = etapas_preprocesado(
        categoricas=CATEGORICAS,
        numericas=NUMERICAS,
        columnas_extra=[f"prev_{p}" for p in PRODUCTOS],
    )
    gbt = GBTClassifier(
        featuresCol=col_features, labelCol="label",
        maxIter=arboles, maxDepth=profundidad, maxBins=256,
        stepSize=0.1, subsamplingRate=0.8, seed=42,
    )
    return Pipeline(stages=[*etapas, gbt])


def umbral_optimo(predicciones, coste=COSTE_CONTACTO, valor=VALOR_RETENCION,
                  exito=TASA_EXITO_RETENCION):
    """Elige el umbral que maximiza el valor esperado de la campana de retencion.

    Por cada cliente al que se llama se paga el coste del contacto. Si de verdad se iba
    a dar de baja y la llamada funciona, se salva el margen. El optimo no es 0,5: con
    una llamada barata y un margen alto, compensa llamar a bastante gente dudosa.
    """
    extraer = F.udf(lambda v: float(v[1]), DoubleType())
    puntuadas = predicciones.select(
        extraer("probability").alias("p"), F.col("label")
    ).cache()

    resultados = []
    for umbral in [i / 100 for i in range(1, 100)]:
        fila = puntuadas.agg(
            F.sum(F.when(F.col("p") >= umbral, 1).otherwise(0)).alias("contactados"),
            F.sum(F.when((F.col("p") >= umbral) & (F.col("label") == 1), 1)
                  .otherwise(0)).alias("aciertos"),
        ).collect()[0]
        contactados = fila["contactados"] or 0
        aciertos = fila["aciertos"] or 0
        beneficio = aciertos * exito * valor - contactados * coste
        resultados.append({
            "umbral": umbral, "contactados": int(contactados),
            "bajas_capturadas": int(aciertos), "beneficio_estimado": round(beneficio, 2),
        })

    puntuadas.unpersist()
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

    largo = marcar_faltantes(a_formato_largo(feats), NUMERICAS)
    entrenamiento = largo.filter(
        (F.col("mes_idx") < corte) & (F.col("mes_idx") >= corte - args.meses_entrenamiento)
    ).cache()
    validacion = largo.filter(F.col("mes_idx") == corte).cache()

    n_train, n_val = entrenamiento.count(), validacion.count()
    tasa_base = validacion.agg(F.avg("label")).collect()[0][0]
    metricas.update({
        "filas_entrenamiento": n_train,
        "filas_validacion": n_val,
        "tasa_base_bajas": round(tasa_base, 5),
        "meses_entrenamiento": args.meses_entrenamiento,
    })
    print(f"  Entrenamiento: {n_train:,} pares cliente-producto "
          f"({args.meses_entrenamiento} meses previos)")
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

    with cronometro("busqueda_umbral", metricas):
        mejor, curva = umbral_optimo(predicciones)

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
        extraer = F.udf(lambda v: float(v[1]), DoubleType())
        (
            predicciones.select(
                "ncodpers", "producto", extraer("probability").alias("prob_baja")
            )
            .write.mode("overwrite")
            .parquet(str(DATA_EXPORT / "prob_baja"))
        )
        print(f"\n  Probabilidades exportadas a {DATA_EXPORT / 'prob_baja'}")

    guardar_metricas("churn_metrics.json", metricas)
    spark.stop()


if __name__ == "__main__":
    main()
