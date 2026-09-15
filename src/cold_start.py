"""
ANALISIS -- El cliente que acaba de entrar (arranque en frio).

    python src/cold_start.py

--------------------------------------------------------------------------------------
El agujero que deja el pipeline principal
--------------------------------------------------------------------------------------
La Fase 2 descarta el primer mes de cada cliente: sin mes anterior no hay con que
comparar, asi que no se le pueden calcular altas. Es lo correcto metodologicamente, pero
deja una pregunta de producto sin responder:

    Y al cliente que acaba de abrir su primera cuenta, que le ofrezco?

No es un caso raro. En una cartera que crece, cada mes entra gente nueva, y son
precisamente los clientes donde una recomendacion acertada vale mas -- es el momento en
que se decide si el banco sera el principal o el secundario.

El problema tecnico es que el modelo principal se apoya en variables que esos clientes no
tienen: cuantos productos tienen, cuanto llevan, que han movido en los ultimos tres
meses. Para un cliente con un mes de vida, casi todo eso es cero o nulo. El modelo no
falla ruidosamente: responde, pero apoyandose en variables vacias.

--------------------------------------------------------------------------------------
Que hace este script
--------------------------------------------------------------------------------------
1. Mide cuanta gente hay en cada situacion: cuantos meses de historia tiene cada cliente
   del mes de validacion.

2. Evalua el modelo principal POR TRAMO DE HISTORIA. La hipotesis es que su ventaja sobre
   el baseline se estrecha segun baja la historia disponible, y que en el tramo de los
   recien llegados puede incluso perder.

3. Entrena un modelo alternativo que solo usa lo que SI se sabe de un cliente nuevo:
   edad, provincia, segmento, canal de entrada, si es residente, si esta activo. Ni un
   solo dato de producto ni de comportamiento pasado.

4. Compara los tres (principal, demografico y popularidad) en cada tramo, y construye la
   regla de enrutado: para cada tipo de cliente, cual conviene usar.

El resultado no es "he entrenado otro modelo". Es una regla de decision que dice, para
cada cliente que llega, cual de los tres sistemas debe atenderle.
"""

from __future__ import annotations

import argparse

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, StringType

from common import (
    DATA_FEATURES,
    MES_VALIDACION,
    NOMBRES_PRODUCTO,
    crear_sesion,
    cronometro,
    etapas_preprocesado,
    guardar_metricas,
    marcar_faltantes,
)
from recommend import (
    K,
    altas_reales,
    evaluar,
    preparar,
    ranking_popularidad,
    predecir_popularidad,
    ya_tiene,
    _a_formato_largo,
    _clasificador,
)

PRODUCTOS = list(NOMBRES_PRODUCTO)

# Lo unico que se sabe de alguien que acaba de llegar. Nada de productos, nada de
# comportamiento pasado, nada de contadores de los ultimos tres meses.
CATEGORICAS_FRIAS = ["prev_segmento", "prev_canal_entrada", "prev_sexo", "prev_nomprov",
                     "prev_indresi", "prev_ind_empleado"]
NUMERICAS_FRIAS = ["prev_age", "prev_ind_actividad_cliente", "prev_ind_nuevo",
                   "mes_del_ano"]

# Tramos de historia disponible, en meses observados antes del mes que se predice.
TRAMOS = [
    ("1 mes (recien llegado)", 0, 1),
    ("2-3 meses", 2, 3),
    ("4-6 meses", 4, 6),
    ("7-11 meses", 7, 11),
    ("12+ meses (veterano)", 12, 999),
]


def predecir_demografico(entrenamiento, validacion, arboles: int, profundidad: int):
    """Ranker entrenado SOLO con lo que se sabe de un cliente sin historial."""
    train_largo = marcar_faltantes(_a_formato_largo(entrenamiento), NUMERICAS_FRIAS)
    validacion = marcar_faltantes(validacion, NUMERICAS_FRIAS)

    etapas, col_features = etapas_preprocesado(
        categoricas=CATEGORICAS_FRIAS,
        numericas=NUMERICAS_FRIAS,
        columnas_extra=[],  # deliberadamente vacio: ni una columna de producto
    )
    indexador = StringIndexer(inputCol="producto", outputCol="label",
                              handleInvalid="keep")
    n_clases = train_largo.select("producto").distinct().count()

    modelo = Pipeline(stages=[
        indexador, *etapas,
        _clasificador("xgboost", n_clases, arboles, profundidad),
    ]).fit(train_largo)

    etiquetas = next(
        e.labels for e in modelo.stages
        if hasattr(e, "labels") and e.getOutputCol() == "label"
    )

    def ordenar(probabilidades) -> list[str]:
        pares = sorted(zip(etiquetas, probabilidades.toArray()), key=lambda kv: -kv[1])
        return [p for p, _ in pares]

    udf_ordenar = F.udf(ordenar, ArrayType(StringType()))

    return (
        modelo.transform(validacion)
        .withColumn("ordenados", udf_ordenar("probability"))
        .join(ya_tiene(validacion), "ncodpers")
        .withColumn("recomendados", F.slice(F.array_except("ordenados", "poseidos"), 1, K))
        .select("ncodpers", "recomendados")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Arranque en frio: clientes sin historial")
    parser.add_argument("--mes-validacion", default=MES_VALIDACION)
    parser.add_argument("--arboles", type=int, default=60)
    parser.add_argument("--profundidad", type=int, default=7)
    args = parser.parse_args()

    spark = crear_sesion("cold-start")
    spark.sparkContext.setLogLevel("ERROR")
    metricas: dict = {"mes_validacion": args.mes_validacion}

    print("\n=== ANALISIS: arranque en frio ===\n")

    entrenamiento, validacion = preparar(spark, args.mes_validacion)
    entrenamiento.cache()
    validacion.cache()
    reales = altas_reales(validacion).cache()

    # --- 1. Cuanta gente hay en cada situacion -----------------------------------------
    reparto = (
        validacion.join(reales, "ncodpers")
        .groupBy("meses_observados").count()
        .orderBy("meses_observados")
        .collect()
    )
    total = sum(r["count"] for r in reparto)
    metricas["reparto_por_historia"] = {
        int(r["meses_observados"]): int(r["count"]) for r in reparto
    }

    print("  Clientes evaluables por meses de historia disponible:")
    acumulado = 0
    for r in reparto[:6]:
        acumulado += r["count"]
        print(f"    {int(r['meses_observados']):>2} meses: {r['count']:>8,} "
              f"({100 * r['count'] / total:>5.1f}%)  acumulado {100 * acumulado / total:>5.1f}%")
    poco_historial = sum(r["count"] for r in reparto if r["meses_observados"] <= 3)
    metricas["pct_con_3_meses_o_menos"] = round(100 * poco_historial / total, 2)
    print(f"\n  Con 3 meses de historia o menos: {poco_historial:,} "
          f"({metricas['pct_con_3_meses_o_menos']}% de los evaluables)")

    # --- 2. Los tres sistemas -----------------------------------------------------------
    from recommend import predecir_ranker  # import local para no circular en la cabecera

    with cronometro("modelo_completo", metricas):
        pred_completo = predecir_ranker(
            entrenamiento, validacion, algoritmo="xgboost",
            arboles=args.arboles, profundidad=args.profundidad,
        ).select("ncodpers", "recomendados").cache()

    with cronometro("modelo_demografico", metricas):
        pred_frio = predecir_demografico(
            entrenamiento, validacion, args.arboles, args.profundidad
        ).cache()

    ranking = ranking_popularidad(entrenamiento)
    pred_popular = predecir_popularidad(validacion, ranking).cache()

    # --- 3. Comparativa por tramo -------------------------------------------------------
    historia = validacion.select("ncodpers", "meses_observados")
    base = reales.join(historia, "ncodpers").cache()

    print(f"\n  {'tramo de historia':<24} {'n':>8} {'completo':>10} "
          f"{'demografico':>13} {'popularidad':>13}  mejor")
    filas = []
    for nombre, minimo, maximo in TRAMOS:
        segmento = base.filter(
            (F.col("meses_observados") >= minimo) & (F.col("meses_observados") <= maximo)
        )
        n = segmento.count()
        if n == 0:
            continue
        valores = {
            "completo": evaluar(pred_completo.join(segmento, "ncodpers")),
            "demografico": evaluar(pred_frio.join(segmento, "ncodpers")),
            "popularidad": evaluar(pred_popular.join(segmento, "ncodpers")),
        }
        ganador = max(valores, key=valores.get)
        filas.append({"tramo": nombre, "n": n, "ganador": ganador,
                      **{k: round(v, 5) for k, v in valores.items()}})
        print(f"  {nombre:<24} {n:>8,} {valores['completo']:>10.5f} "
              f"{valores['demografico']:>13.5f} {valores['popularidad']:>13.5f}  {ganador}")

    metricas["por_tramo"] = filas

    # --- 4. La regla de enrutado --------------------------------------------------------
    print("\n  REGLA DE ENRUTADO -- que sistema atiende a cada cliente:")
    reglas = {}
    for fila in filas:
        reglas[fila["tramo"]] = fila["ganador"]
        print(f"    {fila['tramo']:<24} -> {fila['ganador']}")
    metricas["regla_enrutado"] = reglas

    # Cuanto se gana enrutando en vez de usar el modelo completo para todo.
    total_n = sum(f["n"] for f in filas)
    map_completo = sum(f["completo"] * f["n"] for f in filas) / total_n
    map_enrutado = sum(f[f["ganador"]] * f["n"] for f in filas) / total_n
    metricas["map_solo_completo"] = round(map_completo, 5)
    metricas["map_con_enrutado"] = round(map_enrutado, 5)
    metricas["ganancia_enrutado_pct"] = round(
        100 * (map_enrutado / map_completo - 1), 2
    )
    print(f"\n  MAP@7 usando el modelo completo para todos: {map_completo:.5f}")
    print(f"  MAP@7 enrutando segun el tramo:              {map_enrutado:.5f} "
          f"({metricas['ganancia_enrutado_pct']:+.2f}%)")

    if metricas["ganancia_enrutado_pct"] < 0.5:
        nota = (
            "El enrutado apenas aporta: el modelo completo ya se defiende solo en los "
            "clientes con poca historia, probablemente porque las variables demograficas "
            "que usa el modelo frio ya estan dentro de el. Es un resultado util igual: "
            "significa que no hace falta mantener un segundo modelo, y eso es una "
            "decision de arquitectura que conviene tomar con un numero delante."
        )
    else:
        nota = (
            "El enrutado aporta de verdad. Merece la pena mantener el modelo demografico "
            "como via alternativa para los clientes recien llegados, porque el modelo "
            "completo se apoya ahi en variables que estan vacias."
        )
    metricas["conclusion"] = nota
    print(f"\n  {nota}")

    guardar_metricas("cold_start_metrics.json", metricas)
    spark.stop()


if __name__ == "__main__":
    main()
