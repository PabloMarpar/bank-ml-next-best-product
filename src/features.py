"""
FASE 2 -- Features y definicion de los objetivos.

Construye una tabla con una fila por (cliente, mes) en la que:

  * TODAS las variables describen el mes anterior (t-1),
  * TODOS los objetivos describen lo que paso en el mes actual (t).

Esa separacion es el corazon del proyecto. Si se mezclan, el modelo aprende la respuesta
en vez de predecirla (ver "La trampa de este dataset" mas abajo).

    python src/features.py

--------------------------------------------------------------------------------------
La trampa de este dataset
--------------------------------------------------------------------------------------
Cada fila del CSV original trae, para un cliente y un mes, las 24 columnas de producto
de ESE mes. Es tentador usarlas como variables para predecir que contrata el cliente ese
mismo mes -- y el modelo saldria casi perfecto, porque la columna ind_tjcr_fin_ult1 de
junio ya contiene si contrato la tarjeta en junio. Eso no es predecir: es leer la
respuesta.

Aqui todas las variables pasan por lag(), asi que describen el estado del cliente al
cerrar el mes anterior, que es exactamente la informacion disponible cuando alguien tiene
que decidir a quien llamar el mes que viene.

En src/leakage_check.py se entrena a proposito con la fuga puesta, para medir cuanto
"mejora" el modelo cuando hace trampas. Ese contraste es la prueba de que la separacion
esta bien hecha.

--------------------------------------------------------------------------------------
Dos detalles de correccion que no son obvios
--------------------------------------------------------------------------------------
1. lag() devuelve la fila anterior del cliente, que no tiene por que ser el mes anterior:
   hay clientes que desaparecen del banco unos meses y vuelven. Si no se comprueba, se
   compararia mayo con enero como si fueran meses consecutivos. Por eso se guarda tambien
   el lag de la fecha y solo se conservan los saltos de exactamente un mes.

2. Un cliente en su primer mes no tiene mes anterior con el que comparar, asi que no se le
   pueden calcular altas: todos sus productos pareceria que los acaba de contratar. Esas
   filas se descartan, y el descarte se cuenta y se reporta.
"""

from __future__ import annotations

import shutil

from pyspark.sql import Window
from pyspark.sql import functions as F

from common import (
    DATA_FEATURES,
    DATA_PARQUET,
    crear_sesion,
    cronometro,
    guardar_metricas,
    productos_de,
)

# Variables del cliente que se arrastran del mes anterior.
ATRIBUTOS_CLIENTE = [
    "age", "antiguedad", "renta", "segmento", "canal_entrada", "sexo",
    "tiprel_1mes", "ind_actividad_cliente", "nomprov", "ind_nuevo", "indresi",
    "ind_empleado",
]


def construir(df):
    """Devuelve la tabla (cliente, mes) con variables de t-1 y objetivos de t."""
    productos = productos_de(df)

    # Indice de mes entero (0 = enero 2015). Permite razonar sobre distancias entre meses
    # con aritmetica normal, que es lo que necesita la ventana deslizante de mas abajo.
    df = df.withColumn(
        "mes_idx",
        (F.year("fecha_dato") - 2015) * 12 + F.month("fecha_dato") - 1,
    )

    w = Window.partitionBy("ncodpers").orderBy("mes_idx")

    # --- Lags -------------------------------------------------------------------------
    lags = {f"prev_{p}": F.lag(p).over(w) for p in productos}
    lags.update({f"prev_{c}": F.lag(c).over(w) for c in ATRIBUTOS_CLIENTE})
    lags["prev_mes_idx"] = F.lag("mes_idx").over(w)
    df = df.withColumns(lags)

    # Detalle 1 del docstring: solo valen los saltos de exactamente un mes.
    df = df.withColumn("mes_consecutivo", F.col("mes_idx") - F.col("prev_mes_idx") == 1)

    # --- Objetivos --------------------------------------------------------------------
    # Un nulo en un producto se trata como "no lo tiene". Es la lectura razonable en este
    # dataset (solo ocurre en nomina y pension, y solo en los primeros meses), pero se
    # decide aqui de forma explicita en vez de heredarlo por accidente de la ingesta.
    objetivos = {}
    for p in productos:
        actual = F.coalesce(F.col(p), F.lit(0))
        anterior = F.coalesce(F.col(f"prev_{p}"), F.lit(0))
        objetivos[f"alta_{p}"] = ((actual == 1) & (anterior == 0)).cast("int")
        objetivos[f"baja_{p}"] = ((actual == 0) & (anterior == 1)).cast("int")
        objetivos[f"prev_{p}"] = anterior
    df = df.withColumns(objetivos)

    # --- Variables derivadas del estado anterior --------------------------------------
    n_prod_prev = sum(F.col(f"prev_{p}") for p in productos)
    n_altas = sum(F.col(f"alta_{p}") for p in productos)
    n_bajas = sum(F.col(f"baja_{p}") for p in productos)
    df = df.withColumns({
        "n_prod_prev": n_prod_prev,
        "n_altas": n_altas,
        "n_bajas": n_bajas,
        "mes_del_ano": F.month("fecha_dato"),
    })

    # Detalle 2: fuera las filas sin mes anterior consecutivo.
    df = df.filter(F.col("mes_consecutivo"))

    # --- Antiguedad de cada producto ---------------------------------------------------
    # Cuantos de los ultimos 6 meses ha tenido el cliente cada producto.
    #
    # Es la variable mas predictiva para las bajas y faltaba en la primera version. Quien
    # acaba de contratar una tarjeta se comporta de forma completamente distinta a quien
    # la tiene desde hace dos anos: las cancelaciones se concentran en los primeros meses.
    # Sin esto, el modelo no podia distinguir los dos casos, porque meses_observados dice
    # cuanto llevamos viendo al CLIENTE, no cuanto lleva el con ESE producto.
    #
    # Se mide sobre una ventana de 6 meses y no sobre todo el historial a proposito: lo
    # que discrimina es "recien contratado o asentado", y a partir de medio ano la
    # distincion entre 6 y 16 meses aporta mucho menos.
    w6 = Window.partitionBy("ncodpers").orderBy("mes_idx").rangeBetween(-5, 0)
    df = df.withColumns({
        f"tenencia_{p}": F.coalesce(F.sum(f"prev_{p}").over(w6), F.lit(0))
        for p in productos
    })

    # --- Actividad reciente: altas y bajas de los 3 meses previos ---------------------
    # rangeBetween sobre mes_idx (no rowsBetween) para que "3 meses" signifique 3 meses
    # de calendario aunque al cliente le falte algun snapshot intermedio.
    w3 = Window.partitionBy("ncodpers").orderBy("mes_idx").rangeBetween(-3, -1)
    df = df.withColumns({
        "altas_3m": F.coalesce(F.sum("n_altas").over(w3), F.lit(0)),
        "bajas_3m": F.coalesce(F.sum("n_bajas").over(w3), F.lit(0)),
        "meses_observados": F.count("mes_idx").over(
            Window.partitionBy("ncodpers").orderBy("mes_idx").rangeBetween(Window.unboundedPreceding, -1)
        ),
    })

    columnas = (
        ["ncodpers", "fecha_dato", "mes_idx", "mes_del_ano"]
        + [f"prev_{c}" for c in ATRIBUTOS_CLIENTE]
        + ["n_prod_prev", "altas_3m", "bajas_3m", "meses_observados"]
        + [f"prev_{p}" for p in productos]
        + [f"tenencia_{p}" for p in productos]
        + [f"alta_{p}" for p in productos]
        + [f"baja_{p}" for p in productos]
        + ["n_altas", "n_bajas"]
    )
    return df.select(*columnas)


def main() -> None:
    spark = crear_sesion("features")
    spark.sparkContext.setLogLevel("ERROR")
    metricas: dict = {}

    if not DATA_PARQUET.exists():
        raise SystemExit(f"No existe {DATA_PARQUET}. Ejecuta antes: python src/ingest.py")

    print("\n=== FASE 2: features y objetivos ===\n")

    crudo = spark.read.parquet(str(DATA_PARQUET))
    productos = productos_de(crudo)
    filas_entrada = crudo.count()
    metricas["filas_entrada"] = filas_entrada

    feats = construir(crudo)

    if DATA_FEATURES.exists():
        shutil.rmtree(DATA_FEATURES)
    with cronometro("construccion_y_escritura", metricas):
        (
            feats.repartition("fecha_dato")
            .write.mode("overwrite")
            .partitionBy("fecha_dato")
            .parquet(str(DATA_FEATURES))
        )

    feats = spark.read.parquet(str(DATA_FEATURES))
    filas_salida = feats.count()
    metricas["filas_salida"] = filas_salida
    metricas["filas_descartadas"] = filas_entrada - filas_salida
    metricas["pct_descartado"] = round(100 * (1 - filas_salida / filas_entrada), 2)

    print(f"  Filas de entrada:  {filas_entrada:,}")
    print(f"  Filas utilizables: {filas_salida:,} "
          f"(se descarta {metricas['pct_descartado']}%: primer mes de cada cliente "
          f"y saltos no consecutivos)")

    # --- Comprobaciones de coherencia --------------------------------------------------
    with cronometro("comprobaciones", metricas):
        resumen = feats.agg(
            F.sum("n_altas").alias("altas_totales"),
            F.sum("n_bajas").alias("bajas_totales"),
            F.countDistinct("ncodpers").alias("clientes"),
            F.countDistinct("fecha_dato").alias("meses"),
            F.avg("n_prod_prev").alias("productos_por_cliente"),
        ).collect()[0].asDict()

    metricas["resumen"] = {k: (float(v) if v is not None else 0.0) for k, v in resumen.items()}
    print(f"  Clientes: {int(resumen['clientes']):,} | Meses: {int(resumen['meses'])}")
    print(f"  Altas totales: {int(resumen['altas_totales']):,} | "
          f"Bajas totales: {int(resumen['bajas_totales']):,}")
    print(f"  Productos por cliente (media): {resumen['productos_por_cliente']:.2f}")

    # Una fila-cliente-mes no puede tener mas de 24 altas.
    maximos = feats.agg(F.max("n_altas").alias("max_altas"),
                        F.max("n_bajas").alias("max_bajas")).collect()[0]
    assert maximos["max_altas"] <= len(productos), "Mas altas que productos: revisar lags"
    assert maximos["max_bajas"] <= len(productos), "Mas bajas que productos: revisar lags"

    # --- Altas y bajas por producto ----------------------------------------------------
    with cronometro("altas_por_producto", metricas):
        agregados = feats.agg(
            *[F.sum(f"alta_{p}").alias(f"alta_{p}") for p in productos],
            *[F.sum(f"baja_{p}").alias(f"baja_{p}") for p in productos],
        ).collect()[0].asDict()

    metricas["altas_por_producto"] = {
        p: int(agregados[f"alta_{p}"] or 0) for p in productos
    }
    metricas["bajas_por_producto"] = {
        p: int(agregados[f"baja_{p}"] or 0) for p in productos
    }

    top = sorted(metricas["altas_por_producto"].items(), key=lambda kv: -kv[1])[:5]
    print("\n  Productos con mas altas:")
    for nombre, cuenta in top:
        pct = 100 * cuenta / max(resumen["altas_totales"], 1)
        print(f"    {nombre:<24} {cuenta:>9,}  ({pct:4.1f}% de todas las altas)")

    guardar_metricas("features_metrics.json", metricas)
    spark.stop()


if __name__ == "__main__":
    main()
