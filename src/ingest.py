"""
FASE 1 -- Ingesta: CSV -> Parquet particionado.

Convierte los 2,3 GB de CSV crudo en Parquet particionado por mes, dejando los tipos
ya limpios para que ninguna fase posterior tenga que volver a pelearse con el formato.

    python src/ingest.py

Decisiones que no son obvias leyendo el codigo:

1. Esquema explicito, sin inferSchema. Con inferSchema, Spark lee el fichero entero una
   primera vez solo para adivinar los tipos, y luego lo vuelve a leer para cargarlo: son
   2,3 GB leidos dos veces. Declarar el esquema lo deja en una sola pasada.

2. Todas las columnas se declaran como texto y se castean despues, una a una. Es
   deliberado: este CSV trae "NA" como texto en renta, numeros con espacios de relleno en
   age y antiguedad, y celdas vacias en dos productos. Si se declarara el tipo final en el
   esquema, Spark aplicaria su modo permisivo y convertiria en null lo que no encaje sin
   avisar -- perdiendo la oportunidad de contar cuanto se pierde. Casteando a mano
   controlamos exactamente que se convierte en null y podemos medirlo.

3. antiguedad usa -999999 como valor centinela para clientes nuevos. Dejarlo tal cual
   envenenaria cualquier media o cualquier corte de un arbol, asi que pasa a null.
"""

from __future__ import annotations

import shutil

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from common import (
    CSV_TRAIN,
    DATA_PARQUET,
    FILAS_ESPERADAS,
    N_PRODUCTOS,
    OUTPUTS,
    crear_sesion,
    cronometro,
    guardar_metricas,
    productos_de,
    tamano_directorio_mb,
)

# Orden exacto de las 48 columnas del CSV. Todas como texto: ver nota 2 del docstring.
COLUMNAS = [
    "fecha_dato", "ncodpers", "ind_empleado", "pais_residencia", "sexo", "age",
    "fecha_alta", "ind_nuevo", "antiguedad", "indrel", "ult_fec_cli_1t", "indrel_1mes",
    "tiprel_1mes", "indresi", "indext", "conyuemp", "canal_entrada", "indfall",
    "tipodom", "cod_prov", "nomprov", "ind_actividad_cliente", "renta", "segmento",
    "ind_ahor_fin_ult1", "ind_aval_fin_ult1", "ind_cco_fin_ult1", "ind_cder_fin_ult1",
    "ind_cno_fin_ult1", "ind_ctju_fin_ult1", "ind_ctma_fin_ult1", "ind_ctop_fin_ult1",
    "ind_ctpp_fin_ult1", "ind_deco_fin_ult1", "ind_deme_fin_ult1", "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1", "ind_plan_fin_ult1",
    "ind_pres_fin_ult1", "ind_reca_fin_ult1", "ind_tjcr_fin_ult1", "ind_valo_fin_ult1",
    "ind_viv_fin_ult1", "ind_nomina_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1",
]

ESQUEMA = StructType([StructField(c, StringType(), nullable=True) for c in COLUMNAS])

COLS_ENTERAS = [
    "ncodpers", "ind_nuevo", "indrel", "tipodom", "cod_prov", "ind_actividad_cliente",
]
COLS_FECHA = ["fecha_dato", "fecha_alta", "ult_fec_cli_1t"]

# Columnas cuyo casteo puede perder informacion y que por eso se vigilan al final.
COLS_VIGILADAS = ["age", "antiguedad", "renta", "segmento", "ind_nomina_ult1"]


def leer_csv(spark):
    """Lee el CSV crudo con esquema explicito.

    quote y escape importan: nomprov trae valores con coma dentro y entrecomillados,
    como "CORUNA, A". Sin tratarlos, esa fila se desplazaria una columna entera y
    todo lo que viene detras quedaria corrido.
    """
    return (
        spark.read.option("header", True)
        .option("quote", '"')
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .schema(ESQUEMA)
        .csv(str(CSV_TRAIN))
    )


def limpiar(df):
    """Castea cada columna a su tipo real, de forma controlada."""
    productos = productos_de(df)

    # Todo el dataset viene con espacios de relleno; se quitan antes de castear nada.
    df = df.select([F.trim(F.col(c)).alias(c) for c in df.columns])

    # Lo que queda vacio tras el trim, y los "NA" escritos como texto, son nulos de verdad.
    df = df.select([
        F.when(F.col(c).isin("", "NA"), None).otherwise(F.col(c)).alias(c)
        for c in df.columns
    ])

    conversiones = {c: F.col(c).cast("int") for c in COLS_ENTERAS}
    conversiones.update({c: F.to_date(F.col(c), "yyyy-MM-dd") for c in COLS_FECHA})
    conversiones["renta"] = F.col("renta").cast("double")
    conversiones["age"] = F.col("age").cast("int")

    # antiguedad trae -999999 como centinela de "cliente nuevo": se convierte en null
    # para no contaminar medias ni cortes de arboles con un valor inventado.
    conversiones["antiguedad"] = (
        F.when(F.col("antiguedad").cast("int") < 0, None)
        .otherwise(F.col("antiguedad").cast("int"))
    )

    # Productos: 0/1. Los nulos se dejan como nulos a proposito -- decidir si un nulo
    # significa "no tiene el producto" es una decision de modelado, y se toma de forma
    # explicita en features.py, no escondida dentro de la ingesta.
    conversiones.update({p: F.col(p).cast("int") for p in productos})

    # El resto de columnas (ind_empleado, sexo, segmento...) se quedan como texto limpio.
    return df.withColumns(conversiones)


def main() -> None:
    spark = crear_sesion("ingesta")
    spark.sparkContext.setLogLevel("ERROR")
    metricas: dict = {}

    if not CSV_TRAIN.exists():
        raise SystemExit(
            f"No existe {CSV_TRAIN}. Ejecuta antes: python scripts/download_data.py"
        )
    metricas["csv_mb"] = round(CSV_TRAIN.stat().st_size / (1024 ** 2), 1)

    print("\n=== FASE 1: ingesta CSV -> Parquet ===\n")

    # --- Medicion 1: contar filas leyendo el CSV ---------------------------------------
    # Un count() sobre CSV obliga a parsear el fichero entero. Es el peor caso, y sirve
    # de referencia para medir cuanto gana Parquet.
    crudo = leer_csv(spark)
    with cronometro("count_csv", metricas):
        filas = crudo.count()
    metricas["filas"] = filas
    print(f"  Filas leidas: {filas:,}")
    if filas != FILAS_ESPERADAS:
        print(f"  AVISO: se esperaban {FILAS_ESPERADAS:,} filas.")

    # --- Escritura a Parquet -----------------------------------------------------------
    limpio = limpiar(crudo)
    n_productos = len(productos_de(limpio))
    print(f"  Columnas de producto detectadas: {n_productos}")
    assert n_productos == N_PRODUCTOS

    if DATA_PARQUET.exists():
        shutil.rmtree(DATA_PARQUET)
    DATA_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    with cronometro("escritura_parquet", metricas):
        (
            # repartition por la misma columna del partitionBy deja un fichero por mes
            # en vez de un fragmento por cada tarea: 17 ficheros en lugar de cientos.
            limpio.repartition("fecha_dato")
            .write.mode("overwrite")
            .partitionBy("fecha_dato")
            .parquet(str(DATA_PARQUET))
        )

    # --- Medicion 2: contar filas leyendo el Parquet -----------------------------------
    parquet = spark.read.parquet(str(DATA_PARQUET))
    with cronometro("count_parquet", metricas):
        filas_parquet = parquet.count()

    metricas["filas_parquet"] = filas_parquet
    metricas["parquet_mb"] = tamano_directorio_mb(DATA_PARQUET)
    metricas["particiones_mes"] = len([d for d in DATA_PARQUET.iterdir() if d.is_dir()])
    metricas["ficheros_parquet"] = len(list(DATA_PARQUET.rglob("*.parquet")))

    assert filas_parquet == filas, "El Parquet no tiene las mismas filas que el CSV"

    # --- Calidad del casteo ------------------------------------------------------------
    # Cuantos nulos ha generado la limpieza en las columnas problematicas conocidas.
    # Si esto se dispara, algo del casteo esta mal y hay que mirarlo antes de seguir.
    with cronometro("recuento_nulos", metricas):
        nulos = parquet.select([
            F.sum(F.col(c).isNull().cast("long")).alias(c) for c in COLS_VIGILADAS
        ]).collect()[0].asDict()
    metricas["nulos_tras_casteo"] = {k: int(v or 0) for k, v in nulos.items()}
    metricas["pct_nulos"] = {
        k: round(100 * v / filas, 2) for k, v in metricas["nulos_tras_casteo"].items()
    }

    # --- Resumen -----------------------------------------------------------------------
    metricas["reduccion_tamano"] = round(metricas["csv_mb"] / metricas["parquet_mb"], 1)
    metricas["aceleracion_count"] = round(
        metricas["count_csv"] / metricas["count_parquet"], 1
    )

    print("\n--- Resultado ---")
    print(f"  CSV:     {metricas['csv_mb']:>9,.1f} MB   count: {metricas['count_csv']:>6.1f}s")
    print(f"  Parquet: {metricas['parquet_mb']:>9,.1f} MB   count: {metricas['count_parquet']:>6.1f}s")
    print(f"  Parquet ocupa {metricas['reduccion_tamano']}x menos espacio y se cuenta "
          f"{metricas['aceleracion_count']}x mas rapido")
    print(f"  Particiones mensuales: {metricas['particiones_mes']}")
    print(f"  Nulos tras el casteo (%): {metricas['pct_nulos']}")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    guardar_metricas("ingest_metrics.json", metricas)
    spark.stop()


if __name__ == "__main__":
    main()
