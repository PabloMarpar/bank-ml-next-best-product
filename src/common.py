"""
Piezas compartidas por todas las fases del pipeline: sesion de Spark, rutas y
catalogo de productos.

Todo lo que se use en mas de un script vive aqui, para que no haya dos definiciones
de la lista de productos que puedan desincronizarse.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# --------------------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------------------
# El repo se monta en /app dentro del contenedor, pero los scripts tambien deben poder
# ejecutarse desde el propio repo, asi que la raiz se deduce de la ubicacion del fichero.
ROOT = Path(__file__).resolve().parent.parent

# NBP_DATA_DIR mueve todo el arbol de datos a otro sitio sin tocar el codigo. Lo usa
# scripts/smoke_test.py para ejecutar el pipeline completo sobre un dataset sintetico
# diminuto, sin pisar los datos reales ni las metricas buenas.
DATA = Path(os.environ.get("NBP_DATA_DIR") or (ROOT / "data"))
DATA_RAW = DATA / "raw"
DATA_PARQUET = DATA / "parquet"
DATA_FEATURES = DATA / "features"
DATA_EXPORT = DATA / "export"
OUTPUTS = Path(os.environ.get("NBP_OUTPUTS_DIR") or (ROOT / "outputs"))

CSV_TRAIN = DATA_RAW / "train_ver2.csv"

# --------------------------------------------------------------------------------------
# Calendario del dataset
# --------------------------------------------------------------------------------------
# 17 snapshots mensuales. Las fechas son fin de mes con dia 28.
PRIMER_MES = "2015-01-28"
ULTIMO_MES = "2016-05-28"
# Split temporal: se entrena con todo lo anterior y se valida sobre el ultimo mes.
# Nunca un split aleatorio -- ver README, seccion "Como se evalua".
MES_VALIDACION = "2016-05-28"
FILAS_ESPERADAS = 13_647_309

# --------------------------------------------------------------------------------------
# Catalogo de productos
# --------------------------------------------------------------------------------------
# Las 24 columnas de producto siguen el patron ind_*_ult1. Se derivan del DataFrame con
# productos_de(df) en vez de hardcodearse, pero el catalogo de nombres legibles si vive
# aqui: hace falta para la demo y para los graficos, donde "ind_cco_fin_ult1" no dice nada.
NOMBRES_PRODUCTO = {
    "ind_ahor_fin_ult1": "Cuenta de ahorro",
    "ind_aval_fin_ult1": "Aval",
    "ind_cco_fin_ult1": "Cuenta corriente",
    "ind_cder_fin_ult1": "Cuenta derivada",
    "ind_cno_fin_ult1": "Cuenta nomina",
    "ind_ctju_fin_ult1": "Cuenta junior",
    "ind_ctma_fin_ult1": "Cuenta mas particular",
    "ind_ctop_fin_ult1": "Cuenta particular",
    "ind_ctpp_fin_ult1": "Cuenta particular plus",
    "ind_deco_fin_ult1": "Deposito a corto plazo",
    "ind_deme_fin_ult1": "Deposito a medio plazo",
    "ind_dela_fin_ult1": "Deposito a largo plazo",
    "ind_ecue_fin_ult1": "Cuenta digital",
    "ind_fond_fin_ult1": "Fondos de inversion",
    "ind_hip_fin_ult1": "Hipoteca",
    "ind_plan_fin_ult1": "Plan de pensiones",
    "ind_pres_fin_ult1": "Prestamo",
    "ind_reca_fin_ult1": "Domiciliacion de impuestos",
    "ind_tjcr_fin_ult1": "Tarjeta de credito",
    "ind_valo_fin_ult1": "Valores",
    "ind_viv_fin_ult1": "Cuenta vivienda",
    "ind_nomina_ult1": "Nomina domiciliada",
    "ind_nom_pens_ult1": "Pension domiciliada",
    "ind_recibo_ult1": "Recibos domiciliados",
}
N_PRODUCTOS = 24


def productos_de(df: DataFrame) -> list[str]:
    """Devuelve las columnas de producto presentes en el DataFrame.

    Se derivan del esquema en vez de hardcodearse para que un cambio de columnas rompa
    ruidosamente aqui y no silenciosamente tres fases mas adelante. El patron
    ind_*_ult1 excluye correctamente ind_empleado, ind_nuevo e ind_actividad_cliente,
    que empiezan por "ind_" pero no son productos.
    """
    cols = [c for c in df.columns if c.startswith("ind_") and c.endswith("_ult1")]
    if len(cols) != N_PRODUCTOS:
        raise ValueError(
            f"Se esperaban {N_PRODUCTOS} columnas de producto y se han encontrado "
            f"{len(cols)}: {cols}"
        )
    return cols


# --------------------------------------------------------------------------------------
# Sesion de Spark
# --------------------------------------------------------------------------------------
def crear_sesion(nombre: str, shuffle_partitions: int = 64) -> SparkSession:
    """Crea la SparkSession con la configuracion comun a todo el proyecto.

    shuffle_partitions=64: el valor por defecto de Spark es 200, pensado para un cluster.
    En una sola maquina con 14 cores, 200 particiones significa 200 tareas diminutas y mas
    tiempo de planificacion que de calculo. 64 va sobrado para 13,6M de filas en local.
    Medido en PERFORMANCE.md.
    """
    memoria = os.environ.get("SPARK_DRIVER_MEMORY", "8g")
    return (
        SparkSession.builder.appName(nombre)
        .master("local[*]")
        .config("spark.driver.memory", memoria)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        # Adaptive Query Execution: deja que Spark reajuste el plan con las estadisticas
        # reales de cada stage (fusiona particiones pequenas, parte las sesgadas).
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Sin esto, escribir un Parquet particionado con fechas da error de casting en
        # las versiones nuevas de Spark por el cambio de calendario juliano/gregoriano.
        .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        .getOrCreate()
    )


@contextmanager
def cronometro(etiqueta: str, registro: dict | None = None):
    """Mide cuanto tarda un bloque y lo imprime. Si se pasa un dict, guarda el tiempo."""
    print(f"  [inicio] {etiqueta}...", flush=True)
    t0 = time.perf_counter()
    yield
    segundos = time.perf_counter() - t0
    print(f"  [fin]    {etiqueta}: {segundos:.1f}s", flush=True)
    if registro is not None:
        registro[etiqueta] = round(segundos, 2)


def guardar_metricas(nombre_fichero: str, metricas: dict) -> Path:
    """Escribe un dict de metricas en outputs/ como JSON legible."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    destino = OUTPUTS / nombre_fichero
    destino.write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\nMetricas guardadas en {destino}")
    return destino


def tamano_directorio_mb(ruta: Path) -> float:
    """Tamano total de un directorio en MB, recursivo."""
    if not ruta.exists():
        return 0.0
    total = sum(f.stat().st_size for f in ruta.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 1)


# --------------------------------------------------------------------------------------
# Preprocesado compartido por los modelos (Fases 3 y 4)
# --------------------------------------------------------------------------------------
def marcar_faltantes(df: DataFrame, numericas: list[str]) -> DataFrame:
    """Pasa las numericas a double y anade una columna 0/1 por cada una diciendo si falta.

    Dos motivos:

    1. VectorAssembler convierte los nulos en NaN, y los arboles de Spark rechazan un
       vector con NaN con un error que no explica de donde viene. Hay que imputar si o si.

    2. Que a un cliente le falte la renta no es ruido: en este dataset son sobre todo
       clientes antiguos con la ficha incompleta, y eso en si mismo predice. Si se imputa
       la mediana a secas, esa informacion se pierde. Guardando aparte la marca de
       "faltaba", el modelo puede usar las dos cosas: el valor imputado y el hecho de que
       fuese desconocido.
    """
    nuevas = {}
    for col in numericas:
        nuevas[col] = F.col(col).cast("double")
        nuevas[f"{col}_falta"] = F.col(col).isNull().cast("double")
    return df.withColumns(nuevas)


def agrupar_categorias_raras(
    df: DataFrame, columna: str, maximo: int = 30, etiqueta: str = "OTROS"
) -> DataFrame:
    """Deja las `maximo` categorias mas frecuentes y agrupa el resto bajo una etiqueta.

    Por que hace falta: prev_canal_entrada tiene 161 valores distintos. Los arboles de
    Spark necesitan que maxBins sea mayor que la cardinalidad de la categorica mas
    grande, y el coste de construir los histogramas de cada nodo crece con maxBins. Con
    161 canales hay que poner maxBins por encima de 161, y el entrenamiento se arrastra.

    La cola larga ademas no aporta: los canales que aparecen cuatro veces no dan senal,
    dan ruido y sobreajuste. Agruparlos en OTROS baja maxBins a 32 y conserva lo util.
    """
    frecuentes = [
        f[columna]
        for f in df.groupBy(columna).count().orderBy(F.desc("count")).limit(maximo).collect()
        if f[columna] is not None
    ]
    return df.withColumn(
        columna,
        F.when(F.col(columna).isin(frecuentes), F.col(columna)).otherwise(F.lit(etiqueta)),
    )


def etapas_preprocesado(
    categoricas: list[str],
    numericas: list[str],
    columnas_extra: list[str],
    one_hot: bool = False,
) -> tuple[list, str]:
    """Construye las etapas de preprocesado comunes a los modelos.

    Devuelve (etapas, nombre_columna_features). El orden importa: indexar -> (codificar)
    -> imputar -> ensamblar.

    one_hot=False por defecto, y es deliberado. El one-hot es para modelos LINEALES, que
    necesitan una columna por categoria porque no saben agrupar valores. Los arboles si
    saben: StringIndexer deja metadata que marca la columna como nominal, VectorAssembler
    la propaga, y el arbol hace particiones por subconjuntos de categorias -- algo que el
    one-hot precisamente le impide, porque cada corte solo puede aislar una categoria.

    Medido en este proyecto: con one_hot=True el vector pasaba de ~45 a ~250 dimensiones
    por culpa de los 161 canales de entrada, y el GBT usaba el 23% de la CPU disponible
    construyendo histogramas de columnas casi siempre a cero.

    Se deja el parametro porque si algun dia se anade un modelo lineal (una regresion
    logistica de referencia, por ejemplo) ahi si hara falta ponerlo a True.
    """
    indexadores = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in categoricas
    ]
    codificadores = (
        [
            OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe", handleInvalid="keep")
            for c in categoricas
        ]
        if one_hot
        else []
    )
    sufijo = "_ohe" if one_hot else "_idx"

    # Mediana y no media: renta y edad tienen colas largas y la media se desplaza sola.
    imputador = Imputer(
        inputCols=numericas,
        outputCols=[f"{c}_imp" for c in numericas],
        strategy="median",
    )
    ensamblador = VectorAssembler(
        inputCols=(
            [f"{c}_imp" for c in numericas]
            + [f"{c}_falta" for c in numericas]
            + columnas_extra
            + [f"{c}{sufijo}" for c in categoricas]
        ),
        outputCol="features",
        handleInvalid="keep",
    )
    return [*indexadores, *codificadores, imputador, ensamblador], "features"
