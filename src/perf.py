"""
FASE 6 -- Experimentos de rendimiento sobre el dataset real.

Genera los numeros que se comentan en PERFORMANCE.md. No es teoria copiada: cada
experimento se ejecuta sobre los 13,6M de filas de este proyecto y se mide.

    python src/perf.py

Los cinco experimentos:

  1. FORMATO      CSV contra Parquet leyendo solo dos columnas.
  2. PARTICIONES  leer un mes concreto con y sin aprovechar el particionado.
  3. JOIN         broadcast join contra sort-merge join.
  4. CACHE        reutilizar un resultado con y sin cache.
  5. SESGO        como estan repartidas de verdad las filas entre particiones.

Todos los tiempos se miden con una accion que fuerza el calculo completo (count), porque
Spark es perezoso: construir un DataFrame no ejecuta nada, y cronometrar la construccion
daria siempre cero.
"""

from __future__ import annotations

import time

from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast

from common import (
    CSV_TRAIN,
    DATA_PARQUET,
    MES_VALIDACION,
    crear_sesion,
    guardar_metricas,
)


def medir(descripcion: str, funcion) -> tuple[float, object]:
    """Ejecuta y cronometra. Devuelve (segundos, resultado)."""
    inicio = time.perf_counter()
    resultado = funcion()
    segundos = time.perf_counter() - inicio
    print(f"    {descripcion:<52} {segundos:>7.2f}s")
    return segundos, resultado


def exp_formato(spark, resultados: dict) -> None:
    """Leer dos columnas de 13,6M de filas: CSV frente a Parquet.

    Parquet guarda los datos por columnas, asi que puede leer solo las dos que se piden
    y saltarse las otras 46. El CSV es texto por filas: para llegar a la columna 3 hay
    que parsear la 1 y la 2 de todas y cada una de las filas.
    """
    print("\n  1. FORMATO -- leer solo dos columnas")

    t_csv, _ = medir("CSV, seleccionando 2 columnas de 48", lambda: (
        spark.read.option("header", True).csv(str(CSV_TRAIN))
        .select("ncodpers", "renta").count()
    ))
    t_parquet, _ = medir("Parquet, seleccionando 2 columnas de 48", lambda: (
        spark.read.parquet(str(DATA_PARQUET)).select("ncodpers", "renta").count()
    ))
    resultados["formato"] = {
        "csv_s": round(t_csv, 2),
        "parquet_s": round(t_parquet, 2),
        "veces_mas_rapido": round(t_csv / max(t_parquet, 0.01), 1),
    }
    print(f"    -> Parquet es {resultados['formato']['veces_mas_rapido']}x mas rapido")


def exp_particiones(spark, resultados: dict) -> None:
    """Filtrar por el mes: con particionado, Spark ni abre los ficheros de los otros meses.

    Se llama "partition pruning". La tabla esta escrita en carpetas por fecha_dato, asi
    que un filtro por esa columna se resuelve descartando carpetas enteras antes de leer
    nada. Filtrar por una columna que NO es de particionado obliga a leerlo todo.
    """
    print("\n  2. PARTICIONADO -- filtrar por mes frente a filtrar por otra columna")

    t_particion, _ = medir(f"filtro por fecha_dato = {MES_VALIDACION}", lambda: (
        spark.read.parquet(str(DATA_PARQUET))
        .filter(F.col("fecha_dato") == MES_VALIDACION).count()
    ))
    t_columna, _ = medir("filtro por una columna no particionada (renta)", lambda: (
        spark.read.parquet(str(DATA_PARQUET))
        .filter(F.col("renta") > 100000).count()
    ))
    resultados["particionado"] = {
        "filtro_particion_s": round(t_particion, 2),
        "filtro_columna_s": round(t_columna, 2),
    }

    # El plan fisico deja constancia escrita de que la poda ocurre.
    plan = (
        spark.read.parquet(str(DATA_PARQUET))
        .filter(F.col("fecha_dato") == MES_VALIDACION)
        ._jdf.queryExecution().executedPlan().toString()
    )
    linea = next(
        (l.strip() for l in plan.splitlines() if "PartitionFilters" in l), "no encontrado"
    )
    resultados["particionado"]["plan"] = linea[:300]
    print(f"    -> en el plan: {linea[:110]}")


def exp_join(spark, resultados: dict) -> None:
    """Cruzar la tabla grande con una pequena: broadcast frente a sort-merge.

    Sin ayuda, Spark cruza dos tablas ordenando y barajando las dos por la clave: eso
    mueve 13,6M de filas por la red interna (un "shuffle", la operacion mas cara que
    hay). Si una de las dos tablas es diminuta, sale mucho mas barato mandarle una copia
    entera a cada ejecutor y cruzar en memoria, sin mover la grande. Eso es el broadcast.
    """
    print("\n  3. JOIN -- broadcast frente a sort-merge")

    grande = spark.read.parquet(str(DATA_PARQUET)).select("ncodpers", "cod_prov", "renta")

    # Tabla pequena de referencia: renta media por provincia (52 filas).
    pequena = (
        grande.groupBy("cod_prov").agg(F.avg("renta").alias("renta_media_provincia"))
    ).cache()
    pequena.count()

    t_sortmerge, _ = medir("sort-merge join (baraja las dos tablas)", lambda: (
        grande.hint("shuffle_merge").join(pequena, "cod_prov").count()
    ))
    t_broadcast, _ = medir("broadcast join (copia la tabla pequena)", lambda: (
        grande.join(broadcast(pequena), "cod_prov").count()
    ))

    resultados["join"] = {
        "sort_merge_s": round(t_sortmerge, 2),
        "broadcast_s": round(t_broadcast, 2),
        "veces_mas_rapido": round(t_sortmerge / max(t_broadcast, 0.01), 1),
        "filas_tabla_pequena": pequena.count(),
    }
    print(f"    -> broadcast es {resultados['join']['veces_mas_rapido']}x mas rapido")
    pequena.unpersist()


def exp_cache(spark, resultados: dict) -> None:
    """Reutilizar un resultado caro: con cache y sin cache.

    Spark no guarda nada entre acciones por defecto. Si sobre el mismo DataFrame se
    lanzan tres consultas, recalcula las tres veces desde el fichero. cache() lo deja en
    memoria tras el primer calculo.

    No es gratis: ocupa memoria que le quitas a los calculos. Solo compensa si de verdad
    se va a reutilizar, y por eso en el pipeline solo esta cacheado lo que se usa mas de
    una vez.
    """
    print("\n  4. CACHE -- tres consultas sobre el mismo resultado intermedio")

    def construir():
        return (
            spark.read.parquet(str(DATA_PARQUET))
            .filter(F.col("renta").isNotNull())
            .groupBy("cod_prov")
            .agg(F.avg("renta").alias("m"), F.count("*").alias("n"))
        )

    sin_cache = construir()
    t_sin, _ = medir("3 consultas sin cache (recalcula 3 veces)", lambda: [
        sin_cache.count(), sin_cache.agg(F.max("m")).collect(),
        sin_cache.agg(F.min("n")).collect(),
    ])

    con_cache = construir().cache()
    con_cache.count()  # primera accion: materializa la cache
    t_con, _ = medir("3 consultas con cache (ya materializado)", lambda: [
        con_cache.count(), con_cache.agg(F.max("m")).collect(),
        con_cache.agg(F.min("n")).collect(),
    ])
    con_cache.unpersist()

    resultados["cache"] = {
        "sin_cache_s": round(t_sin, 2),
        "con_cache_s": round(t_con, 2),
        "veces_mas_rapido": round(t_sin / max(t_con, 0.01), 1),
    }
    print(f"    -> con cache es {resultados['cache']['veces_mas_rapido']}x mas rapido")


def exp_sesgo(spark, resultados: dict) -> None:
    """Como estan repartidas las filas entre particiones.

    Spark reparte el trabajo por particiones, y un stage no termina hasta que acaba su
    tarea mas lenta. Si una particion tiene diez veces mas filas que las demas, trece
    ejecutores esperan a uno. Eso es el sesgo (skew), y es la causa numero uno de que un
    job vaya lento sin motivo aparente.
    """
    print("\n  5. SESGO -- reparto real de filas por particion")

    df = spark.read.parquet(str(DATA_PARQUET))
    conteos = (
        df.withColumn("particion", F.spark_partition_id())
        .groupBy("particion").count()
        .orderBy(F.desc("count"))
        .collect()
    )
    valores = [c["count"] for c in conteos]
    resultados["sesgo"] = {
        "n_particiones": len(valores),
        "filas_max": max(valores),
        "filas_min": min(valores),
        "filas_media": round(sum(valores) / len(valores)),
        "ratio_max_media": round(max(valores) / (sum(valores) / len(valores)), 2),
    }
    s = resultados["sesgo"]
    print(f"    particiones: {s['n_particiones']} | "
          f"maxima: {s['filas_max']:,} | media: {s['filas_media']:,} | "
          f"minima: {s['filas_min']:,}")
    print(f"    -> la particion mayor tiene {s['ratio_max_media']}x la media")

    # Reparto por mes: aqui si se espera desigualdad, porque el banco fue creciendo.
    por_mes = (
        df.groupBy("fecha_dato").count().orderBy("fecha_dato").collect()
    )
    resultados["filas_por_mes"] = {
        str(r["fecha_dato"]): r["count"] for r in por_mes
    }


def main() -> None:
    spark = crear_sesion("rendimiento")
    spark.sparkContext.setLogLevel("ERROR")
    resultados: dict = {}

    print("\n=== FASE 6: experimentos de rendimiento ===")
    print(f"  Memoria del driver: {spark.conf.get('spark.driver.memory')}")
    print(f"  Particiones de shuffle: {spark.conf.get('spark.sql.shuffle.partitions')}")
    print(f"  Nucleos disponibles: {spark.sparkContext.defaultParallelism}")
    resultados["configuracion"] = {
        "driver_memory": spark.conf.get("spark.driver.memory"),
        "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
        "nucleos": spark.sparkContext.defaultParallelism,
    }

    if CSV_TRAIN.exists():
        exp_formato(spark, resultados)
    else:
        print("\n  (sin CSV original: se omite el experimento de formato)")

    exp_particiones(spark, resultados)
    exp_join(spark, resultados)
    exp_cache(spark, resultados)
    exp_sesgo(spark, resultados)

    guardar_metricas("perf_metrics.json", resultados)
    print("\n  Los numeros de PERFORMANCE.md salen de este fichero.")
    spark.stop()


if __name__ == "__main__":
    main()
