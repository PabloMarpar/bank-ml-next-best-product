"""
Prepara el fichero que consume la demo de Streamlit.

Junta las predicciones de las Fases 3 y 4, toma una muestra de clientes y la escribe como
un unico Parquet pequeno que si se puede versionar y desplegar.

    python scripts/exportar_demo.py

Por que una muestra y no todo: la demo se despliega en Streamlit Cloud, que monta el repo
entero en una maquina pequena. Subir las predicciones de 900.000 clientes para que alguien
consulte cinco es gastar espacio y tiempo de arranque a cambio de nada. Con unos miles de
clientes la demo se comporta igual y el fichero pesa poco.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pyspark.sql import functions as F  # noqa: E402

from common import (  # noqa: E402
    DATA_EXPORT,
    DATA_FEATURES,
    MES_VALIDACION,
    NOMBRES_PRODUCTO,
    crear_sesion,
)

PRODUCTOS = list(NOMBRES_PRODUCTO)
DESTINO = ROOT / "app" / "demo_data.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clientes", type=int, default=4000)
    args = parser.parse_args()

    for ruta in (DATA_EXPORT / "prob_compra", DATA_EXPORT / "prob_baja"):
        if not ruta.exists():
            raise SystemExit(
                f"Falta {ruta}. Ejecuta antes:\n"
                "  python src/recommend.py --exportar\n"
                "  python src/churn.py --exportar"
            )

    spark = crear_sesion("exportar-demo")
    spark.sparkContext.setLogLevel("ERROR")

    compra = spark.read.parquet(str(DATA_EXPORT / "prob_compra"))
    baja = (
        spark.read.parquet(str(DATA_EXPORT / "prob_baja"))
        .groupBy("ncodpers")
        .agg(F.max("prob_baja").alias("prob_baja"))
    )

    columnas_poseidos = F.array(*[
        F.when(F.col(f"prev_{p}") == 1, F.lit(p)).otherwise(F.lit(None))
        for p in PRODUCTOS
    ])
    contexto = (
        spark.read.parquet(str(DATA_FEATURES))
        .filter(F.col("fecha_dato") == MES_VALIDACION)
        .select(
            "ncodpers", "n_prod_prev", "prev_segmento", "prev_age", "prev_antiguedad",
            F.array_compact(columnas_poseidos).alias("poseidos"),
        )
    )

    # La muestra se toma de forma determinista (no aleatoria) para que la demo no cambie
    # de contenido cada vez que se regenera: se cogen los de mayor propension, que ademas
    # son los interesantes de ensenar.
    demo = (
        compra.join(baja, "ncodpers", "left")
        .join(contexto, "ncodpers", "left")
        .fillna({"prob_baja": 0.0})
        .orderBy(F.desc("prob_compra"))
        .limit(args.clientes)
    )

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    pandas_df = demo.toPandas()
    pandas_df.to_parquet(DESTINO, index=False)

    tamano_kb = DESTINO.stat().st_size / 1024
    print(f"Demo escrita en {DESTINO}")
    print(f"  {len(pandas_df):,} clientes, {tamano_kb:,.0f} KB")

    spark.stop()


if __name__ == "__main__":
    main()
