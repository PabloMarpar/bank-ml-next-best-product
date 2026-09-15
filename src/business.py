"""
FASE 5 -- De las probabilidades a un plan comercial.

Un modelo devuelve numeros entre 0 y 1. Un equipo comercial necesita una lista de
nombres y un motivo para llamar a cada uno. Esta fase hace esa traduccion y produce las
dos cosas que se pueden ensenar a alguien que no es tecnico:

  1. LA CURVA DE CAPTURA
     "Si solo puedo llamar a N clientes este mes, cuantas de las contrataciones reales
     estoy alcanzando?" Se dibuja contra la linea de llamar al azar, que es lo que se
     consigue sin modelo. La distancia entre las dos curvas es, literalmente, lo que
     aporta el sistema.

  2. LA MATRIZ VALOR x RIESGO
     Cruzar "cuanto probable es que contrate" con "cuanto probable es que se vaya" parte
     la cartera en cuatro grupos, y cada grupo tiene una accion distinta. Un cliente con
     alta propension de compra y bajo riesgo es una venta; el mismo cliente con alto
     riesgo de fuga es una retencion, y llamarle para venderle algo seria el peor
     movimiento posible.

    python src/business.py

La cifra de titular que sale de aqui es la que resume el proyecto en una frase.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # sin servidor grafico dentro del contenedor
import matplotlib.pyplot as plt  # noqa: E402

from pyspark.sql import functions as F  # noqa: E402

from common import (  # noqa: E402
    DATA_EXPORT,
    OUTPUTS,
    crear_sesion,
    cronometro,
    guardar_metricas,
)

# Puntos de la curva que se comentan en el README.
HITOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]


def curva_captura(df, col_puntuacion: str, col_acierto: str) -> list[dict]:
    """Para cada porcentaje de cartera contactada, que porcentaje de aciertos se captura.

    La idea es la de una curva de ganancia acumulada de siempre: ordenar la cartera por
    la puntuacion del modelo y ver cuanto se lleva capturado en cada tramo.

    Como se calcula, y por que asi:

    La forma directa seria numerar las filas con row_number() sobre una ventana ordenada
    por puntuacion. El problema es que una ventana SIN partitionBy obliga a Spark a
    juntar toda la tabla en un unico ejecutor para poder ordenarla entera -- con cientos
    de miles de clientes eso es un cuello de botella, y con millones se cae.

    Aqui se hace al reves y se queda distribuido: primero se piden los percentiles de la
    puntuacion con approxQuantile (que Spark resuelve en paralelo con un algoritmo
    aproximado), y despues se cuenta, tambien en paralelo, cuantos aciertos hay por
    encima de cada umbral. Ninguna de las dos operaciones necesita un orden global.
    """
    totales = df.agg(
        F.sum(col_acierto).alias("aciertos"), F.count("*").alias("filas")
    ).collect()[0]
    total_aciertos = totales["aciertos"] or 0
    total_filas = totales["filas"]

    # approxQuantile espera probabilidades ascendentes; contactar al 5% mejor equivale a
    # quedarse por encima del percentil 95.
    probabilidades = [1 - f for f in HITOS]
    umbrales = df.approxQuantile(col_puntuacion, probabilidades, 0.001)

    # Una sola pasada calcula todos los tramos a la vez, en vez de una pasada por tramo.
    agregados = df.agg(*[
        agg
        for i, umbral in enumerate(umbrales)
        for agg in (
            F.sum(F.when(F.col(col_puntuacion) >= umbral, F.col(col_acierto))
                  .otherwise(0)).alias(f"aciertos_{i}"),
            F.sum(F.when(F.col(col_puntuacion) >= umbral, 1)
                  .otherwise(0)).alias(f"contactados_{i}"),
        )
    ]).collect()[0].asDict()

    puntos = []
    for i, fraccion in enumerate(HITOS):
        capturados = agregados[f"aciertos_{i}"] or 0
        contactados = agregados[f"contactados_{i}"] or 0
        pct_capturado = 100 * capturados / max(total_aciertos, 1)
        # El % real contactado puede desviarse algo del objetivo si hay muchos empates
        # en la puntuacion; se reporta el real, no el pedido.
        pct_real = 100 * contactados / max(total_filas, 1)
        puntos.append({
            "pct_cartera": round(100 * fraccion, 1),
            "pct_cartera_real": round(pct_real, 1),
            "clientes_contactados": int(contactados),
            "aciertos_capturados": int(capturados),
            "pct_capturado": round(pct_capturado, 1),
            "pct_capturado_azar": round(pct_real, 1),
            "multiplicador": round(pct_capturado / max(pct_real, 1e-9), 2),
        })
    return puntos


def dibujar_curva(puntos: list[dict], destino, titulo: str) -> None:
    """Grafico de la curva de captura frente a la linea del azar."""
    x = [0] + [p["pct_cartera"] for p in puntos] + [100]
    y = [0] + [p["pct_capturado"] for p in puntos] + [100]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, y, marker="o", linewidth=2.2, color="#1f4e79",
            label="Ordenando por el modelo")
    ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1.5, color="#999999",
            label="Llamando al azar")
    ax.fill_between(x, y, x, alpha=0.12, color="#1f4e79")

    # Se anota el hito del 5%, que es el que se cita en el README.
    hito = next((p for p in puntos if p["pct_cartera"] == 5.0), None)
    if hito:
        ax.annotate(
            f"Contactando al 5%\nse captura el {hito['pct_capturado']}%",
            xy=(5, hito["pct_capturado"]),
            xytext=(22, max(hito["pct_capturado"] - 22, 8)),
            arrowprops=dict(arrowstyle="->", color="#1f4e79"),
            fontsize=10,
        )

    ax.set_xlabel("% de la cartera contactada")
    ax.set_ylabel("% de contrataciones reales capturadas")
    ax.set_title(titulo)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    print(f"  Grafico guardado en {destino}")


def matriz_valor_riesgo(df, umbral_compra: float, umbral_fuga: float) -> dict:
    """Parte la cartera en cuatro cuadrantes y les pone nombre de accion comercial."""
    clasificado = df.withColumn(
        "cuadrante",
        F.when(
            (F.col("prob_compra") >= umbral_compra) & (F.col("prob_baja") < umbral_fuga),
            F.lit("VENDER"),
        )
        .when(
            (F.col("prob_compra") >= umbral_compra) & (F.col("prob_baja") >= umbral_fuga),
            F.lit("RETENER PRIMERO"),
        )
        .when(
            (F.col("prob_compra") < umbral_compra) & (F.col("prob_baja") >= umbral_fuga),
            F.lit("RETENER"),
        )
        .otherwise(F.lit("NO TOCAR")),
    )
    filas = clasificado.groupBy("cuadrante").count().collect()
    total = sum(f["count"] for f in filas)
    return {
        f["cuadrante"]: {
            "clientes": f["count"],
            "pct": round(100 * f["count"] / max(total, 1), 1),
        }
        for f in filas
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Traduce las predicciones a plan comercial")
    # 0,80 y no 0,50, que fue la primera version y estaba mal pensada.
    #
    # Partir por la mediana significa declarar "riesgo alto" a la mitad peor de la
    # cartera por definicion. Con eso salia el 75% de los clientes marcados para
    # retencion, y eso no es un plan: ningun equipo comercial puede retener a tres
    # cuartas partes de su cartera. El cuadrante dejaba de decidir nada.
    #
    # El corte tiene que atarse a la CAPACIDAD de la campana, no a la forma de la
    # distribucion. Con 0,80 el cuarto superior es "alto", que ya se parece al orden de
    # magnitud de a cuanta gente se puede llamar de verdad en un mes.
    parser.add_argument("--umbral-compra", type=float, default=0.80,
                        help="Percentil de propension de compra que separa alto de bajo")
    parser.add_argument("--umbral-fuga", type=float, default=0.80)
    args = parser.parse_args()

    spark = crear_sesion("negocio")
    spark.sparkContext.setLogLevel("ERROR")
    metricas: dict = {}

    ruta_compra = DATA_EXPORT / "prob_compra"
    ruta_baja = DATA_EXPORT / "prob_baja"
    if not ruta_compra.exists():
        raise SystemExit(
            f"Falta {ruta_compra}.\n"
            "Ejecuta antes:  python src/recommend.py --exportar\n"
            "                python src/churn.py --exportar"
        )

    print("\n=== FASE 5: del modelo al plan comercial ===\n")
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    compra = spark.read.parquet(str(ruta_compra)).cache()

    # --- Curva de captura de ventas ----------------------------------------------------
    with cronometro("curva_captura_ventas", metricas):
        puntos = curva_captura(compra, "prob_compra", "contrato")
    metricas["curva_captura_ventas"] = puntos

    print("  Curva de captura -- contrataciones")
    print(f"  {'% cartera':>10} {'contactados':>12} {'% capturado':>12} {'vs azar':>9}")
    for p in puntos:
        print(f"  {p['pct_cartera']:>9.1f}% {p['clientes_contactados']:>12,} "
              f"{p['pct_capturado']:>11.1f}% {p['multiplicador']:>8.2f}x")

    dibujar_curva(
        puntos,
        OUTPUTS / "curva_captura.png",
        "Cuanto se captura segun a cuanta gente se llame",
    )

    hito = next(p for p in puntos if p["pct_cartera"] == 5.0)
    metricas["titular"] = (
        f"Contactando al 5% de la cartera ({hito['clientes_contactados']:,} clientes) "
        f"se alcanza el {hito['pct_capturado']}% de las contrataciones del mes, "
        f"{hito['multiplicador']}x lo que se conseguiria llamando al azar."
    )

    # --- Matriz valor x riesgo ---------------------------------------------------------
    if ruta_baja.exists():
        baja = (
            spark.read.parquet(str(ruta_baja))
            .groupBy("ncodpers")
            .agg(F.max("prob_baja").alias("prob_baja"))
        )
        cruce = compra.join(baja, "ncodpers", "left").fillna({"prob_baja": 0.0})

        # Los umbrales se fijan por percentil y no por un valor absoluto: lo que importa
        # es separar el tercio alto del resto, no que la probabilidad pase de 0,5.
        cortes = cruce.approxQuantile(
            ["prob_compra", "prob_baja"], [args.umbral_compra, args.umbral_fuga], 0.01
        )
        corte_compra, corte_fuga = cortes[0][0], cortes[1][0]

        with cronometro("matriz_valor_riesgo", metricas):
            matriz = matriz_valor_riesgo(cruce, corte_compra, corte_fuga)

        metricas["matriz_valor_riesgo"] = matriz
        metricas["cortes"] = {"compra": corte_compra, "fuga": corte_fuga}

        print("\n  Matriz valor x riesgo")
        acciones = {
            "VENDER": "ofrecerle el siguiente producto",
            "RETENER PRIMERO": "retener antes de vender nada",
            "RETENER": "campana de retencion",
            "NO TOCAR": "no gastar contacto este mes",
        }
        for cuadrante, datos in sorted(matriz.items(), key=lambda kv: -kv[1]["clientes"]):
            print(f"    {cuadrante:<18} {datos['clientes']:>9,} clientes "
                  f"({datos['pct']:>4.1f}%)  -> {acciones.get(cuadrante, '')}")
    else:
        print("\n  (sin prob_baja exportada: se omite la matriz valor x riesgo)")

    print(f"\n  TITULAR: {metricas['titular']}")

    guardar_metricas("business_metrics.json", metricas)
    spark.stop()


if __name__ == "__main__":
    main()
