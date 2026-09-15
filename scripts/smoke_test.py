"""
Prueba de humo del pipeline completo sobre un dataset sintetico diminuto.

Genera un CSV con exactamente el mismo esquema y la misma suciedad que el real (espacios
de relleno, "NA" como texto, el centinela -999999 en antiguedad) pero con unos pocos miles
de filas, y ejecuta todas las fases encadenadas.

Sirve para dos cosas:
  * comprobar que la logica funciona sin esperar a un run de 20 minutos sobre 13,6M filas,
  * poder tocar el codigo con la tranquilidad de que un fallo tonto salta en 40 segundos.

    python scripts/smoke_test.py

Que esperar del resultado:

  * RECOMENDACION: los datos sinteticos llevan senal a proposito -- la probabilidad de
    contratar depende del segmento, la edad y los productos que ya se tienen. Los modelos
    supervisados (bosque y XGBoost) deben quedar por encima del baseline de popularidad.
    Si empatan o pierden, hay algo roto.

  * EL HIBRIDO DEBE EMPEORAR AQUI, Y ESO ES CORRECTO. Los productos se generan de forma
    independiente para cada cliente a partir de su segmento y su edad, sin ninguna
    estructura de co-ocurrencia entre productos. O sea: en estos datos NO existe la senal
    colaborativa que el ALS intenta capturar, asi que sus 16 vectores latentes son ruido
    y lo unico que hacen es diluir las variables buenas. Si el hibrido "mejorase" aqui,
    seria senal de que algo esta mal medido.

    Sobre el dataset real la pregunta queda abierta: los clientes de banca si contratan
    productos en combinaciones que se repiten, asi que ahi la senal colaborativa puede
    existir de verdad. Se reportara lo que salga, gane o pierda.

  * BAJAS: aqui las bajas se generan con una probabilidad fija del 2%, sin depender de
    nada. No hay senal que aprender, asi que el AUC debe rondar 0,5. Esta fase solo
    comprueba que el codigo corre; su metrica sobre datos sinteticos no significa nada.
    La calibracion si es medible aunque el modelo no prediga: el Brier score debe mejorar,
    porque calibrar corrige la magnitud incluso de un modelo que no distingue.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "data" / "_smoke"
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))
from common import NOMBRES_PRODUCTO  # noqa: E402

PRODUCTOS = list(NOMBRES_PRODUCTO)

MESES = [
    f"{anio}-{mes:02d}-28"
    for anio, mes in (
        [(2015, m) for m in range(1, 13)] + [(2016, m) for m in range(1, 6)]
    )
]

CABECERA = [
    "fecha_dato", "ncodpers", "ind_empleado", "pais_residencia", "sexo", "age",
    "fecha_alta", "ind_nuevo", "antiguedad", "indrel", "ult_fec_cli_1t", "indrel_1mes",
    "tiprel_1mes", "indresi", "indext", "conyuemp", "canal_entrada", "indfall",
    "tipodom", "cod_prov", "nomprov", "ind_actividad_cliente", "renta", "segmento",
] + PRODUCTOS

SEGMENTOS = ["01 - TOP", "02 - PARTICULARES", "03 - UNIVERSITARIO"]
CANALES = ["KAT", "KHE", "KFC", "KHQ", "KFA"]
PROVINCIAS = ["MADRID", "BARCELONA", "CORUNA, A", "VALENCIA", "SEVILLA"]


def probabilidad_alta(producto: str, segmento: str, edad: int, n_productos: int) -> float:
    """Probabilidad sintetica de contratar un producto. Con senal deliberada."""
    base = 0.004
    # Los tres primeros productos de la lista son "populares" para todo el mundo.
    if producto in PRODUCTOS[:3]:
        base = 0.05
    # Senal por segmento: los TOP contratan inversion, los universitarios cuenta junior.
    if segmento == "01 - TOP" and producto in ("ind_fond_fin_ult1", "ind_valo_fin_ult1"):
        base = 0.12
    if segmento == "03 - UNIVERSITARIO" and producto == "ind_ctju_fin_ult1":
        base = 0.15
    # Senal por edad: la hipoteca y el plan de pensiones van con la edad.
    if producto == "ind_hip_fin_ult1" and 30 <= edad <= 50:
        base = 0.08
    if producto == "ind_plan_fin_ult1" and edad > 45:
        base = 0.09
    # Quien ya tiene muchos productos contrata menos cosas nuevas.
    return base * (0.85 ** n_productos)


def generar(n_clientes: int = 1500, semilla: int = 7) -> Path:
    rng = random.Random(semilla)
    SMOKE.mkdir(parents=True, exist_ok=True)
    destino = SMOKE / "raw" / "train_ver2.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)

    perfiles = []
    for cid in range(1_000_000, 1_000_000 + n_clientes):
        perfiles.append({
            "id": cid,
            "segmento": rng.choice(SEGMENTOS),
            "edad": rng.randint(18, 85),
            "canal": rng.choice(CANALES),
            "provincia": rng.choice(PROVINCIAS),
            "sexo": rng.choice(["H", "V"]),
            "renta": rng.choice([None, round(rng.uniform(12000, 180000), 2)]),
            "alta": f"201{rng.randint(0, 4)}-0{rng.randint(1, 9)}-15",
            "estado": {p: 0 for p in PRODUCTOS},
            # Algunos clientes entran tarde, para ejercitar el filtro de mes consecutivo.
            "primer_mes": rng.choice([0, 0, 0, 0, rng.randint(1, 5)]),
        })

    filas = []
    for idx_mes, mes in enumerate(MESES):
        for p in perfiles:
            if idx_mes < p["primer_mes"]:
                continue
            # Un 2% de los clientes se salta un mes: prueba el control de huecos.
            if idx_mes > 0 and rng.random() < 0.02:
                continue

            n_prod = sum(p["estado"].values())
            for producto in PRODUCTOS:
                if p["estado"][producto] == 0:
                    if rng.random() < probabilidad_alta(
                        producto, p["segmento"], p["edad"], n_prod
                    ):
                        p["estado"][producto] = 1
                elif rng.random() < 0.02:  # baja
                    p["estado"][producto] = 0

            antiguedad = idx_mes + 10 if p["primer_mes"] == 0 else -999999
            fila = [
                mes,
                str(p["id"]),
                "N",
                "ES",
                p["sexo"],
                f"  {p['edad']}",            # espacios de relleno, como el CSV real
                p["alta"],
                "0",
                f" {antiguedad}",
                "1",
                "",
                "1.0",
                "A",
                "S",
                "N",
                "",
                p["canal"],
                "N",
                "1",
                "28",
                p["provincia"],
                str(rng.randint(0, 1)),
                "NA" if p["renta"] is None else str(p["renta"]),  # "NA" textual
                p["segmento"],
            ] + [str(p["estado"][x]) for x in PRODUCTOS]
            filas.append(fila)

    def escapar(valor: str) -> str:
        return f'"{valor}"' if "," in valor else valor

    with destino.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(CABECERA) + "\n")
        for fila in filas:
            fh.write(",".join(escapar(v) for v in fila) + "\n")

    print(f"Dataset sintetico: {len(filas):,} filas, {n_clientes:,} clientes "
          f"-> {destino}")
    return destino


def ejecutar(modulo: str, *extra: str) -> None:
    entorno = os.environ.copy()
    entorno["NBP_DATA_DIR"] = str(SMOKE)
    entorno["NBP_OUTPUTS_DIR"] = str(SMOKE / "outputs")
    entorno["PYTHONPATH"] = str(SRC)
    print(f"\n{'=' * 70}\n>>> {modulo} {' '.join(extra)}\n{'=' * 70}")
    resultado = subprocess.run(
        [sys.executable, str(SRC / modulo), *extra], env=entorno, cwd=str(ROOT)
    )
    if resultado.returncode != 0:
        sys.exit(f"\nFALLO en {modulo} (codigo {resultado.returncode})")


if __name__ == "__main__":
    if SMOKE.exists():
        shutil.rmtree(SMOKE)
    generar()
    ejecutar("ingest.py")
    ejecutar("features.py")
    ejecutar("recommend.py", "--arboles", "25", "--profundidad", "6", "--exportar")
    ejecutar("churn.py", "--arboles", "10", "--meses-entrenamiento", "4", "--exportar")
    ejecutar("business.py")
    ejecutar("leakage_check.py", "--arboles", "10", "--meses", "3")
    print("\n" + "=" * 70)
    print("PRUEBA DE HUMO COMPLETA: todas las fases funcionan de principio a fin.")
    print("=" * 70)
