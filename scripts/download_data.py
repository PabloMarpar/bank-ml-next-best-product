"""
Descarga el dataset de Santander Product Recommendation desde Kaggle.

Requiere credenciales de la API de Kaggle en ~/.kaggle/kaggle.json (en Windows,
%USERPROFILE%\.kaggle\kaggle.json). Nunca se guardan en el repo.

Uso:
    python scripts/download_data.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
COMPETICION = "santander-product-recommendation"
FICHERO_PRINCIPAL = "train_ver2.csv"


def comprobar_credenciales() -> None:
    ruta = Path.home() / ".kaggle" / "kaggle.json"
    if ruta.exists():
        return
    sys.exit(
        f"\nNo se encuentra {ruta}\n\n"
        "Para crearlo:\n"
        "  1. Entra en https://www.kaggle.com/settings\n"
        "  2. Seccion 'API' -> 'Create New Token' (descarga un kaggle.json)\n"
        f"  3. Guardalo en {ruta.parent}\n"
    )


def descargar() -> None:
    # El import va dentro de la funcion porque la libreria de Kaggle se autentica en el
    # momento de importarse, y revienta con un error poco claro si no hay credenciales.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"Descargando '{COMPETICION}' en {DATA_RAW} (~2,3 GB, puede tardar)...")

    try:
        api.competition_download_files(COMPETICION, path=str(DATA_RAW), quiet=False)
    except Exception as exc:  # noqa: BLE001 - queremos traducir el error, no tragarlo
        if "403" in str(exc) or "Forbidden" in str(exc):
            sys.exit(
                "\nKaggle ha respondido 403 (Forbidden).\n\n"
                "Casi siempre significa que no has aceptado las reglas de la competicion.\n"
                f"Entra en https://www.kaggle.com/c/{COMPETICION}/rules,\n"
                "pulsa 'I Understand and Accept', y vuelve a lanzar este script.\n"
            )
        raise


def descomprimir() -> None:
    """Descomprime el zip de la competicion y, dentro, los .csv.zip que trae."""
    for zip_externo in DATA_RAW.glob("*.zip"):
        print(f"Descomprimiendo {zip_externo.name}...")
        with zipfile.ZipFile(zip_externo) as z:
            z.extractall(DATA_RAW)
        zip_externo.unlink()

    # Los ficheros de esta competicion vienen a su vez como .csv.zip
    for zip_interno in DATA_RAW.glob("*.csv.zip"):
        print(f"Descomprimiendo {zip_interno.name}...")
        with zipfile.ZipFile(zip_interno) as z:
            z.extractall(DATA_RAW)
        zip_interno.unlink()


def verificar() -> None:
    destino = DATA_RAW / FICHERO_PRINCIPAL
    if not destino.exists():
        sys.exit(f"\nEsperaba encontrar {destino} y no esta. Ficheros: "
                 f"{[f.name for f in DATA_RAW.iterdir()]}")
    gb = destino.stat().st_size / (1024**3)
    print(f"\nOK: {destino} ({gb:.2f} GB)")
    if gb < 1.5:
        print("AVISO: el fichero deberia pesar ~2,3 GB. Revisa que la descarga termino.")


if __name__ == "__main__":
    comprobar_credenciales()
    if (DATA_RAW / FICHERO_PRINCIPAL).exists():
        print(f"{FICHERO_PRINCIPAL} ya existe, no se vuelve a descargar.")
    else:
        descargar()
        descomprimir()
    verificar()
