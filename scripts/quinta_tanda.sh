#!/usr/bin/env bash
# Quinta tanda: cerrar el proyecto con el mejor modelo en toda la cadena.
#
# Hasta aqui el mejor modelo (el Transformer) solo existia en la fase de recomendacion.
# La capa de negocio, que es el titular del proyecto, seguia corriendo sobre XGBoost
# porque no habia tuberia entre el .npz de la red y el parquet que consume business.py.
# Esta tanda la usa de punta a punta.
#
# Los hiperparametros NO se escriben a mano: se leen de outputs/tuning_metrics.json, que
# es donde tuning.py dejo la configuracion elegida en el mes de parada. Copiarlos a mano
# es la forma clasica de que el numero que se publica no sea el que se midio.
set -u
mkdir -p logs

ejecutar() {
  local nombre="$1"; shift
  echo ""
  echo "======================================================================"
  echo ">>> $nombre   ($(date +%H:%M:%S))"
  echo "======================================================================"
  if python -u "$@" > "logs/${nombre}.log" 2>&1; then
    echo "    OK"
    grep -aE "MAP@7|AUC|Brier|titular|clientes|Exportado|escritos|capturad|umbral" \
      "logs/${nombre}.log" | head -20
  else
    echo "    FALLO -- ver logs/${nombre}.log"
    tail -20 "logs/${nombre}.log"
  fi
}

# --- Los hiperparametros ganadores, leidos del JSON -----------------------------------
CONFIG=$(python - <<'PY'
import json
from pathlib import Path

ruta = Path("outputs/tuning_metrics.json")
if not ruta.exists():
    raise SystemExit("FALTA outputs/tuning_metrics.json: ejecuta antes src/tuning.py")

c = json.loads(ruta.read_text(encoding="utf-8"))["mejor_config"]
print(
    f"--dim {c['dim']} --capas {c['capas']} --cabezas {c['cabezas']} "
    f"--dropout {c['dropout']} --lr {c['lr']:.6f} --wd {c['wd']} "
    f"--lote {c['lote']} --largo {c['largo']} --lambda-aux {c['lambda_aux']} "
    f"--supervision todas"
    + (" --atar-pesos" if c["atar_pesos"] else "")
)
PY
) || exit 1

echo "Configuracion ganadora: $CONFIG"

# 1. El mejor modelo sobre el mes de medicion, y la cartera completa exportada.
#    --epocas 20 con early stopping: el mes de parada decide donde cortar.
ejecutar secuencia_final src/secuencia.py --mes-validacion 2016-05-28 --epocas 20 \
  --modelos transformer --exportar-parquet $CONFIG

# 2. Las bajas, ahora con tenencia_producto (los meses que el cliente lleva con CADA
#    producto). Se anadio a features.py pero nunca se midio su efecto.
ejecutar churn_tenencia src/churn.py --exportar

# 3. La capa de negocio, por fin con el mejor modelo como fuente.
ejecutar business_final src/business.py

# 4. El fichero que necesita la demo de Streamlit.
ejecutar exportar_demo scripts/exportar_demo.py

echo ""
echo "QUINTA TANDA TERMINADA  ($(date +%H:%M:%S))"
