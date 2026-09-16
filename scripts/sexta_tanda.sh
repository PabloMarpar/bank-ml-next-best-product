#!/usr/bin/env bash
# Sexta tanda: medir por fin el efecto de tenencia_producto en las bajas.
#
# churn.py fallo en la quinta tanda con UNRESOLVED_COLUMN sobre tenencia_ind_ahor_fin_ult1.
# La causa no estaba en churn.py: la variable se anadio a features.py hace tiempo pero
# features.py NUNCA se volvio a ejecutar, asi que el parquet de data/features no tiene esas
# columnas. Un caso de libro de dependencia entre fases que no se ve hasta que se rompe.
#
# tenencia_producto son los meses (de los ultimos 6) que el cliente lleva con CADA producto.
# Antes solo existia meses_observados, que dice cuanto llevamos viendo al cliente pero no
# cuanto lleva con el producto concreto que podria cancelar -- y eso es lo que de verdad
# predice una baja.
#
# El regenerado de features es puramente aditivo: mismas filas y mismos valores, mas las 24
# columnas nuevas. Por eso no hace falta rehacer la exportacion del Transformer, que tardo
# dos horas y media.
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
    grep -aE "AUC|Brier|TITULAR|importancia|tenencia|capturad|umbral|clientes|filas" \
      "logs/${nombre}.log" | head -22
  else
    echo "    FALLO -- ver logs/${nombre}.log"
    tail -25 "logs/${nombre}.log"
  fi
}

# 1. Regenerar features, ahora si con las columnas tenencia_*.
ejecutar features_tenencia src/features.py

# 2. Las bajas con la variable nueva. Aqui esta el experimento de verdad.
ejecutar churn_tenencia src/churn.py --exportar

# 3. La matriz valor x riesgo depende de prob_baja, asi que se rehace.
ejecutar business_final src/business.py

# 4. Y la demo, que lee las dos.
ejecutar exportar_demo scripts/exportar_demo.py

echo ""
echo "SEXTA TANDA TERMINADA  ($(date +%H:%M:%S))"
