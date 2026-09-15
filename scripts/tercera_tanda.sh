#!/usr/bin/env bash
# Tercera tanda: arreglar el ensemble, que estructuralmente no podia usar las redes.
#
# El fallo: las predicciones de las redes secuenciales se guardaban con nombre fijo y solo
# se cargaban para el mes de medicion. Como los pesos de la mezcla se eligen en el mes
# ANTERIOR, la mezcla nunca tuvo esas redes disponibles al decidir. El "no aporta" que
# salio era un artefacto del codigo, no un hallazgo.
set -u
mkdir -p logs

ejecutar() {
  local nombre="$1"; shift
  echo ""
  echo "======================================================================"
  echo ">>> $nombre"
  echo "======================================================================"
  if python -u "$@" > "logs/${nombre}.log" 2>&1; then
    echo "    OK"
    grep -aE "MAP@7|Pesos elegidos|MEDICION FINAL|<-- mezcla|aporta|^    [a-z]+ +0\." \
      "logs/${nombre}.log" | head -14
  else
    echo "    FALLO -- ver logs/${nombre}.log"
    tail -6 "logs/${nombre}.log"
  fi
}

# Las redes, ahora tambien para el mes en que se eligen los pesos (2016-04-28).
ejecutar secuencia_abril src/secuencia.py --mes-validacion 2016-04-28 --epocas 5 --dim 64

# Y la mezcla, que por fin puede contar con las cinco.
ejecutar ensemble src/ensemble.py --arboles 80 --profundidad 8

echo ""
echo "TERCERA TANDA TERMINADA"
