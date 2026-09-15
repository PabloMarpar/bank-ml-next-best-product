#!/usr/bin/env bash
# Encadena el resto del pipeline sobre los datos reales.
#
# Se ejecuta DENTRO del contenedor. Cada paso escribe su propio log en logs/ para que un
# fallo en uno no obligue a repetir los anteriores, y para poder mirar el progreso en
# vivo: el log se escribe crudo y se filtra al leerlo, nunca al escribirlo.
#
#   docker compose run --rm -e PYTHONPATH=/app/src spark bash scripts/ejecutar_todo.sh
set -u
mkdir -p logs

ejecutar() {
  local nombre="$1"; shift
  echo ""
  echo "======================================================================"
  echo ">>> $nombre"
  echo "======================================================================"
  if python -u "$@" > "logs/${nombre}.log" 2>&1; then
    echo "    OK -- logs/${nombre}.log"
    grep -aE "MAP@7|AUC|Brier|TITULAR|RECOMENDACION|veredicto|Salto de AUC|CORRECTO|REVISAR" \
      "logs/${nombre}.log" | head -12
  else
    echo "    FALLO (continuo con el resto) -- ver logs/${nombre}.log"
    tail -5 "logs/${nombre}.log"
  fi
}

ejecutar leakage   src/leakage_check.py --arboles 25 --meses 4
ejecutar recommend src/recommend.py --arboles 80 --profundidad 8 --exportar
ejecutar secuencia src/secuencia.py --epocas 5 --dim 64
ejecutar ensemble  src/ensemble.py --arboles 80 --profundidad 8
ejecutar business  src/business.py
ejecutar renta     src/analisis_renta.py --arboles 25 --muestra 0.12
ejecutar deriva    src/deriva.py --arboles 25 --profundidad 5 --horizonte 6
ejecutar coldstart src/cold_start.py --arboles 60 --profundidad 7
ejecutar perf      src/perf.py

echo ""
echo "======================================================================"
echo "TODO EJECUTADO. Metricas en outputs/, logs completos en logs/"
echo "======================================================================"
