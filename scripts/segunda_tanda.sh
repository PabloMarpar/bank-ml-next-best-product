#!/usr/bin/env bash
# Segunda tanda: lo que fallo, lo nuevo, y el ensemble rehecho con todos los modelos.
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
    grep -aE "MAP@7|AUC|Diferencia|veredicto|NO aporta|aporta|Salto|CORRECTO|REVISAR|cierra el" \
      "logs/${nombre}.log" | head -12
  else
    echo "    FALLO -- ver logs/${nombre}.log"
    tail -6 "logs/${nombre}.log"
  fi
}

# 1. El secuencial, que fallaba por un Row.get() inexistente
ejecutar secuencia src/secuencia.py --epocas 5 --dim 64

# 2. Learning to rank sobre datos reales
ejecutar ranker src/ranker.py --arboles 100 --profundidad 8 --negativos 6

# 3. Control de fuga con el veredicto corregido (metrica relativa)
ejecutar leakage src/leakage_check.py --arboles 25 --meses 4

# 4. Ensemble rehecho: ahora si tiene el secuencial disponible
ejecutar ensemble src/ensemble.py --arboles 80 --profundidad 8


# 5. Fase de negocio rehecha: los cuadrantes se cortaban por la mediana y salia el 75%
# de la cartera marcada para retencion, lo que no es un plan accionable.
ejecutar business src/business.py

echo ""
echo "======================================================================"
echo "SEGUNDA TANDA TERMINADA"
echo "======================================================================"
