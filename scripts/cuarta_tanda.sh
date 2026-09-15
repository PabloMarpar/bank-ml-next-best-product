#!/usr/bin/env bash
# Cuarta tanda: early stopping honesto + las dos preguntas abiertas de la Fase 3.
#
# Contexto: el refactor a tres tramos (train / parada / test) estaba construido pero sin
# cablear -- entrenar_red() recibia el mes de TEST donde debia recibir el de PARADA. Las
# ejecuciones anteriores no llegaron a sufrir la fuga porque corrian con epocas fijas y
# early stopping desactivado, pero tampoco aprovechaban el tramo de parada para nada.
#
# Ahora se cablea y se sube a 10 epocas: con la pérdida todavia bajando en la epoca 5, el
# numero de epocas era un hiperparametro sin elegir. Que lo elija el mes t-1.
#
# Despues, las dos preguntas que quedaban sin respuesta medida:
#   - ensemble: ¿mezclar los 5 modelos supera al Transformer solo?
#   - router:   ¿un modelo distinto por segmento supera al Transformer solo?
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
    grep -aE "MAP@7|MAP parada|early stopping|Clientes de|Pesos elegidos|MEDICION FINAL|aporta|veredicto|Segmento|router" \
      "logs/${nombre}.log" | head -30
  else
    echo "    FALLO -- ver logs/${nombre}.log"
    tail -15 "logs/${nombre}.log"
  fi
}

# Las redes, con el mes de parada ya cableado. Primero el mes en que el ensemble elige
# sus pesos (t-1), luego el mes de medicion final (t).
ejecutar secuencia_abril src/secuencia.py --mes-validacion 2016-04-28 --epocas 10 --dim 64
ejecutar secuencia_mayo  src/secuencia.py --mes-validacion 2016-05-28 --epocas 10 --dim 64

# La mezcla, sobre los .npz recien generados.
ejecutar ensemble src/ensemble.py --arboles 80 --profundidad 8

# El router por segmento: nunca se ha ejecutado.
ejecutar router src/router.py --arboles 80 --profundidad 8

echo ""
echo "CUARTA TANDA TERMINADA  ($(date +%H:%M:%S))"
