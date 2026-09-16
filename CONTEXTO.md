# Contexto del proyecto (documento vivo)

> Este fichero se actualiza al terminar cada fase. Sirve para retomar el trabajo en una sesión
> nueva sin releer el histórico de conversación. El `README.md` es la narrativa pulida para
> quien visita el repo; `RESUMEN.md` es la versión sin jerga; este documento es el cuaderno de
> bitácora: qué se ha hecho, qué falta, y por qué se decidió cada cosa que no sea obvia
> leyendo el código.

## Qué es esto, en una frase

Sistema de *next best product* sobre datos reales de banca: decide a qué clientes debe llamar
un equipo comercial el mes que viene, qué producto ofrecerle a cada uno, y a cuáles está a
punto de perder. Construido en PySpark sobre 13,6M de filas reales (Santander Product
Recommendation, Kaggle). Corre en Docker porque la máquina no tiene Java.

## AL RETOMAR: el proyecto está terminado en lo técnico

**Sesión autónoma del 2026-09-15/16.** Todo el pipeline está ejecutado sobre datos reales, el
README no tiene huecos, y el repo está en GitHub. Lo que queda es decisión del usuario, no
trabajo técnico (ver "Lo que falta" al final de esta sección).

### Los titulares finales

| | Valor |
|---|---|
| **Recomendación** | Transformer causal, MAP@7 **0.90179** (desplegado) / 0.90315 (mejor semilla) |
| **Bajas** | AUC **0.8835**, **14,0x** sobre el azar |
| **Plan comercial** | Contactando al **5%** de la cartera se captura el **38,3%** de las contrataciones, **7,6x** el azar |
| **Umbral de retención** | 0,16 en vez de 0,50 → **292.033 €** frente a 104.156 € |

El titular de negocio era 16,4% y 3,22x hasta esta sesión. Se multiplicó por dos al conectar
por fin el Transformer con la capa de negocio (antes corría sobre XGBoost) y al enmascarar
los productos ya poseídos al calcular `prob_compra`.

### Los cuatro hallazgos que más dicen del proyecto

1. **Dos bugs de early stopping.** Uno era una fuga latente que nunca llegó a ejecutarse
   (detectada antes de contaminar nada). El otro era peor: la métrica de parada devolvía
   siempre 0.0, así que el entrenamiento **se cortaba en la época 1** y parecía normal.
2. **El tuning no sirvió de nada, y se puede demostrar.** 12 configuraciones sobre 9
   hiperparámetros mejoraron +0.00016 de MAP. La desviación entre 3 semillas del MISMO
   modelo fue 0.00092. La mejora está cinco veces por debajo del ruido.
3. **Lo que sí funcionó fue una variable.** `tenencia_producto` subió el AUC de bajas de
   0.8659 a 0.8835 (+0,018). Noventa veces más que toda la optimización de hiperparámetros.
4. **Cuatro enfoques probados y descartados con datos**: ALS solo, ensemble, learning-to-rank
   y router por segmento. Ninguno se queda, y todos se reportan.

### Lo que falta: dos acciones, y las dos son del usuario

Ambas son públicas e irreversibles en la práctica, así que se dejaron preparadas y **sin
ejecutar** para que las lance el usuario tras revisar.

**1. Pasar el repo a público.**
```bash
gh repo edit PabloMarpar/bank-ml-next-best-product \
  --visibility public --accept-visibility-change-consequences
```

**2. Publicar la card del portfolio.** Ya está escrita e insertada en
`pablomarpar.github.io/index.html` (commit local `d2f1041`, primera card después de la
destacada, con la curva de captura real embebida en base64). Falta solo empujarla:
```bash
cd ../pablomarpar.github.io && git push origin master
```
**En este orden**: la card enlaza al repo, así que si se empuja antes de hacerlo público,
cualquier visitante se encuentra un 404.

**Opcional:** desplegar la demo de Streamlit (`app/streamlit_app.py` +
`app/demo_data.parquet`, 4.000 clientes, 103 KB, ya generado y versionado).

**A revisar antes de publicar:** este mismo fichero, `CONTEXTO.md`, es un cuaderno de
bitácora con el detalle de los bugs y los callejones sin salida. Como documento de
ingeniería suma, pero es una decisión del usuario si quiere que sea público.

### Resultado de la búsqueda de hiperparámetros (`tuning.py`)

Búsqueda aleatoria de 12 configuraciones, todas medidas **en el mes de parada** con el 40%
del entrenamiento y tope de 8 épocas. El test no se ha tocado en toda la búsqueda.

| | MAP de parada |
|---|---|
| mejor | 0.88639 |
| mediana | 0.87989 |
| peor | 0.87245 |
| **rango** | **0.01394** |

Un rango de 0.014 no es ruido: los hiperparámetros sí importaban aquí.

**Configuración elegida:**
```
dim=128  capas=1  cabezas=4  dropout=0.1  lote=256  wd=1e-4
largo=16  atar_pesos=NO  lambda_aux=1.0  lr=0.00115
```

Tres cosas que esa configuración dice, y que merece la pena leer:

1. **`lambda_aux = 1.0`** — el valor más alto del espacio. La pérdida auxiliar (predecir el
   mes siguiente desde cada posición) pesa tanto como la tarea principal. Es la evidencia
   más directa de que la supervisión por posición era la mejora que faltaba.
2. **`largo = 16`** — todo el histórico disponible, no los 12 que estaban fijados a ojo.
3. **`capas = 1`** — una sola capa de atención gana a dos y a tres. Con secuencias de 16
   pasos y un vocabulario de 24 productos, apilar capas solo añade parámetros que
   sobreajustan.
4. **`atar_pesos = False`** — el weight tying, que se implementó a propósito para esta
   búsqueda, NO fue elegido. Se reporta igual: era una hipótesis razonable y los datos
   dicen que aquí no ayuda.

Después de la búsqueda se reentrena la ganadora con **3 semillas**, entrenamiento completo
y tope de 20 épocas, para separar la mejora real del ruido de inicialización. El modelo que
se exporta es el de mejor MAP **de parada**, nunca de test.

### ⭐ El veredicto del tuning: la mejora está POR DEBAJO del ruido

Este es el hallazgo más importante de la sesión, y es negativo.

| | MAP@7 en test |
|---|---|
| Sin tuning (dim 64, capas 2, largo 12, supervisión solo en la última posición) | 0.90299 |
| **Con tuning + supervisión por posición** | **0.90315** |
| Diferencia | **+0.00016 (+0,02%)** |

Y la **desviación entre las 3 semillas fue 0.00092** (parada: 0.88777 / 0.88689 / 0.88912).
La mejora es **cinco veces menor que el ruido de inicialización**. En la práctica: el modelo
ya estaba en su techo para esta familia de arquitecturas.

Lo que se probó para llegar a esa conclusión, y no es poco:

- supervisión por posición (12 etiquetas por cliente en vez de 1)
- weight tying entre el embedding de productos y la cabeza de salida
- calentamiento de learning rate + coseno por paso
- 12 configuraciones sobre 9 hiperparámetros
- 3 semillas para medir el ruido

**Cómo contarlo:** la parte valiosa no es el +0,02%, es haber medido la desviación entre
semillas. Sin ese número, +0.00016 se reporta como "mejora" y es mentira. Con él, la
conclusión correcta es "ya no hay nada que rascar aquí, el siguiente euro se gasta mejor
en otra parte" — que es exactamente la decisión que se toma en un equipo real.

**Nota sobre la configuración elegida:** la búsqueda se quedó con dim=128 y largo=16, que
entrena unas 4 veces más lento que el dim=64/largo=12 por defecto, para una diferencia que
está dentro del ruido. En producción la decisión defendible sería quedarse con el barato.
Aquí se usa el que eligió el procedimiento, porque cambiarlo a posteriori sería elegir a
ojo lo que se montó para no elegir a ojo.


### Los dos veredictos de la cuarta tanda (ambos negativos, ambos se reportan)

| Pregunta | Respuesta | Cifra |
|---|---|---|
| ¿La mezcla de los 5 modelos supera al Transformer solo? | **No** | 0.90294 vs 0.90299 (−0.00006) |
| ¿Un modelo distinto por segmento lo supera? | **No, en la práctica** | +0.061% por historia, +0.017% por nº de productos |

Detalle que vale la pena contar del ensemble: los pesos elegidos en t-1 fueron **50/50
entre GRU y Transformer** (0 para XGBoost y popularidad). O sea que el problema no es que
la mezcla esté mal ponderada — es que los dos modelos aciertan en los mismos clientes.

Detalle del router: la segmentación por número de productos es además **inestable**, 1 de
5 segmentos cambia de ganador entre el mes de decisión y el de medición. La segmentación
por historia es estable (0 de 5) pero su ganancia es de 0.06%. El propio script concluye
que no compensa mantener varios modelos.

### Por qué los números subieron tanto respecto a la tabla vieja

| Modelo | Antes | Ahora | Causa |
|---|---|---|---|
| GRU (mayo) | 0.88260 | **0.90157** | 3 canales + 10 épocas con early stopping real |
| Transformer (mayo) | 0.88638 | **0.90299** | idem |

Tres cosas cambiaron a la vez y conviene no atribuirlo todo al early stopping:

1. **La secuencia pasó de 1 canal a 3** (estado / altas / bajas por mes). Estaba en el
   código del refactor a medias pero nunca se había ejecutado.
2. **El early stopping empezó a funcionar** (ver los dos bugs más abajo).
3. **El entrenamiento perdió un mes**: ahora filtra `mes_idx < mes_parada`, antes llegaba
   hasta t-1. Esto juega EN CONTRA, así que la mejora real de 1 y 2 es aún mayor.

### Los dos bugs que se encontraron y arreglaron (ambos de early stopping)

**Bug 1 — fuga latente.** `entrenar_red()` recibía `datos_val` (mes de test) donde debía
recibir `datos_parada` (mes t-1). **No llegó a contaminar ningún resultado**: los logs
previos no imprimen `MAP val` en ninguna época, porque esas ejecuciones corrían con épocas
fijas y el early stopping desactivado. Era una fuga esperando a activarse. Arreglado, y el
parámetro renombrado a `datos_parada` — el bug existió porque se llamaba como el tramo
equivocado.

**Bug 2 — la métrica no medía nada.** `datos_parada` se construye sin `con_objetivo=True`
(un cliente puede contratar varios productos el mismo mes, no hay objetivo único), así que
sus `objetivos` eran todos `-1` y `_map_validacion` devolvía siempre 0.0. El efecto no era
que el early stopping no ayudase: **cortaba el entrenamiento**. Con `mejor_map = -1.0`, la
época 1 fijaba el máximo en 0.0, ninguna posterior lo superaba y a las 2 épocas paraba
devolviendo los pesos de la primera. Arreglado midiendo contra `reales` con `map_at_k`, la
misma función del resultado publicado, más una guarda que avisa si sale exactamente 0.

### Plan acordado con el usuario para lo que viene después

Si ensemble y router no superan al Transformer solo (lo esperable), el encargo es
**exprimir el Transformer y hacerlo lo mejor posible** antes de cerrar el proyecto, con
todas las buenas prácticas de optimización de hiperparámetros. El plan:

**Mejoras de modelo**
1. Supervisar TODAS las posiciones de la secuencia, no solo la última (ver hallazgo 13).
   El objetivo de cada posición j es el canal de altas de la posición j+1, que ya está
   dentro del tensor de entrada — no hace falta tocar la parte de Spark.
2. Pérdida BCE multi-etiqueta en vez de `explode` + cross-entropy: un cliente que contrata
   3 productos deja de ser 3 filas idénticas con un objetivo cada una.
3. Atar pesos (weight tying) entre la proyección de entrada y la capa de salida, como en
   SASRec: menos parámetros y mejor generalización.
4. Fusionar las variables estáticas en cada posición, no solo al final.
5. Probar secuencias más largas (`LARGO_SECUENCIA` 12 → hasta 16, que es todo el histórico).

**Búsqueda de hiperparámetros** (`src/tuning.py`, por escribir)
6. Búsqueda aleatoria sobre dim, capas, cabezas, dropout, lr, weight_decay y tamaño de
   lote, **toda ella medida en el mes de parada**. El mes de test se toca UNA vez, al
   final, con la configuración ya elegida. Cualquier otra cosa es elegir mirando la
   respuesta.
7. Calentamiento de learning rate + coseno (ahora solo hay coseno).
8. Varias semillas para separar mejora real de ruido de inicialización.

**Regla que no se rompe:** ningún hiperparámetro se elige mirando el mes de test. Se elige
en t-1 y se aplica a ciegas en t.

## Estado actual (última actualización: 2026-09-16, sesión autónoma: bugs, tanda y tuning)

| Fase / módulo | Estado |
|---|---|
| 0 — Entorno Docker | ✅ construido y probado (imagen 11 GB, PyTorch incluido) |
| 0 — Descarga de datos | ✅ hecha, 2,3 GB en `data/raw/` |
| 1 — Ingesta CSV→Parquet | ✅ ejecutado sobre datos reales: 13.647.309 filas |
| 2 — Features y objetivos | ✅ ejecutado: 12.682.421 filas utilizables, 946.089 clientes |
| 2b — `tenencia_producto` | ✅ añadida a `features.py` (meses que el cliente lleva con CADA producto) |
| Control de fuga (`leakage_check.py`) | ✅ ejecutado y con el veredicto corregido (ver abajo) |
| 3 — Recomendación (`recommend.py`) | ✅ 5 enfoques ejecutados sobre datos reales |
| 3b — Secuencial (`secuencia.py`) | ✅ dos bugs arreglados y re-ejecutado: Transformer **0.90299**, GRU **0.90157** |
| 3c — Ensemble (`ensemble.py`) | ✅ re-ejecutado sobre los `.npz` nuevos: **NO aporta** (−0.00006) |
| 3d — Learning to rank (`ranker.py`) | ✅ ejecutado: NO mejora sobre el clasificador |
| 3e — Router por segmento (`router.py`) | ✅ ejecutado por fin: **+0,06%**, no compensa mantener varios modelos |
| 4 — Bajas (`churn.py`) | ✅ re-ejecutado con `tenencia_producto`: AUC **0.8835**, **14,0x** sobre el azar |
| 5 — Negocio (`business.py`) | ✅ al día y sobre el Transformer: **38,3% al 5% de cartera, 7,56x** |
| 6 — Rendimiento (`perf.py`) | ✅ ejecutado, `PERFORMANCE.md` escrito con los 7 experimentos |
| Análisis: renta ausente | ✅ ejecutado (`analisis_renta.py`) |
| Análisis: deriva temporal | ✅ ejecutado (`deriva.py`) |
| Análisis: cold start | ✅ ejecutado (`cold_start.py`) — sin el Transformer, solo XGBoost |
| 7 — SageMaker | ✅ escrito (no ejecutado en AWS, declarado así) |
| 8 — Demo Streamlit | ✅ `app/demo_data.parquet` generado (4.000 clientes, 103 KB) y versionado |
| 8 — README.md | ✅ **sin huecos**: sección de resultados reescrita con los 8 enfoques |
| Card en el portfolio | ⏳ pendiente |
| Pasar el repo a público | ⏳ pendiente, acordado hacerlo **al terminar** el proyecto |
| Repo en GitHub | ✅ **subido**: `PabloMarpar/bank-ml-next-best-product`, PRIVADO, rama `main` |
| 3f — Tuning (`tuning.py`) | ✅ 12 configuraciones + 3 semillas: **la mejora queda bajo el ruido** |
| Puente red → negocio | ✅ `secuencia.py --exportar-parquet` puntúa los 926.663 clientes |
| `RESUMEN.md` | ✅ actualizado con los resultados finales |
| `CLAUDE.md` | ✅ **nuevo**: lo carga Claude Code solo al abrir sesión; importa `CONTEXTO.md` con `@` |

## Entorno y cómo se ejecuta

Máquina sin Java, todo corre en Docker (imagen ya construida: `next-best-product-banking-spark`,
~11 GB con PyTorch). **Ahora mismo no hay ningún contenedor corriendo.**

```bash
# Comprobar que la imagen existe
docker images | grep next-best-product-banking-spark

# Patrón de ejecución de cualquier script (MSYS_NO_PATHCONV=1 es obligatorio en Git Bash
# de Windows, si no las rutas /app/... se reescriben como C:\...)
cd next-best-product-banking
MSYS_NO_PATHCONV=1 docker compose run --rm -e PYTHONPATH=/app/src spark python -u src/<script>.py [args] > logs/<nombre>.log 2>&1
```

Config de Spark: `local[*]`, 9 GB de driver (bajado de 12 GB tras un incidente de OOM),
64 particiones de shuffle. `docker-compose.yml` monta `~/.kaggle` en modo lectura para las
credenciales (token nuevo formato `KGAT_...` en `~/.kaggle/access_token`, no el `kaggle.json`
viejo — Kaggle cambió el sistema de auth a mitad de proyecto).

**Prueba rápida sin tocar datos reales** (sintéticos, ~2 min):
```bash
MSYS_NO_PATHCONV=1 docker compose run --rm -e PYTHONPATH=/app/src spark python scripts/smoke_test.py
```

Hay tres scripts de "tanda" ya usados para encadenar ejecuciones largas (`scripts/ejecutar_todo.sh`,
`scripts/segunda_tanda.sh`, `scripts/tercera_tanda.sh`) — sirven de plantilla si hace falta
relanzar varios pasos seguidos con logging a fichero por paso.

## Resultados medidos sobre el dataset real (13,6M filas)

### Recomendación — 7 enfoques comparados (MAP@7, solo sobre clientes que compraron algo)

Medido en 2016-05-28 con el entrenamiento cortado en t-2 y el early stopping decidido en
t-1. **Los tres canales y el early stopping arreglado ya están dentro de estos números.**

| Enfoque | MAP@7 | vs popularidad | Escala Kaggle* |
|---|---|---|---|
| **Transformer causal (secuencial)** | **0.90299** | **1,40x** | **≈0.0272** |
| GRU (secuencial) | 0.90157 | 1,40x | ≈0.0271 |
| Mezcla de los 5 (ensemble) | 0.90294 | 1,40x | ≈0.0272 |
| Enrutado por segmento de historia | 0.90355 | 1,41x | ≈0.0272 |
| XGBoost multiclase | 0.85783 | 1,33x | ≈0.0258 |
| XGBoost + ALS (híbrido corregido) | 0.8606 | 1,34x | ≈0.0259 |
| XGBoost ranker (`rank:ndcg`) | 0.8517 | 1,32x | ≈0.0256 |
| Bosque aleatorio | 0.8332 | 1,30x | ≈0.0251 |
| Popularidad (baseline) | 0.64302 | — | ≈0.0194 |
| ALS solo (sobre posesión) | 0.5283 | 0,82x | ≈0.0159 |

`*` Kaggle mide MAP@7 sobre TODA la cartera (contando ceros); aquí se mide solo sobre quien
compró algo (27.875 de 926.663 clientes = 3,01%). Conversión: `MAP_aquí × 3,01% ≈ MAP_Kaggle`.
Referencia: ganador de la competición 0.031, mediana de ~1.800 equipos ≈0.025.

El ensemble y el enrutado aparecen en la tabla por transparencia, pero **ninguno se queda**:
sus diferencias con el Transformer solo (−0.00005 y +0.00056) no justifican mantener varios
modelos en producción. Ver los veredictos completos arriba.

**Learning-to-rank (`rank:ndcg` con `SparkXGBRanker`):** 0.8517 vs 0.8609 del clasificador
multiclase → **−1,06%, no mejora**. El ranker paga dos peajes: solo entrena con grupos que
tienen algún positivo, y el muestreo de negativos le quita contexto.

**Tres enfoques probados y descartados con datos** (ALS solo, ensemble, ranker) más uno que
aporta tan poco que tampoco se queda (router). Eso es parte del resultado, no un fracaso:
la alternativa habría sido montar una arquitectura de cuatro modelos para ganar 0,06%.

### Caída de negocio (`churn.py`) — ya con `tenencia_producto`

| Métrica | Antes | **Ahora** |
|---|---|---|
| AUC | 0.8659 | **0.8835** |
| Precisión media (AP) | 0.262 | **0.3526** (tasa base 0.0253) |
| Mejora sobre el azar | 10,4x | **14,0x** |
| Brier antes/después de calibrar | — | 0.02022 → 0.01977 |

La mejora la trae `tenencia_producto`: los meses (de los últimos 6) que el cliente lleva con
CADA producto. Antes solo existía `meses_observados`, que dice cuánto llevamos viendo al
cliente pero no cuánto lleva con el producto que podría cancelar.

**Por qué tardó tanto en medirse:** `churn.py` fallaba con `UNRESOLVED_COLUMN` sobre
`tenencia_ind_ahor_fin_ult1`. La causa no estaba en `churn.py` — la variable se añadió a
`features.py` pero **`features.py` nunca se volvió a ejecutar**, así que el parquet de
`data/features` no tenía esas columnas. Dependencia entre fases que no se ve hasta que rompe.

Umbral por valor de negocio (0.16) frente al umbral por inercia (0.50):

| Umbral | Contactos | Bajas capturadas | Valor esperado |
|---|---|---|---|
| **0,16 (óptimo)** | 45.727 | 14.463 (46,4%) | **292.033 €** |
| 0,50 (inercia) | 4.916 | — | 104.156 € |

Diferencia: **187.877 €** por no dejar el umbral donde viene por defecto.

### Fase de negocio (`business.py`) — al día, y con el modelo bueno

**Titular final:** *Contactando al 5% de la cartera (46.993 clientes) se alcanza el 38,3% de
las contrataciones del mes, 7,56x lo que se conseguiría llamando al azar.*

Curva de captura completa (sobre los 926.663 clientes del mes, no solo los que compraron):

| % cartera | Clientes | % capturado | Multiplicador |
|---|---|---|---|
| 1% | 9.399 | 17,0% | **15,79x** |
| 2% | 18.797 | 25,5% | 12,40x |
| **5%** | **46.993** | **38,3%** | **7,56x** |
| 10% | 93.986 | 48,9% | 4,87x |
| 20% | 185.826 | 57,7% | 2,88x |
| 50% | 463.715 | 81,8% | 1,63x |

Matriz valor × riesgo, cortada por percentil 0.80 (no por la mediana):

| Cuadrante | Clientes | % |
|---|---|---|
| NO TOCAR | 584.286 | 63,1% |
| RETENER | 152.647 | 16,5% |
| VENDER | 146.487 | 15,8% |
| RETENER PRIMERO | 43.243 | 4,7% |

**De dónde sale el salto desde el 16,4% / 3,22x anterior.** Dos causas, y conviene separarlas:

1. **El modelo.** Antes `prob_compra` venía de XGBoost (MAP 0.861); ahora del Transformer
   (0.902).
2. **El enmascarado.** `recommend.py` calculaba `prob_compra` como el máximo de la
   probabilidad sobre los 24 productos, **incluidos los que el cliente ya tiene**. A mucha
   gente le asignaba propensión alta por un producto ya contratado. El export nuevo enmascara
   los poseídos antes de la softmax — y aquí no es opcional: el modelo se entrenó con esa
   máscara puesta, así que nunca aprendió a bajar ese logit por su cuenta.


### Otros análisis (ejecutados, con resultado)

- **Fuga de datos** (`leakage_check.py`, veredicto YA corregido a métrica relativa): modelo
  limpio AUC 0.9379, modelo con fuga deliberada AUC 1.0000 → "la fuga cierra el 100% del error
  que le quedaba al limpio". Correcto.
- **Renta ausente** (`analisis_renta.py`): NO es aleatoria (AUC 0.763 prediciendo si falta),
  la explica sobre todo la **provincia** (0.738 de importancia) — probable artefacto operativo,
  no de comportamiento. Imputar con modelo solo mejora un 3,2% sobre la mediana → no compensa.
- **Deriva temporal** (`deriva.py`): el modelo pierde +0.0043 de AUC en 6 meses sin
  reentrenar. Recomendación con datos: **reentrenar cada 4 meses** (a partir de ahí el coste
  supera 0.01 de AUC).
- **Cold start** (`cold_start.py`): el modelo completo (XGBoost) gana en TODOS los tramos de
  historial, incluso con clientes de 1 mes. Un segundo modelo demográfico no aporta (+0,00%).
  **Ejecutado sin el Transformer** — con él en la mezcla podría cambiar, no se ha comprobado.
- **Rendimiento de Spark** (`perf.py` → `PERFORMANCE.md`): Parquet 6,2x más rápido que CSV
  (11,1x menos espacio), broadcast join 7,1x más rápido que sort-merge, cache 2,3x, sesgo de
  particiones sano (1,16x sobre la media). Más dos hallazgos de bugs propios: sacar una acción
  de Spark de un bucle dio 31x, y quitar one-hot de los árboles bajó el vector de ~250 a ~45
  dimensiones.

## Decisiones y hallazgos que no son obvios leyendo el código

1. **Todas las columnas se leen como texto y se castean a mano** (`ingest.py`). El CSV trae
   `NA` textual, espacios de relleno y el centinela `-999999`. Declarar el tipo final en el
   esquema haría que Spark convirtiera en null lo que no encaje sin avisar.

2. **`lag()` no garantiza el mes anterior** (`features.py`). Se guarda también el lag de la
   fecha y solo se conservan saltos de exactamente un mes; sin esto habría resultados mal
   calculados en silencio, no un error.

3. **`tenencia_producto`** (nueva, en `features.py`): meses de los últimos 6 que el cliente
   lleva con CADA producto. Antes solo existía `meses_observados` (cuánto llevamos viendo al
   cliente), que no dice cuánto lleva con el producto concreto que podría cancelar — y eso
   es lo más predictivo de una baja. Se propaga por `churn.py` metiéndola dentro del `struct`
   que se explota por producto, para que cada fila se quede con la tenencia de SU producto.

4. **Bosque aleatorio y XGBoost para recomendar, GBT para bajas.** El `GBTClassifier` de
   Spark solo es binario; recomendar entre 24 productos es multiclase.

5. **`etapas_preprocesado()` en `common.py` ya NO usa one-hot por defecto** (parámetro
   `one_hot: bool = False`). Los árboles manejan categóricas nativamente vía la metadata de
   `StringIndexer`; el one-hot es para modelos lineales y aquí solo inflaba el vector de
   características (~250 dimensiones con los 161 valores de `canal_entrada`) sin necesidad.
   Existe `agrupar_categorias_raras()` para recortar la cola larga de una categórica antes de
   indexarla (usado en `churn.py` sobre `canal_entrada` y `nomprov`, tope 30 + "OTROS").

6. **Calibración isotónica en bajas, sobre un TERCER tramo temporal** que el modelo no ve al
   entrenar. La isotónica es monótona pero NO estrictamente creciente (es escalonada), así que
   el AUC se mueve un poco (deriva medida y reportada, no se afirma que "no cambia nada").

7. **La curva de captura usa `approxQuantile` + una agregación, no una ventana ordenada.**
   Una ventana sin `partitionBy` fuerza juntar toda la tabla en un ejecutor.

8. **El híbrido ALS+XGBoost falló primero por un desajuste de cobertura entrenamiento/
   servicio**, no por mala idea: el ALS se entrenaba sobre "altas" (pocos clientes) y el
   ranker se entrena solo con filas de gente que compró algo, así que en entrenamiento casi
   todos tenían vector ALS y en producción solo el 21%. Se corrigió entrenando el ALS sobre
   "posesión" (`prev_*` en vez de `alta_*`), lo que sube la cobertura a ~99%/~76% (todavía no
   perfecto, el código avisa solo cuando el desajuste supera 15 puntos). Tras el arreglo el
   híbrido pasa de -24,6% a empatar con XGBoost solo (no supera, pero deja de perjudicar).

9. **`MESES_POR_INDICE` en `ensemble.py` y el nombrado `probs_secuencia_<fecha>.npz`**: la
   primera versión del ensemble guardaba las predicciones de las redes con un nombre fijo, sin
   la fecha, y solo para el mes de test. Como los pesos de la mezcla se eligen en el mes
   ANTERIOR (t-1), el ensemble nunca podía usar las redes al decidir los pesos — el veredicto
   "el ensemble no aporta" de la primera ejecución era un artefacto del código, no un hallazgo
   real. Se corrigió nombrando los `.npz` por fecha y generándolos para los dos meses
   (`secuencia.py --mes-validacion <fecha>`). El segundo veredicto (con los 5 modelos
   disponibles en los dos tramos) SÍ es de fiar: el ensemble no aporta porque el Transformer
   domina, no por un bug.

10. **Bug activo pendiente de arreglar** (ver sección de arriba): `entrenar_red()` en
    `secuencia.py` usa `datos_val` (mes de test) en vez de `datos_parada` (mes t-1) para decidir
    cuándo parar el entrenamiento — fuga de datos en la elección del hiperparámetro "número de
    épocas". El patrón de tres tramos (train/parada/test) ya está construido en `main()`, solo
    falta cablear la llamada.

11. **Docker en Windows + Git Bash**: hay que anteponer `MSYS_NO_PATHCONV=1` a cualquier
    `docker compose run` con rutas tipo `/app/...`, si no MSYS las reescribe como rutas de
    Windows y Spark falla con `No FileSystem for scheme "C"`.

12. **Kaggle cambió su sistema de autenticación** a mitad de proyecto: el token nuevo
    (`KGAT_...`) va en `~/.kaggle/access_token` (solo el token, sin JSON), no en el
    `kaggle.json` de usuario+clave de siempre. `scripts/download_data.py` acepta ambos formatos.

13. **El Transformer paga la atención causal y no la aprovecha.** `ModeloTransformer.forward()`
    aplica una máscara triangular y luego se queda solo con `salida[:, -1, :]` — la última
    posición. Con eso, la máscara causal no evita ninguna fuga real (todos los meses de la
    ventana son anteriores al objetivo de todas formas), solo recorta capacidad. Lo que SÍ
    habilita la máscara, que es entrenar prediciendo el mes j+1 desde cada posición j, no se
    está usando: la pérdida se calcula una sola vez por cliente, sobre el último mes. Es el
    diseño de SASRec a medias. Supervisar las 12 posiciones daría ~12x señal de entrenamiento
    por el mismo coste de forward. **Es la mejora candidata número 1.**

    Dos ideas que acompañan a la anterior: (a) el `explode` de `producto_objetivo` duplica la
    misma fila de entrada una vez por producto contratado — una BCE multi-etiqueta la dejaría
    en una sola fila con varios positivos; (b) el entrenamiento filtra `n_altas > 0`, así que
    los meses sin compra no aportan nada, cuando con supervisión por posición sí podrían.

    (Lo que NO hay que tocar: el enmascarado de productos ya poseídos antes de la softmax ya
    está implementado, tanto en entrenamiento como en `puntuar_enmascarado`.)

14. **No hay puente entre las predicciones del Transformer y el resto del pipeline.**
    `secuencia.py` guarda un `.npz` con `ncodpers` + matriz de probabilidades. Pero
    `business.py` y `scripts/exportar_demo.py` leen un **parquet** en
    `data/export/prob_compra`, que produce `recommend.py` (XGBoost). No existe conversión
    entre los dos formatos. Por eso la capa de negocio sigue corriendo sobre XGBoost aunque
    el Transformer sea mejor: no es que esté desactualizada, es que le falta la tubería.
    Decisión pendiente: escribir la conversión npz→parquet, o documentar a propósito que la
    capa de negocio usa XGBoost (defendible: es el modelo que ya está en el flujo Spark y la
    diferencia de MAP es de ~3%).

Nota: `DIARIO_PROBLEMAS.md` contiene el detalle técnico de doce incidencias más; se mantiene
fuera del repositorio a propósito.

Hay **12 incidencias más** con detalle técnico y "cómo lo cuento si me preguntan" en
`DIARIO_PROBLEMAS.md` (notas privadas de trabajo, cubiertas por `.gitignore`).


## Próximos pasos inmediatos (en orden)

Los pasos 1-2 de la lista anterior **ya están hechos** (fix de la fuga) y los pasos 2-4 están
corriendo ahora en `scripts/cuarta_tanda.sh`. Lo que queda, en orden:

1. **Leer el resultado de la cuarta tanda** y decidir: ¿ensemble o router superan al
   Transformer solo? Comparar contra el respaldo en `%TEMP%\nbp_backup_pre_cuarta`.
2. **Si no lo superan (lo esperable): mejorar el Transformer por dentro.** Plan acordado con
   el usuario. La idea con más recorrido, ya identificada leyendo el código (ver hallazgo 13
   más abajo): **supervisar todas las posiciones de la secuencia**, no solo la última.
3. Re-ejecutar `churn.py --exportar` para medir el efecto real de `tenencia_producto`.
4. **Decidir y construir el puente Transformer → `prob_compra` parquet** (ver hallazgo 14).
   Sin esto, `business.py` y la demo no pueden usar el mejor modelo.
5. Re-ejecutar `business.py` con esa fuente — es el titular final del proyecto.
6. Rellenar los 4 huecos de cifras en `README.md`: `<!-- NUMEROS_FUGA -->` (línea 48),
   `<!-- RESULTADO_HIBRIDO_CORREGIDO -->` (122), `<!-- NUMEROS_CHURN -->` (126),
   `<!-- NUMEROS_NEGOCIO -->` (130).
7. `scripts/exportar_demo.py` → generar `app/demo_data.parquet` para la demo Streamlit.
8. **Arreglar `.gitignore`** (ver abajo), actualizar `CONTEXTO.md` y commit local.
9. Añadir card en `pablomarpar.github.io/index.html` (patrón ya identificado: línea ~647).
10. **Confirmar con el usuario antes de subir nada a GitHub** — es la única decisión que no
    se toma sola.

## Ficheros que no van a GitHub (repasar antes de cualquier commit/push)

⚠️ **`.gitignore` tiene un hueco real, comprobado con `git status`:** `logs/` NO está
ignorado (aparece como untracked) y `*.npz` tampoco tiene regla propia (ahora solo lo tapa
`data/` por la ruta). Hay que añadir ambos antes del primer commit.

`DIARIO_PROBLEMAS.md` y `NOTAS_ENTREVISTA.md` sí están bien cubiertos, igual que `data/`,
`*.parquet` y las credenciales de Kaggle. Un `git status` con los ojos antes de cada commit
sigue siendo obligatorio.
