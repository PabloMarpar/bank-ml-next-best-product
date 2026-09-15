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

## ⚠️ AL RETOMAR: qué hay en vuelo ahora mismo

**Sesión del 2026-09-15 (segunda).** El bug de fuga en `secuencia.py` **ya está
arreglado** y hay una tanda larga ejecutándose. Detalle:

### El bug, arreglado
`entrenar_red()` recibía `datos_val` (mes de test) donde debía recibir `datos_parada`
(mes t-1). Arreglado en las dos llamadas de `main()` y, además, se renombró el parámetro
dentro de la función a `datos_parada` (y `map_val` → `map_parada`): el bug ocurrió
precisamente porque el parámetro se llamaba como el tramo equivocado.

### Corrección importante al diagnóstico anterior
La versión previa de este documento decía que los MAP del Transformer/GRU estaban
contaminados. **No lo estaban.** Los logs (`logs/secuencia.log`, `logs/secuencia_abril.log`)
no imprimen `MAP val` en ninguna época ni la línea "Clientes de parada": ambos runs
entrenaron **5 épocas fijas con early stopping desactivado**. La fuga estaba *latente* en
el refactor a medias, pero nunca llegó a ejecutarse. Los números 0.8864 / 0.8826 eran
limpios.

Se re-ejecuta igualmente, por otro motivo: ahora el early stopping sí funciona, y con la
pérdida todavía bajando en la época 5 (1.29, sin aplanar) el número de épocas era un
hiperparámetro sin elegir. La tanda va a **10 épocas** y deja que el mes t-1 decida.

Ojo al comparar: el MAP puede moverse por **dos** causas a la vez, no solo por el early
stopping. El tramo de entrenamiento ahora filtra `mes_idx < mes_parada` (antes llegaba
hasta t-1), así que también **pierde un mes de datos**.

### Segundo bug encontrado y arreglado: el early stopping no medía nada

La primera cuarta tanda se abortó a los 20 min al ver `MAP parada 0.0` en todas las épocas.
Causa: `datos_parada = a_numpy(filas_parada)` se construye **sin** `con_objetivo=True` (un
cliente puede contratar varios productos el mismo mes, así que no hay objetivo único), por
lo que sus `objetivos` son todos `-1`. `_map_validacion` comparaba contra esa columna y
devolvía siempre 0.

El efecto no era que el early stopping no ayudara: **cortaba el entrenamiento**. Con
`mejor_map = -1.0`, la época 1 fijaba el máximo en 0.0, ninguna posterior lo superaba, y a
las 2 épocas paraba devolviendo los pesos de la primera. El GRU entrenó 3 épocas de 10 y
devolvió el estado de la 1.

Arreglado: `_map_validacion` mide contra `reales` reutilizando `map_at_k`, la misma función
con la que se publica el resultado final. Más una guarda que avisa si el MAP de parada sale
exactamente 0 en la primera época. Commit `96a73c4`.

### Tercer hallazgo: el refactor a medias traía más cambios de los documentados

La secuencia ya no tiene 1 canal sino **3** (`CANALES = 3`: estado / altas / bajas por mes).
Se ve en el log: 377 MB de secuencias frente a 135 MB, y la pérdida arranca en 1.20 en vez
de 1.54. **Los 0.8864 / 0.8826 de la tabla de abajo son con UN canal.** Los nuevos números
no son comparables uno a uno con los viejos: cambian a la vez el nº de canales, el mes de
entrenamiento que se pierde y el early stopping.

### Tanda en curso: `scripts/cuarta_tanda.sh` (segundo lanzamiento, 21:06)

Cuatro pasos secuenciales. Log maestro `logs/cuarta_tanda.log`.

| Paso | Estado | Resultado |
|---|---|---|
| 1. `secuencia.py --mes-validacion 2016-04-28 --epocas 10` | ✅ | GRU **0.88534**, Transformer **0.88507** |
| 2. `secuencia.py --mes-validacion 2016-05-28 --epocas 10` | 🔄 en curso | — |
| 3. `ensemble.py` | ⏳ | — |
| 4. `router.py` (primera ejecución de su vida) | ⏳ | — |

Lo interesante de abril: **el GRU gana al Transformer por poco** (0.88534 vs 0.88507), al
revés que antes. Y el early stopping por fin actúa de verdad — el Transformer paró en la
época 8 quedándose con la 6. En el GRU el MAP de parada seguía subiendo en la 10, así que
10 épocas siguen quedándose cortas.

**Respaldo de los resultados previos** en `%TEMP%\nbp_backup_pre_cuarta`: los dos `.npz`,
`secuencia_metrics.json`, `ensemble_metrics.json` y los tres logs.

### ⚠️ Trabajo escrito que AÚN NO está en el repo (si se pierde la sesión, está aquí)

`src/secuencia.py` no se puede editar mientras la tanda lo ejecuta (lo lanza dos veces;
cambiarlo a mitad rompería la comparación). Así que la versión nueva está esperando en:

```
%TEMP%\claude\c--Users-pablo-mparera-Desktop-portfolio-projects-next-best-product-banking\
  b5571111-cce1-44ae-b840-bc92796f857b\scratchpad\secuencia_nuevo.py
```

929 líneas, ya validada con `ast.parse`. **Cuando termine la tanda: copiar encima de
`src/secuencia.py`, pasar el smoke test y commitear.** Lo que añade:

- `LARGO_MAXIMO = 16` y `recortar_largo()`: los tensores se preparan una vez al largo
  máximo y cada configuración se queda con la cola que necesita, sin repetir Spark.
- `a_numpy()` devuelve además `longitudes` (meses reales frente a relleno). Las tuplas de
  datos pasan de 4 a 5 elementos.
- `preparar_tramos()`: la preparación de los tres tramos, extraída de `main()` para que
  `tuning.py` la pueda reutilizar.
- Modelos configurables (`capas`, `cabezas`, `dropout`, `largo`, `atar_pesos`) y
  `fabricar_arquitectura()`.
- **Dos cabezas.** La principal ve secuencia + estáticas. La auxiliar ve SOLO la
  secuencia, y predice el mes siguiente desde cada posición intermedia. La auxiliar no
  puede ver las estáticas porque describen al cliente en el mes objetivo: dárselas a una
  posición del pasado sería información futura respecto a lo que esa posición predice.
- **Weight tying** opcional: la cabeza auxiliar reutiliza la matriz de embedding de altas
  en vez de aprender una segunda tabla de 24×dim.
- **Supervisión por posición** (`--supervision todas`) con tres máscaras: relleno,
  última posición, y productos ya poseídos en esa posición.
- **Calentamiento + coseno por paso**, no por época (con 8-20 épocas, un coseno de 8
  escalones no es una curva).

También están escritos y sin commitear `src/tuning.py` (búsqueda aleatoria) y la
extensión de `scripts/smoke_test.py` para cubrir ambos. No se commitean todavía porque
`tuning.py` importa funciones que solo existen en la versión nueva de `secuencia.py`.

### La pregunta que responde esta tanda
¿Ensemble o router superan al Transformer solo? Si la respuesta es no en ambos —que es lo
esperable— el plan acordado con el usuario es **dejar de añadir modelos encima y mejorar el
Transformer por dentro**, y solo después cerrar el resto del proyecto.


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

## Estado actual (última actualización: 2026-09-15, segunda sesión — fix de fuga + cuarta tanda)

| Fase / módulo | Estado |
|---|---|
| 0 — Entorno Docker | ✅ construido y probado (imagen 11 GB, PyTorch incluido) |
| 0 — Descarga de datos | ✅ hecha, 2,3 GB en `data/raw/` |
| 1 — Ingesta CSV→Parquet | ✅ ejecutado sobre datos reales: 13.647.309 filas |
| 2 — Features y objetivos | ✅ ejecutado: 12.682.421 filas utilizables, 946.089 clientes |
| 2b — `tenencia_producto` | ✅ añadida a `features.py` (meses que el cliente lleva con CADA producto) |
| Control de fuga (`leakage_check.py`) | ✅ ejecutado y con el veredicto corregido (ver abajo) |
| 3 — Recomendación (`recommend.py`) | ✅ 5 enfoques ejecutados sobre datos reales |
| 3b — Secuencial (`secuencia.py`) | 🔄 fuga arreglada; **re-ejecutándose** con 10 épocas y early stopping real |
| 3c — Ensemble (`ensemble.py`) | 🔄 ejecutado (veredicto: NO aporta); **re-ejecutándose** sobre los `.npz` nuevos |
| 3d — Learning to rank (`ranker.py`) | ✅ ejecutado: NO mejora sobre el clasificador |
| 3e — Router por segmento (`router.py`) | 🔄 escrito; **ejecutándose por primera vez** en la cuarta tanda |
| 4 — Bajas (`churn.py`) | ✅ ejecutado y optimizado; **falta re-ejecutar con `tenencia_producto`** |
| 5 — Negocio (`business.py`) | ⚠️ parcialmente al día: **ya re-ejecutado con el umbral 0.80** (matriz sana: 6,1 / 63,8 / 15,4 / 14,7%). Lo que sigue pendiente es la **fuente**: lee `prob_compra` de XGBoost, no del Transformer |
| 6 — Rendimiento (`perf.py`) | ✅ ejecutado, `PERFORMANCE.md` escrito con los 7 experimentos |
| Análisis: renta ausente | ✅ ejecutado (`analisis_renta.py`) |
| Análisis: deriva temporal | ✅ ejecutado (`deriva.py`) |
| Análisis: cold start | ✅ ejecutado (`cold_start.py`) — sin el Transformer, solo XGBoost |
| 7 — SageMaker | ✅ escrito (no ejecutado en AWS, declarado así) |
| 8 — Demo Streamlit | ⏳ código escrito, **falta generar `demo_data.parquet`** |
| 8 — README.md | ✅ escrito, con **huecos de cifras pendientes de rellenar** (marcados `<!-- ... -->`) |
| Card en el portfolio | ⏳ pendiente |
| Subir a GitHub | ⏳ **pendiente, esperando confirmación del usuario** — nada se ha subido |
| Commit local | ⏳ hay cambios grandes sin commitear (ver `git status` abajo) |
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

### Recomendación — 6 enfoques comparados (MAP@7, solo sobre clientes que compraron algo)

| Enfoque | MAP@7 | vs popularidad | Escala Kaggle* |
|---|---|---|---|
| **Transformer causal (secuencial)** | **0.8864** | **1,38x** | **≈0.0267** |
| GRU (secuencial) | 0.8826 | 1,37x | ≈0.0266 |
| XGBoost multiclase | 0.8609 | 1,34x | ≈0.0259 |
| XGBoost + ALS (híbrido corregido) | 0.8606 | 1,34x | ≈0.0259 |
| XGBoost ranker (`rank:ndcg`) | 0.8517 | 1,32x | ≈0.0256 |
| Bosque aleatorio | 0.8332 | 1,30x | ≈0.0251 |
| Popularidad (baseline) | 0.6431 | — | ≈0.0194 |
| ALS solo (sobre posesión) | 0.5283 | 0,82x | ≈0.0159 |

`*` Kaggle mide MAP@7 sobre TODA la cartera (contando ceros); aquí se mide solo sobre quien
compró algo (27.875 de 926.663 clientes = 3,01%). Conversión: `MAP_aquí × 3,01% ≈ MAP_Kaggle`.
Referencia: ganador de la competición 0.031, mediana de ~1.800 equipos ≈0.025.

**⚠️ Los números del Transformer/GRU de esta tabla son PRE-arreglo de la fuga de early
stopping** descrita arriba. Hay que re-ejecutar `secuencia.py` tras el fix y actualizar esta
tabla.

**Ensemble de los 5 modelos:** pesos elegidos honestamente en el mes t-1 (90% Transformer,
10% GRU, 0% el resto) → en el mes de test da exactamente el mismo MAP que el Transformer solo
(0.88638). **Conclusión: no aporta**, el Transformer domina tanto que mezclar no ayuda.

**Learning-to-rank (`rank:ndcg` con `SparkXGBRanker`):** 0.8517 vs 0.8609 del clasificador
multiclase → **−1,06%, no mejora**. El ranker paga dos peajes: solo entrena con grupos que
tienen algún positivo, y el muestreo de negativos le quita contexto.

**Router por segmento:** escrito (`src/router.py`) pero **sin ejecutar todavía**. Decide qué
modelo usar por tramo de historial/nº de productos, con la regla fijada en t-1 y aplicada a
ciegas en t (mismo patrón anti-fuga). Predicción hecha antes de ejecutarlo: probablemente no
aporte, porque el Transformer gana de forma bastante uniforme y no por arrasar en un segmento
concreto — pero hay que ejecutarlo para confirmarlo, no asumirlo.

### Caída de negocio (`churn.py`)

| Métrica | Valor |
|---|---|
| AUC | 0.866 |
| Precisión media (AP) | 0.262 (tasa base: 0.025) |
| Mejora sobre el azar | **10,4x** |
| Brier antes/después de calibrar | 0.02177 → 0.02130 (+2,2%) |

Umbral por valor de negocio (0.19) vs umbral por inercia (0.50):

| Umbral | Contactos | Bajas capturadas | Valor esperado |
|---|---|---|---|
| 0.19 (óptimo) | 33.566 | 32,6% | **197.966 €** |
| 0.50 (inercia) | 2.032 | — | 49.420 € |

**Pendiente:** re-ejecutar `churn.py` con la nueva variable `tenencia_producto` (meses que el
cliente lleva con CADA producto — la variable más predictiva de bajas, y todavía no se ha
medido su efecto real porque se añadió a `features.py`/`churn.py` pero no se ha vuelto a
correr el pipeline con ella).

### Fase de negocio (`business.py`)

**Desactualizada.** La ejecución que hay en `outputs/business_metrics.json` es de ANTES de
corregir el umbral de la matriz valor×riesgo (se partía por la mediana 0.5, dejando el 75% de
la cartera marcada como "riesgo alto" — inútil como plan). Se corrigió a percentil 0.80 en el
código pero no se ha vuelto a ejecutar sobre datos reales. El titular viejo (con XGBoost, antes
de saber que el Transformer es mejor) era: *"Contactando al 5% de la cartera se alcanza el
16,4% de las contrataciones, 3,22x lo que se conseguiría al azar."* Hay que rehacer esto con
el mejor modelo real (Transformer, una vez arreglada su fuga) y el umbral 0.80.

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

Hay **12 incidencias más** con detalle técnico y "cómo lo cuento si me preguntan" en
`DIARIO_PROBLEMAS.md` (no se sube a GitHub — está en `.gitignore`, es material de preparación
de entrevista, no de portfolio).

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
