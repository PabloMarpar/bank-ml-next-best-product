# Next Best Product — plan comercial sobre 13,6M de registros bancarios

Sistema que decide **a qué clientes debe llamar un equipo comercial el mes que viene, qué
producto ofrecerle a cada uno, y a cuáles está a punto de perder**. Construido en PySpark
sobre el dataset público de [Santander Product
Recommendation](https://www.kaggle.com/c/santander-product-recommendation): 13,6 millones
de filas, 946.089 clientes reales, 24 productos financieros y 17 meses de fotos mensuales.

> Si prefieres la versión sin jerga, está en **[RESUMEN.md](RESUMEN.md)** — una página,
> pensada para leerse sin saber programar.

---

## El problema

Un banco con 1,5 millones de clientes tiene un equipo comercial que puede contactar, como
mucho, a unos miles al mes. La pregunta no es "¿quién podría querer un plan de pensiones?"
sino **"¿a quién llamo primero, con qué presupuesto de llamadas?"**.

Eso se traduce en tres modelos y una capa que los junta:

| Pregunta | Cómo se responde |
|---|---|
| ¿Qué va a contratar cada cliente? | Clasificador multiclase sobre 24 productos |
| ¿A quién estoy a punto de perder? | Clasificador binario de baja de producto |
| ¿A quién llamo, entonces? | Curva de captura + matriz valor × riesgo |

---

## La trampa del dataset, y cómo se evita

Cada fila trae, para un cliente y un mes, **los 24 productos que tiene ese mes**. Es
tentador usar esas columnas para predecir qué contrata ese mismo mes, y el modelo saldría
casi perfecto — porque la columna de la tarjeta de junio ya contiene si contrató la tarjeta
en junio. Eso no es predecir: es leer la respuesta.

Todo el pipeline está construido sobre una separación estricta:

```
variables  ->  describen el mes t-1    (lo que se sabe al decidir a quién llamar)
objetivo   ->  describe el mes t        (lo que pasó después)
```

**Y se comprueba, no se afirma.** [`src/leakage_check.py`](src/leakage_check.py) entrena el
mismo modelo dos veces: una limpia y otra metiendo a propósito la columna del mes actual.
Si la separación está bien hecha, el modelo tramposo tiene que dispararse:

<!-- NUMEROS_FUGA -->

Dos detalles de corrección que no son obvios y que el código sí contempla:

1. **`lag()` no devuelve el mes anterior, devuelve la fila anterior del cliente.** Hay
   clientes que desaparecen unos meses y vuelven; sin comprobarlo, se compararía mayo con
   enero como si fueran consecutivos. Se guarda también el lag de la fecha y solo se
   conservan los saltos de exactamente un mes.
2. **El primer mes de cada cliente se descarta**, porque no tiene con qué compararse: todos
   sus productos parecerían recién contratados. Son el 7,07% de las filas, y el descarte se
   cuenta y se reporta en vez de ocurrir en silencio.

---

## Resultados

### Cómo hay que leer el MAP@7 (importante)

La métrica es MAP@7, la misma de la competición original. **Aquí se calcula solo sobre los
27.875 clientes que contrataron algo**, porque a quien no contrató nada no se le puede medir
el acierto. Kaggle la calcula sobre los 926.663 clientes del mes, contando como cero a los
que no contrataron.

Son dos denominadores distintos y **no son comparables directamente**. La conversión es:

```
MAP@7 (escala Kaggle) = MAP@7 (este repo) x 3,01%
```

### Recomendación — cinco enfoques comparados

| Enfoque | MAP@7 | vs popularidad | Escala Kaggle |
|---|---|---|---|
| **XGBoost** | **0.861** | **1,34x** | **≈0.026** |
| Bosque aleatorio | 0.809 | 1,26x | ≈0.024 |
| XGBoost + ALS (híbrido) | 0.649 | 1,01x | ≈0.020 |
| Popularidad (baseline) | 0.643 | — | ≈0.019 |
| ALS (filtrado colaborativo) | 0.638 | 0,99x | ≈0.019 |

Para situarlo: el ganador de la competición sacó **0.031** y la mediana de los ~1.800
equipos rondó **0.025**.

La progresión está diseñada para aislar una variable cada vez:

- **Popularidad → ALS** mide qué aporta la señal colaborativa sola. Aquí: **nada**. El
  filtrado colaborativo queda por debajo del baseline.
- **ALS → Bosque** mide qué aportan las características del cliente. Aquí: mucho, +26%.
- **Bosque → XGBoost** mide qué aporta cambiar de algoritmo con la misma información. +6%.

### Lo que no funcionó, y por qué (esto es lo interesante)

**El recomendador híbrido perdió un 24,6%.** La idea era buena — inyectar los vectores
latentes del ALS como variables dentro de XGBoost, que es la arquitectura estándar de los
recomendadores grandes. Falló por una razón que no se ve en ninguna métrica de entrenamiento:

| | Cobertura del vector ALS |
|---|---|
| En entrenamiento | ~100% |
| En validación | **21%** |

El ALS se entrenó sobre el historial de **altas**, así que solo existía vector para los
195.548 clientes que habían contratado algo. Y el ranker se entrena **solo con filas de
clientes que contrataron algo** — es cómo se construye el problema multiclase. Resultado: el
modelo nunca vio durante el entrenamiento el caso *"este cliente no tiene vector"*, y al
predecir se lo encontró en 4 de cada 5 clientes.

Esto es **desajuste entre entrenamiento y servicio** (*train/serve skew*), y es de los fallos
más difíciles de detectar porque no produce ningún error: el modelo responde con total
normalidad, solo que peor.

La corrección: entrenar el ALS sobre los productos que cada cliente **posee**, no solo sobre
los que ha contratado nuevos. Así cualquiera con al menos un producto recibe vector y la
cobertura se iguala. El código ahora mide la cobertura en ambos lados y avisa si divergen.

<!-- RESULTADO_HIBRIDO_CORREGIDO -->

### Caída de negocio

<!-- NUMEROS_CHURN -->

### Del modelo al plan comercial

<!-- NUMEROS_NEGOCIO -->

---

## Arquitectura

```
data/raw/train_ver2.csv          2,3 GB  ·  13.647.309 filas
        |
        |  src/ingest.py         esquema explícito, casteo controlado
        v
data/parquet/                    197 MB  ·  17 particiones mensuales
        |
        |  src/features.py       window functions, lag de 24 productos
        v
data/features/                   12.682.421 filas utilizables
        |
        +--> src/recommend.py    popularidad · ALS · bosque · XGBoost · híbrido
        +--> src/secuencia.py    GRU · Transformer causal (PyTorch vía TorchDistributor)
        +--> src/ensemble.py     fusión por posición, pesos elegidos en t-1
        +--> src/churn.py        GBT + calibración isotónica
                    |
                    v
             src/business.py     curva de captura · matriz valor × riesgo
```

Análisis complementarios:

| Módulo | Qué responde |
|---|---|
| [`src/leakage_check.py`](src/leakage_check.py) | ¿Está el pipeline libre de fuga? (se comprueba haciendo trampa a propósito) |
| [`src/analisis_renta.py`](src/analisis_renta.py) | Al 20,5% le falta la renta: ¿falta al azar? ¿mediana o modelo? |
| [`src/deriva.py`](src/deriva.py) | ¿Cada cuánto caduca el modelo y hay que reentrenar? |
| [`src/cold_start.py`](src/cold_start.py) | ¿Qué le ofrezco al cliente que acaba de entrar? |
| [`src/perf.py`](src/perf.py) | Experimentos de rendimiento de Spark, medidos ([PERFORMANCE.md](PERFORMANCE.md)) |

---

## Decisiones técnicas que se pueden defender

**Esquema explícito, sin `inferSchema`.** Con `inferSchema`, Spark lee el fichero entero una
vez solo para adivinar los tipos y luego lo relee para cargarlo: 2,3 GB leídos dos veces.

**Todas las columnas se leen como texto y se castean a mano.** El CSV trae `NA` textual,
espacios de relleno en `age` y `antiguedad`, y `-999999` como centinela de cliente nuevo. Si
se declara el tipo final en el esquema, Spark convierte en nulo lo que no encaja **sin
avisar**, y se pierde la oportunidad de medir cuánto se pierde.

**Imputación por mediana más indicador de ausencia.** Los árboles de Spark rechazan `NaN`,
así que imputar es obligatorio. Pero al 20,5% de los clientes les falta la renta, y eso no es
ruido: suele ser cliente antiguo con la ficha incompleta. Imputando a secas esa señal
desaparece; guardando además una columna 0/1 de "faltaba", el modelo puede usar las dos cosas.

**Bosque aleatorio y XGBoost para recomendar, GBT para las bajas.** No es capricho: el
`GBTClassifier` de Spark **solo resuelve problemas binarios**. Elegir entre 24 productos es
multiclase, así que ahí queda descartado; la alternativa dentro de Spark ML era envolver 24
clasificadores con `OneVsRest`. XGBoost soporta multiclase de forma nativa. Para las bajas,
que sí es binario, se usa el GBT nativo.

**Calibración isotónica en el modelo de bajas.** El umbral de la campaña se elige maximizando
valor esperado, y ese cálculo multiplica por una probabilidad. Los ensembles de árboles
ordenan bien pero comprimen las magnitudes hacia el centro. Se calibra sobre un **tercer
tramo temporal** que el modelo no vio al entrenar.

Matiz que el código reporta en vez de esconder: la isotónica es monótona **no decreciente**,
no estrictamente creciente. Es escalonada, empata a los clientes que caen en el mismo escalón,
y eso mueve el AUC ligeramente. Preservarlo exacto requeriría escalado de Platt.

**La curva de captura se calcula con `approxQuantile`, no con una ventana ordenada.** Una
ventana sin `partitionBy` obliga a juntar toda la tabla en un solo ejecutor. La versión con
percentiles más una agregación da el mismo resultado de forma distribuida.

**Split temporal en todas partes, nunca aleatorio.** Un split aleatorio repartiría al mismo
cliente entre entrenamiento y validación y mezclaría meses: el modelo vería junio al entrenar
y luego se le pediría predecir marzo. En el ensemble el reparto es de tres tramos — modelos
base en ≤ t-2, pesos de la mezcla en t-1, medición en t — para que el mes de test no participe
en ninguna decisión.

---

## Cómo ejecutarlo

Requiere **Docker**. No hace falta tener Java ni una versión concreta de Python: el contenedor
fija OpenJDK 17, Python 3.11 y PySpark 3.5.3.

```bash
docker compose build

# Credenciales de Kaggle: el token va en ~/.kaggle/access_token
docker compose run --rm spark python scripts/download_data.py

R="docker compose run --rm -e PYTHONPATH=/app/src spark python -u"
$R src/ingest.py            # CSV  -> Parquet particionado
$R src/features.py          # lags, altas y bajas
$R src/leakage_check.py     # control de fuga de datos
$R src/recommend.py --exportar
$R src/churn.py --exportar
$R src/business.py          # curva de captura y matriz valor x riesgo
```

**Prueba rápida sin descargar nada** (~2 minutos): genera un dataset sintético con la misma
suciedad que el real y ejecuta el pipeline entero.

```bash
docker compose run --rm -e PYTHONPATH=/app/src spark python scripts/smoke_test.py
```

---

## Lo que este proyecto no hace

- **No se ha ejecutado en AWS.** Hay un empaquetado para SageMaker en
  [`sagemaker/`](sagemaker/README.md) que respeta su contrato de rutas y podría lanzarse sin
  reescribir nada, pero no hay ningún job real que enseñar. Se dice aquí y allí.
- **No promete euros.** La cifra de valor de la campaña depende de supuestos de coste que
  están escritos explícitamente en el código para poder discutirlos.
- **No decide nada solo.** Ordena una lista; la llamada la hace una persona.
- **La demo no ejecuta Spark.** Streamlit Cloud no puede. Lee predicciones ya calculadas por
  el pipeline, que es además el reparto correcto en producción: el cálculo pesado va en lote
  una vez al mes, la consulta solo lee.

---

## Datos

[Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation),
competición pública de Kaggle (2016). Datos reales anonimizados de clientes de banca, con
fecha, producto contratado y atributos de perfil. No se incluyen en el repositorio: se
descargan con `scripts/download_data.py`.
