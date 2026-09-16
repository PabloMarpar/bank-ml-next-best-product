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

| Modelo | AUC |
|---|---|
| Limpio (solo mes t-1) | 0.9379 |
| Con la columna del mes actual metida a propósito | **1.0000** |

El modelo tramposo **cierra el 100% del error que le quedaba al limpio**. Es exactamente lo
que tenía que pasar: esa columna contiene la respuesta. Y que el limpio se quede por debajo
confirma que el pipeline real no la está usando.

El veredicto se da en términos relativos, no como diferencia de AUC. Un salto de 0.938 a
1.000 son "solo" 6 puntos, que suena a poco; dicho como *fracción del error restante que
elimina la fuga*, es el 100%, que es lo que de verdad significa.

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

### Recomendación — ocho enfoques comparados

| Enfoque | MAP@7 | vs popularidad | Escala Kaggle |
|---|---|---|---|
| **Transformer causal (SASRec)** | **0.903** | **1,40x** | **≈0.027** |
| GRU (GRU4Rec) | 0.902 | 1,40x | ≈0.027 |
| XGBoost multiclase | 0.861 | 1,34x | ≈0.026 |
| XGBoost + ALS (híbrido corregido) | 0.861 | 1,34x | ≈0.026 |
| XGBoost ranker (`rank:ndcg`) | 0.852 | 1,32x | ≈0.026 |
| Bosque aleatorio | 0.833 | 1,30x | ≈0.025 |
| Popularidad (baseline) | 0.643 | — | ≈0.019 |
| ALS (filtrado colaborativo) | 0.528 | 0,82x | ≈0.016 |

Para situarlo: el ganador de la competición sacó **0.031** y la mediana de los ~1.800
equipos rondó **0.025**.

La progresión está diseñada para aislar una variable cada vez:

- **Popularidad → ALS** mide qué aporta la señal colaborativa sola. Aquí: **nada**. El
  filtrado colaborativo queda muy por debajo del baseline.
- **ALS → Bosque** mide qué aportan las características del cliente. Aquí: mucho.
- **Bosque → XGBoost** mide qué aporta cambiar de algoritmo con la misma información. +3%.
- **XGBoost → Transformer** mide qué aporta **el orden de la trayectoria**. +4,9%, y es el
  salto más interesante del proyecto.

### Por qué el orden importa

XGBoost mira la foto: qué productos tiene el cliente ahora y cuántos ha movido hace poco.
Para él, dos clientes con cuenta, nómina y tarjeta son idénticos. Pero uno pudo llegar así:

```
cuenta -> nómina -> tarjeta      (domicilia la nómina y luego pide crédito)
```

y el otro así:

```
tarjeta -> nómina -> cuenta      (entró por una tarjeta y se fue trayendo lo demás)
```

Son dos historias comerciales distintas. La trayectoria dice hacia dónde va alguien; la foto
solo dice dónde está. Los modelos secuenciales leen esa trayectoria mes a mes: el GRU
arrastrando un estado, el Transformer con atención causal sobre los 16 meses.

Cada mes entra con **tres canales** — qué tenía, qué acababa de contratar y qué acababa de
cancelar. Con solo el estado, el modelo tendría que deducir los cambios comparando meses
consecutivos por su cuenta; darle los eventos directamente vale casi dos puntos de MAP.

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

| Cobertura del vector ALS | Antes | Después |
|---|---|---|
| En entrenamiento | ~100% | 99,2% |
| En validación | **21%** | 76,5% |
| MAP@7 del híbrido | 0.649 (−24,6%) | **0.8606** |

Corregido, el híbrido **empata** con XGBoost solo (0.8606 vs 0.8609). No lo supera: deja de
perjudicar. La señal colaborativa, en estos datos, no añade nada que las variables del
cliente no tuvieran ya — pero ahora se sabe que eso es un hallazgo y no un bug.

El código mide la cobertura en ambos lados y avisa cuando el desajuste pasa de 15 puntos.
Sigue habiendo 22,7 de desajuste, así que el aviso salta: es información, no un fallo
tapado.

### Tres cosas más que se probaron y tampoco funcionaron

**Learning to rank.** Optimizar el orden directamente con `rank:ndcg` en vez de clasificar y
ordenar por probabilidad: **0.852 vs 0.861, un 1,06% peor**. El ranker paga dos peajes: solo
entrena con grupos que tienen algún positivo, y el muestreo de negativos le quita contexto.

**Mezclar los modelos (ensemble).** Con los pesos elegidos honestamente en el mes anterior
al de medición: **0.90294 frente a 0.90299 del Transformer solo**. Los pesos salieron 50/50
entre GRU y Transformer, así que el problema no es la ponderación — es que los dos aciertan
en los mismos clientes.

**Un modelo distinto por segmento (router).** +0,06% enrutando por tramo de historial, y
+0,017% por número de productos. Además la segunda segmentación es inestable: uno de cada
cinco segmentos cambia de modelo ganador entre el mes de decisión y el de medición. No
compensa mantener cuatro modelos en producción para eso.

### Y el límite: exprimir el Transformer tampoco dio nada

Lo último que se intentó fue sacarle más al mejor modelo, y el resultado merece contarse
porque es el más honesto de todos.

Se implementó **supervisión por posición**: el modelo aplicaba atención causal pero solo
usaba la última posición de la secuencia, así que de doce meses de trayectoria salía una
sola etiqueta. La máscara causal permite predecir el mes j+1 desde cada posición j — que es
de donde SASRec saca su señal — y eso multiplica por doce las etiquetas con el mismo coste
de cálculo. Se añadieron también weight tying, calentamiento de learning rate, y una
búsqueda aleatoria de 12 configuraciones sobre 9 hiperparámetros.

| | MAP@7 |
|---|---|
| Antes de todo esto | 0.90299 |
| Después | **0.90315** |
| Mejora | **+0.00016 (+0,02%)** |

Y entonces la pregunta que decide si eso es una mejora: **¿cuánto se mueve el modelo solo
por cambiar la semilla aleatoria?** Tres semillas con la configuración ganadora dieron una
desviación de **0.00092**.

La mejora es **cinco veces menor que el ruido de inicialización**. No es una mejora.

Por si hiciera falta rematarlo: el modelo que finalmente se despliega, entrenado con la
configuración ganadora pero con otra semilla, saca **0.90179**. La misma configuración,
el mismo código, los mismos datos — 0.0014 de diferencia solo por el número con el que se
inicializan los pesos. Nueve veces la ganancia de toda la optimización.

Esto es lo que aporta medir la varianza entre semillas: sin ese número, +0.00016 se publica
como "la optimización mejoró el modelo" y es falso. Con él, la conclusión es que el modelo
ya estaba en su techo para esta familia de arquitecturas, y que el siguiente esfuerzo se
gasta mejor en otro sitio — más datos, otras variables, otro planteamiento.

### Caída de negocio

Segundo modelo, problema distinto: **¿qué cliente va a cancelar qué producto el mes que
viene?** Binario, y muy desbalanceado — solo el 2,53% de los pares cliente-producto acaban
en baja.

| Métrica | Valor |
|---|---|
| AUC | **0.8835** |
| Precisión media (AP) | 0.3526 |
| Tasa base | 0.0253 |
| **Mejora sobre el azar** | **14,0x** |
| Brier antes / después de calibrar | 0.02022 → 0.01977 |

Con un desbalance así, la exactitud no sirve de nada: un modelo que diga "nadie se va"
acierta el 97,5% de las veces. Por eso se reporta **precisión media contra la tasa base**,
que es la pregunta real — de los clientes que marco como riesgo, ¿cuántos se van de verdad,
comparado con marcar al azar?

**La variable que más subió el modelo: `tenencia_producto`**, los meses que el cliente lleva
con *cada* producto concreto. Antes solo existía `meses_observados`, que dice cuánto tiempo
llevamos viendo al cliente — pero no cuánto lleva con el producto que podría cancelar, que
es lo que de verdad importa. Añadirla movió el AUC de 0.8659 a 0.8835 y la mejora sobre el
azar de 10,4x a 14,0x.

Merece la pena comparar ese salto con el capítulo anterior: **una variable bien pensada dio
+0,018 de AUC, mientras que doce configuraciones de hiperparámetros sobre el recomendador
dieron +0,0002 de MAP.** No suele estar ahí el cuello de botella.

### El umbral no es 0,5

La decisión de a quién llamar no sale de la probabilidad, sale del dinero. Cada contacto
cuesta, cada baja evitada vale, y el punto óptimo no tiene por qué caer en 0,5 — de hecho
casi nunca cae ahí.

| Umbral | Contactos | Bajas capturadas | Valor esperado |
|---|---|---|---|
| **0,16 (óptimo)** | **45.727** | **14.463 (46,4%)** | **292.033 €** |
| 0,50 (por inercia) | 4.916 | — | 104.156 € |

Usar el 0,5 que viene por defecto deja **187.877 € sobre la mesa**. No porque el modelo sea
peor, sino porque 0,5 es un valor que nadie eligió: es el que sale de no decidir.

### Del modelo al plan comercial

Un modelo con buen MAP no es todavía un plan. La pregunta del equipo comercial es
**"tengo presupuesto para N llamadas este mes, ¿a quién llamo?"**, y eso se responde con una
curva de captura: se ordena la cartera entera por propensión y se mira qué fracción de las
contrataciones reales cae dentro de cada tramo.

| Llamas al… | Clientes | Capturas… | vs llamar al azar |
|---|---|---|---|
| 1% de la cartera | 9.399 | **17,0%** de las contrataciones | **15,8x** |
| 2% | 18.797 | 25,5% | 12,4x |
| **5%** | **46.993** | **38,3%** | **7,6x** |
| 10% | 93.986 | 48,9% | 4,9x |
| 20% | 185.826 | 57,7% | 2,9x |
| 50% | 463.715 | 81,8% | 1,6x |

**Contactando al 5% de la cartera se alcanza el 38,3% de las contrataciones del mes, 7,6
veces lo que se conseguiría llamando al azar.**

La curva se calcula sobre los **926.663 clientes** del mes, no sobre los 27.875 que
contrataron. Esa distinción es la que hace que el número signifique algo: si solo puntúas a
quien iba a comprar, "capturas el 38% llamando al 5%" no quiere decir nada.

Y luego está la decisión que no es del modelo sino del negocio: cruzar propensión de compra
con riesgo de fuga da cuatro cuadrantes, y a cada uno le corresponde una acción distinta.

| Cuadrante | Clientes | Qué se hace |
|---|---|---|
| NO TOCAR | 584.286 (63,1%) | no gastar contacto este mes |
| RETENER | 152.647 (16,5%) | campaña de retención |
| VENDER | 146.487 (15,8%) | ofrecerle el siguiente producto |
| RETENER PRIMERO | 43.243 (4,7%) | retener antes de intentar venderle nada |

El corte se hace por **percentil 80**, no por el 0,5 de probabilidad que sale por defecto.
Partiendo por la mediana, el 75% de la cartera quedaba marcada como riesgo alto — y una
lista con el 75% de los clientes no es un plan de acción, es la guía de teléfonos.

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
