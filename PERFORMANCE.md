# Rendimiento: cinco experimentos medidos sobre este dataset

Todos los números de este documento salen de ejecutar
[`src/perf.py`](src/perf.py) sobre las **13.647.309 filas reales** del proyecto. Nada está
copiado de un blog: cada cifra se puede reproducir con un comando.

**Configuración de la medición** — importa al leer los números:

| | |
|---|---|
| Modo | `local[*]` (una sola máquina, sin clúster) |
| Núcleos disponibles | 14 |
| Memoria del driver | 9 GB |
| Particiones de shuffle | 64 |
| Contenedor | Docker, 15,37 GB de RAM asignados |

Todos los tiempos se miden forzando el cálculo con un `count()`, porque Spark es perezoso:
construir un DataFrame no ejecuta nada, y cronometrar la construcción daría siempre cero.

---

## 1. Parquet contra CSV: 6,2x

Leer **dos columnas** de las 48 que tiene la tabla.

| Formato | Tiempo |
|---|---|
| CSV | 20,94 s |
| Parquet | **3,37 s** |

**Por qué.** Son dos ventajas distintas y conviene no mezclarlas:

**Almacenamiento por columnas.** El CSV guarda los datos fila a fila, así que para llegar a
la columna 40 hay que parsear las 39 anteriores de cada una de los 13,6 millones de filas.
Parquet guarda cada columna junta: si pides dos, lee dos y se salta el resto del fichero.

**Compresión.** El CSV escribe todo como texto — el número `87218.1` ocupa 7 caracteres. En
Parquet cada columna se comprime por separado, y como los datos de una misma columna se
parecen entre sí, comprimen mucho mejor. Las 24 columnas de producto son casi todo ceros.

El efecto en disco, medido en la ingesta:

| | Tamaño |
|---|---|
| `train_ver2.csv` | 2.186,5 MB |
| `data/parquet/` | **197,2 MB** |

**11,1 veces menos espacio.**

---

## 2. Poda de particiones: filtrar sin leer

La tabla está escrita en carpetas por `fecha_dato` (`partitionBy`). Eso cambia qué pasa al
filtrar:

| Filtro | Tiempo |
|---|---|
| Por `fecha_dato` (columna de particionado) | **1,28 s** |
| Por `renta` (columna normal) | 3,54 s |

**Por qué.** Filtrar por la columna de particionado se resuelve **descartando carpetas
enteras antes de abrir un solo fichero**. Spark sabe que la carpeta `fecha_dato=2015-03-28`
no puede contener filas de mayo de 2016, así que ni la mira. Filtrar por `renta` obliga a
abrir los 17 meses y mirar dentro.

Esto se llama *partition pruning* y queda escrito en el plan de ejecución, en la línea
`PartitionFilters` del `FileScan`. Es la forma de comprobar que de verdad está ocurriendo y
no solo suponerlo.

**La lección práctica:** elegir la columna de particionado es elegir qué filtros van a ser
gratis. Aquí es la fecha porque todo el proyecto trabaja mes a mes.

---

## 3. Broadcast join contra sort-merge join: 7,1x

Cruzar la tabla grande (13,6M filas) con una tabla pequeña (52 filas: la renta media por
provincia).

| Estrategia | Tiempo |
|---|---|
| Sort-merge join | 15,83 s |
| **Broadcast join** | **2,22 s** |

**Por qué.** Sin ayuda, Spark cruza dos tablas **barajando las dos** por la clave: manda
cada fila al ejecutor que le corresponde según su clave, para que las parejas acaben
juntas. Ese movimiento de datos entre ejecutores se llama *shuffle*, y es la operación más
cara que hay en Spark porque implica serializar, mover y reordenar.

Con 13,6 millones de filas, ese barajado es enorme. Y es absurdo, porque la otra tabla son
**52 filas**.

El broadcast join invierte el planteamiento: manda una **copia completa** de la tabla
pequeña a cada ejecutor y cruza en memoria local. La tabla grande no se mueve ni un byte.

**Cuándo NO usarlo:** cuando la tabla "pequeña" no lo es tanto. Si no cabe holgadamente en
la memoria de cada ejecutor, se replica algo enorme N veces y el remedio es peor que la
enfermedad. Spark lo hace solo por debajo de `spark.sql.autoBroadcastJoinThreshold` (10 MB
por defecto); forzarlo con `broadcast()` es decirle que confías en que cabe.

---

## 4. Cache: 2,3x cuando se reutiliza

Tres consultas sobre el mismo resultado intermedio.

| | Tiempo |
|---|---|
| Sin cache (recalcula 3 veces) | 2,3x más lento |
| **Con cache** | referencia |

**Por qué.** Spark no guarda nada entre acciones por defecto. Si sobre el mismo DataFrame
lanzas tres consultas, recalcula las tres desde el fichero original. `cache()` lo materializa
en memoria tras el primer cálculo.

**Pero no es gratis, y aquí está el matiz que importa.** La memoria que ocupa la cache es
memoria que le quitas al cálculo. En este proyecto eso pasó de verdad: cachear tres tablas
grandes a la vez dejó al modelo de bajas con el 86% del contenedor ocupado y la CPU
desplomada del 1.496% al 274% — la JVM se pasaba el tiempo recolectando basura en vez de
calcular.

**La regla:** cachear solo lo que se reutiliza de verdad, y proyectar las columnas
necesarias **antes** de cachear. En el pipeline solo está cacheado lo que se usa más de una
vez.

---

## 5. Sesgo de particiones: bien repartido

| | Filas |
|---|---|
| Partición mayor | 931.453 |
| Media | 802.783 |
| Partición menor | 625.457 |
| **Ratio máximo/media** | **1,16x** |

**Por qué importa.** Spark reparte el trabajo por particiones, y **un stage no termina hasta
que acaba su tarea más lenta**. Si una partición tiene diez veces más filas que las demás,
trece ejecutores se quedan mirando mientras uno sufre. Eso es el *skew*, y es la causa
número uno de que un job vaya lento sin motivo aparente.

Aquí el reparto es sano: 1,16x sobre la media. Tiene sentido, porque el particionado es por
mes y todos los meses tienen un número parecido de clientes — con un crecimiento gradual de
la cartera que explica la diferencia entre el primero y el último.

**Si estuviera sesgado**, las salidas serían: cambiar la clave de particionado, añadir un
sufijo aleatorio a las claves conflictivas (*salting*), o dejar que el ejecutor adaptativo
de Spark parta las particiones grandes — que es justo lo que hace
`spark.sql.adaptive.skewJoin.enabled`, activado en este proyecto.

---

## 6. Lo que no salió de un experimento, sino de un fallo

Estas dos no son experimentos: son cosas que se rompieron y se midieron al arreglarlas.

### El orden de las operaciones pesa más que el algoritmo

Para pasar de "un cliente con 24 columnas de producto" a "una fila por producto que tiene",
hay que multiplicar las filas por 24. La primera versión explotaba la tabla **entera** —unas
130 columnas— y proyectaba después las veinte que usa el modelo.

Explotar primero multiplica por 24 lo que no se va a usar. Proyectar primero y explotar
después hace el mismo cálculo cabiendo en memoria.

### Una acción de Spark dentro de un bucle

La búsqueda del umbral óptimo probaba 99 valores, y cada iteración era un `.agg().collect()`
—o sea, **99 jobs completos recorriendo la tabla entera**.

| | Tiempo |
|---|---|
| 99 acciones (una por umbral) | 112,8 s |
| **Una agregación + cálculo en el driver** | **3,6 s** |

**31 veces más rápido**, y eso sobre un dataset de juguete; sobre los 13,6M reales la
diferencia habría sido mucho mayor.

La versión buena agrupa una sola vez por la probabilidad redondeada a dos decimales —lo que
deja como mucho 101 filas— y calcula el acumulado en el driver sobre esas 101 filas. Es el
patrón de siempre: **agregar en el clúster, rematar en local**.

**La regla que queda:** dentro de un bucle no puede haber una acción de Spark.

---

## 7. One-hot con árboles: el error que más costó

`prev_canal_entrada` tiene **161 valores distintos**. Con one-hot eso son 161 columnas
nuevas, y el vector de características pasaba de ~45 a ~250 dimensiones.

| | Antes | Después |
|---|---|---|
| Dimensiones del vector | ~250 | **~45** |
| `maxBins` | 256 | **40** |
| Categorías de canal | 161 | 31 (top 30 + OTROS) |
| CPU aprovechada | 23% | 52% |

**Y el one-hot no solo era lento: estaba mal.** El one-hot encoding es una técnica para
modelos **lineales**, que no saben agrupar categorías y necesitan una columna por valor.

Los árboles sí saben. `StringIndexer` deja metadata que marca la columna como nominal,
`VectorAssembler` la propaga, y el árbol hace **particiones por subconjuntos**: puede separar
"canales A, D y F" contra el resto en un solo corte. El one-hot se lo impide — cada corte
solo puede aislar una categoría, así que necesita árboles más profundos para expresar lo
mismo.

Además, `maxBins` solo necesita superar la cardinalidad de la categórica mayor. Recortando
la cola larga a 30 categorías, baja de 256 a 40 y el coste de construir los histogramas de
cada nodo cae en proporción.

---

## Reproducir estos números

```bash
docker compose run --rm -e PYTHONPATH=/app/src spark python -u src/perf.py
```

Los resultados quedan en `outputs/perf_metrics.json`.
