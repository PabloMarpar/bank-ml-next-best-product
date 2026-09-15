# Contexto del proyecto (documento vivo)

> Este fichero se actualiza al terminar cada fase. Sirve para retomar el trabajo en una sesión
> nueva sin releer el histórico de conversación. El `README.md` es la narrativa pulida para
> quien visita el repo; `RESUMEN.md` es la versión sin jerga; este documento es el cuaderno de
> bitácora: qué se ha hecho, qué falta, y por qué se decidió cada cosa que no sea obvia
> leyendo el código.

## Qué es esto, en una frase

Sistema de *next best product* sobre datos reales de banca: decide a qué clientes debe llamar
un equipo comercial el mes que viene, qué producto ofrecerle a cada uno, y a cuáles está a
punto de perder. Construido en PySpark sobre 13,6M de filas.

## Estado actual (última actualización: 2026-09-15)

**Fases 0 a 6: código escrito y validado de punta a punta sobre un dataset sintético.**
**Pendiente: ejecutar sobre el dataset real** (bloqueado por las credenciales de Kaggle).

| Fase | Estado | Fichero |
|---|---|---|
| 0 — Entorno Docker | ✅ construido y probado | `Dockerfile`, `docker-compose.yml` |
| 0 — Descarga de datos | ⏳ **falta `~/.kaggle/kaggle.json`** | `scripts/download_data.py` |
| 1 — Ingesta CSV→Parquet | ✅ código listo, probado en sintético | `src/ingest.py` |
| 2 — Features y objetivos | ✅ código listo, probado en sintético | `src/features.py` |
| 3 — Recomendación (3 enfoques) | ✅ código listo, probado en sintético | `src/recommend.py` |
| 4 — Caída de negocio | ✅ código listo, probado en sintético | `src/churn.py` |
| 5 — Capa de negocio | ✅ código listo, probado en sintético | `src/business.py` |
| 6 — Experimentos de rendimiento | ✅ código listo, **sin ejecutar** | `src/perf.py` |
| — Control de fuga de datos | ✅ código listo y **funcionando** | `src/leakage_check.py` |
| 7 — Empaquetado SageMaker | ⏳ pendiente | `sagemaker/` |
| 8 — Demo y README | ⏳ pendiente | `app/`, `README.md` |

## Entorno y cómo se ejecuta

La máquina de desarrollo **no tiene Java**, así que todo corre dentro de Docker. No intentar
ejecutarlo con el Python del sistema: es 3.13 y PySpark 3.5 no lo soporta.

```bash
docker compose build
docker compose run --rm spark python scripts/download_data.py
docker compose run --rm -e PYTHONPATH=/app/src spark python src/ingest.py
docker compose run --rm -e PYTHONPATH=/app/src spark python src/features.py
docker compose run --rm -e PYTHONPATH=/app/src spark python src/recommend.py --exportar
docker compose run --rm -e PYTHONPATH=/app/src spark python src/churn.py --exportar
docker compose run --rm -e PYTHONPATH=/app/src spark python src/business.py
```

**Prueba rápida sin datos reales** (40 s, genera un dataset sintético y corre las 4 fases):

```bash
docker compose run --rm -e PYTHONPATH=/app/src spark python scripts/smoke_test.py
```

Configuración: `local[*]` con 12 GB de driver y 64 particiones de shuffle. La máquina tiene
32 GB de RAM y 14 núcleos lógicos; los tiempos de `PERFORMANCE.md` se medirán con esa
configuración y hay que decirlo al reportarlos.

## Resultados medidos hasta ahora

### Sobre el dataset sintético (solo valida que el código funciona)

| Enfoque | MAP@7 |
|---|---|
| Popularidad | 0.365 |
| ALS | 0.275 |
| **Bosque aleatorio** | **0.438** |

El bosque gana, que es lo esperado porque la señal sintética se generó dependiendo del
segmento y la edad. **Estos números no significan nada sobre el problema real**, solo que el
pipeline está bien conectado.

### Control de fuga de datos — ✅ funcionando

| Modelo | AUC |
|---|---|
| Limpio (solo mes anterior) | 0.4214 |
| Con fuga deliberada (mes actual incluido) | **1.0000** |

Salto de +0,58. Es exactamente el contraste que se buscaba: demuestra que la fuga existe y
sería trivial caer en ella, y que el pipeline real no la tiene.

### Sobre el dataset real

⏳ Pendiente.

## Decisiones y hallazgos que no son obvios leyendo el código

1. **Todas las columnas se leen como texto y se castean a mano** (`ingest.py`). El CSV trae
   `NA` textual, espacios de relleno y el centinela `-999999`. Si se declarara el tipo final
   en el esquema, Spark convertiría en null lo que no encaje sin avisar, y no se podría medir
   cuánto se pierde.

2. **`lag()` no garantiza el mes anterior** (`features.py`). Devuelve la fila anterior del
   cliente, y hay clientes con huecos en su historial. Se guarda también el lag de la fecha y
   solo se conservan los saltos de exactamente un mes. Sin esto no habría error: habría
   resultados mal calculados en silencio.

3. **Bosque aleatorio para recomendar, gradient boosting para bajas.** No es capricho: el
   `GBTClassifier` de Spark **solo resuelve problemas binarios**. Recomendar entre 24
   productos es multiclase, así que ahí hay que usar bosque. Predecir una baja sí es binario.

4. **Imputación por mediana + indicador de faltante** (`common.py`). Los árboles de Spark
   rechazan `NaN`, así que imputar es obligatorio. Pero que falte la renta no es ruido —suele
   ser cliente antiguo con ficha incompleta— y esa señal se perdería imputando a secas. Por eso
   se guarda además una columna 0/1 por variable diciendo si el valor faltaba.

5. **La curva de captura se calcula con `approxQuantile`, no con una ventana ordenada**
   (`business.py`). Una ventana sin `partitionBy` obliga a juntar toda la tabla en un solo
   ejecutor. La versión con percentiles + una agregación da el mismo resultado distribuido.

6. **El umbral de la campaña de retención se elige por valor de negocio, no en 0,5.** Los
   supuestos de coste están escritos explícitamente en `churn.py` para poder discutirlos. Si el
   óptimo resulta ser "no llamar a nadie", se reporta tal cual.

7. **`NBP_DATA_DIR` / `NBP_OUTPUTS_DIR`** redirigen todo el árbol de datos. Es lo que permite
   que `smoke_test.py` ejecute el pipeline entero sin pisar los datos ni las métricas buenas.

## Próximos pasos inmediatos (en orden)

1. **Crear `%USERPROFILE%\.kaggle\kaggle.json`** y aceptar las reglas de la competición en
   kaggle.com/c/santander-product-recommendation. Sin esto no hay datos.
2. `scripts/download_data.py` → `src/ingest.py` → `src/features.py`.
3. `src/leakage_check.py` sobre datos reales: es el primer resultado que merece la pena tener.
4. `src/recommend.py --exportar` y `src/churn.py --exportar`. Comprobar si la hipótesis se
   cumple (el bosque debería ganar al ALS porque con solo 24 productos la señal está en el
   cliente, no en la co-ocurrencia). **Si sale al revés, se reporta igual.**
5. `src/business.py` → la cifra de titular.
6. `src/perf.py` → rellenar `PERFORMANCE.md` con los números medidos.
7. `sagemaker/`, demo Streamlit, `README.md` y card en el portfolio.
