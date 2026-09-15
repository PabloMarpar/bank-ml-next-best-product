# Empaquetado para Amazon SageMaker

> **Esto no se ha ejecutado en AWS.** No hay cuenta, no hay factura y no hay job que enseñar.
> Lo que hay es el pipeline empaquetado siguiendo el contrato que espera SageMaker, para que
> pudiera lanzarse en la nube sin reescribir nada. Se dice aquí arriba y no en una nota al pie
> a propósito: un proyecto que insinúa experiencia que no tiene es peor que uno que no la
> menciona.

## El contrato

SageMaker no importa tu código como una librería. Arranca un contenedor, deja los datos en
unas rutas concretas y ejecuta tu script como un programa normal. El acuerdo tiene cuatro
puntos:

| Variable de entorno | Qué contiene |
|---|---|
| `SM_CHANNEL_TRAIN` | Dónde aparecen los datos de entrada que has apuntado en S3 |
| `SM_MODEL_DIR` | Dónde dejar el modelo. SageMaker lo sube a S3 solo al terminar |
| `SM_OUTPUT_DATA_DIR` | Dónde dejar métricas, gráficos y demás artefactos |
| *(argumentos CLI)* | Los hiperparámetros llegan como `--nombre valor` |

`train.py` respeta los cuatro, con valores por defecto locales. Esa es la gracia: **el mismo
fichero corre en el portátil y correría en SageMaker**, sin ramas ni condicionales.

## Las dos formas de lanzarlo, y cuál corresponde aquí

SageMaker ofrece dos caminos para un trabajo con Spark, y elegir el bueno es la mitad del
conocimiento útil:

**`PySparkProcessor`** — levanta un clúster de Spark gestionado y ejecuta un script encima.
Es lo que corresponde a este proyecto: la carga está en procesar 13,6M de filas y entrenar con
Spark ML, distribuido.

**`Estimator` con imagen propia** — una sola máquina ejecutando tu contenedor. Es lo natural
para entrenar con scikit-learn o PyTorch, donde no hay nada que distribuir. Meter Spark aquí
sería levantar un clúster de un nodo: toda la complejidad y ninguna de las ventajas.

```python
from sagemaker.spark.processing import PySparkProcessor

procesador = PySparkProcessor(
    base_job_name="next-best-product",
    framework_version="3.5",
    role=ROL_DE_EJECUCION,
    instance_type="ml.m5.2xlarge",
    instance_count=4,          # aquí sí es un clúster de verdad, no local[*]
    max_runtime_in_seconds=3600,
)

procesador.run(
    submit_app="sagemaker/train.py",
    submit_py_files=["src/common.py", "src/churn.py"],   # el pipeline viaja con el script
    arguments=[
        "--max-iter", "60",
        "--max-depth", "6",
        "--train-months", "6",
    ],
    inputs=[ProcessingInput(
        source="s3://<bucket>/next-best-product/features/",
        destination="/opt/ml/processing/input",
    )],
    outputs=[ProcessingOutput(
        source="/opt/ml/processing/output",
        destination="s3://<bucket>/next-best-product/model/",
    )],
)
```

## Qué habría que cambiar de verdad para ejecutarlo en la nube

Tres cosas, y ninguna es el código de modelado:

1. **Las rutas.** `common.py` usa rutas de fichero local. En AWS serían `s3a://bucket/...`.
   Spark lee S3 de forma nativa con el conector `hadoop-aws`, así que es cambiar la cadena de
   la ruta y añadir el JAR, no reescribir la lectura.

2. **La sesión de Spark.** `crear_sesion()` fija `master("local[*]")`, que significa "una sola
   máquina, usa todos sus núcleos". En un clúster gestionado esa línea sobra: SageMaker inyecta
   la configuración del clúster.

3. **El particionado.** En local, 64 particiones de shuffle van bien para 14 núcleos. Con
   cuatro instancias habría que subirlo — la regla habitual es dos o tres particiones por
   núcleo total disponible.

## El coste, que es la pregunta que nadie hace hasta que llega la factura

Cuatro instancias `ml.m5.2xlarge` durante una hora salen por unos pocos euros. Lo caro no es
entrenar: es dejarse un endpoint de inferencia levantado, porque se paga por hora esté o no
atendiendo peticiones. Para un caso como este —una tanda de predicciones al mes— lo correcto
es **transform por lotes**, que arranca, procesa y se apaga. Un endpoint permanente para un
proceso mensual es tirar el dinero.
