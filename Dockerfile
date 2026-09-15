# Imagen determinista para PySpark.
#
# Por que no usamos una imagen oficial de Spark: sus tags cambian de contenido con el
# tiempo y arrastran una version de Python que no controlamos. Aqui fijamos las tres
# piezas que importan -- Python 3.11, OpenJDK 17 y PySpark 3.5.3 -- para que el proyecto
# se ejecute igual en cualquier maquina dentro de un ano.
#
# Python 3.11 y no 3.13: PySpark 3.5 no soporta oficialmente 3.13.
# El tag -bookworm es necesario: el "slim" a secas ya apunta a Debian trixie, que retiro
# openjdk-17 del repositorio, y Spark 3.5 solo soporta oficialmente Java 8, 11 y 17.
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PYTHONUNBUFFERED=1
# Spark lanza los workers de Python con este interprete; sin esto puede coger otro.
ENV PYSPARK_PYTHON=python3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
