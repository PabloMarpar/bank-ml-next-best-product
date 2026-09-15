"""
FASE 3b -- Modelos de secuencia: la trayectoria del cliente, no solo su foto actual.

    python src/secuencia.py

--------------------------------------------------------------------------------------
La critica al modelo que ya funciona
--------------------------------------------------------------------------------------
El XGBoost de la Fase 3 gana, pero mira muy poco. Sus variables son el estado del cliente
en el mes anterior (las 24 columnas prev_) y tres contadores de los ultimos meses. Con eso
descarta algo que los datos si tienen: el ORDEN.

Dos clientes que hoy tienen cuenta, nomina y tarjeta son identicos para XGBoost. Pero uno
puede haber llegado asi:

    cuenta -> nomina -> tarjeta      (domicilia la nomina y luego pide credito)

y el otro asi:

    tarjeta -> nomina -> cuenta      (entro por una tarjeta y se fue trayendo lo demas)

Son dos historias comerciales distintas y probablemente dos clientes distintos. La
trayectoria dice hacia donde va alguien; la foto solo dice donde esta.

--------------------------------------------------------------------------------------
Los modelos
--------------------------------------------------------------------------------------
Dos arquitecturas, las dos estandar en recomendacion secuencial:

  GRU          Una red recurrente que lee la secuencia mes a mes y arrastra un estado
               interno. Es la idea de GRU4Rec (2016), el primer trabajo que llevo las
               redes recurrentes a los recomendadores de sesion.

  TRANSFORMER  Auto-atencion causal: cada mes puede "mirar" a cualquier mes anterior
               directamente, en vez de a traves de un estado que se va comprimiendo. Es
               la idea de SASRec (2018). La ventaja teorica es que un evento de hace un
               ano puede influir sin haber tenido que sobrevivir a doce compresiones
               sucesivas.

Causal quiere decir que la mascara de atencion impide mirar hacia delante. Sin esa
mascara el modelo veria el futuro dentro de la propia secuencia -- la misma fuga de datos
de la Fase 2, pero por otra puerta.

--------------------------------------------------------------------------------------
Que hipotesis se prueba, y como se acepta perder
--------------------------------------------------------------------------------------
La pregunta NO es "que modelo puntua mas". Es:

    El orden de la trayectoria aporta algo sobre "que tienes ahora y que has movido
    hace poco"?

Puede que no. Con 24 productos y 16 meses las secuencias son cortas y pobres, y es
perfectamente posible que el estado actual resuma casi toda la informacion util. Si sale
asi se reporta asi, igual que se reporto que el ALS no aportaba.

--------------------------------------------------------------------------------------
Por que TorchDistributor y no un script suelto de PyTorch
--------------------------------------------------------------------------------------
TorchDistributor es la pieza que Spark trae desde la 3.4 para lanzar entrenamiento de
PyTorch usando el cluster como planificador. Aqui corre en local_mode con un proceso,
porque el dataset cabe de sobra en memoria -- pero el codigo es el mismo que se lanzaria
contra varios nodos cambiando num_processes.

El reparto de trabajo es el que tiene sentido en produccion: Spark prepara y agrega los
13,6M de filas, y PyTorch entrena sobre el resultado, que ya es pequeno.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from pyspark.sql import Window
from pyspark.sql import functions as F

from common import (
    DATA_FEATURES,
    MES_VALIDACION,
    NOMBRES_PRODUCTO,
    crear_sesion,
    cronometro,
    guardar_metricas,
)
from recommend import K, average_precision

PRODUCTOS = list(NOMBRES_PRODUCTO)
N_PRODUCTOS = len(PRODUCTOS)

# Longitud de la trayectoria que ve el modelo. Con 16 meses disponibles, 12 deja sitio
# para que casi todos los clientes tengan historia y no infla la memoria.
LARGO_SECUENCIA = 12

# Canales por mes: estado (que tenia), altas (que acababa de contratar) y bajas.
# La primera version solo usaba el estado y el modelo tenia que deducir los cambios.
CANALES = 3

# Variables del cliente que no son secuencia y entran aparte.
ESTATICAS = ["prev_age", "prev_antiguedad", "prev_renta", "n_prod_prev",
             "altas_3m", "bajas_3m", "meses_observados"]


# --------------------------------------------------------------------------------------
# Preparacion de los datos con Spark
# --------------------------------------------------------------------------------------
def construir_secuencias(spark, mes_objetivo: str, largo: int = LARGO_SECUENCIA):
    """Devuelve, por cliente y mes, la trayectoria de estados de producto anteriores.

    Se usa collect_list sobre una ventana que acaba en el mes ANTERIOR al que se predice
    (rowsBetween hasta -1). Ese -1 es el que garantiza que la secuencia no contiene el
    mes que hay que adivinar; es la misma separacion de la Fase 2, aplicada aqui otra vez
    porque una ventana mal acotada es la forma mas facil de colar el futuro.
    """
    feats = spark.read.parquet(str(DATA_FEATURES))

    # Cada mes se empaqueta en TRES enteros de 24 bits, un bit por producto:
    #
    #   estado  que productos tenia
    #   alta    cuales acababa de contratar ese mes
    #   baja    cuales acababa de cancelar
    #
    # La primera version solo pasaba el estado, y el modelo tenia que DEDUCIR los cambios
    # comparando meses consecutivos por su cuenta. Darle los eventos directamente le
    # ahorra ese trabajo: "en marzo tenias cuenta y nomina" no es lo mismo que "en marzo
    # acababas de domiciliar la nomina", y lo segundo es lo que predice.
    #
    # Empaquetar en bits no es un capricho: asi la trayectoria viaja como una lista de
    # enteros en vez de como una lista de listas, que en Spark es mucho mas cara de mover
    # y de serializar.
    def empaquetar(prefijo: str):
        return sum(
            F.coalesce(F.col(f"{prefijo}{p}"), F.lit(0)).cast("long") * (2 ** i)
            for i, p in enumerate(PRODUCTOS)
        )

    base = feats.select(
        "ncodpers", "fecha_dato", "mes_idx", "n_altas",
        *ESTATICAS,
        *[f"prev_{p}" for p in PRODUCTOS],
        *[f"alta_{p}" for p in PRODUCTOS],
        empaquetar("prev_").alias("estado_bits"),
        empaquetar("alta_").alias("alta_bits"),
        empaquetar("baja_").alias("baja_bits"),
    )

    ventana = (
        Window.partitionBy("ncodpers").orderBy("mes_idx")
        .rowsBetween(-largo, -1)          # -1: nunca el mes actual
    )
    return base.withColumns({
        "trayectoria": F.collect_list("estado_bits").over(ventana),
        "trayectoria_altas": F.collect_list("alta_bits").over(ventana),
        "trayectoria_bajas": F.collect_list("baja_bits").over(ventana),
    })


def a_numpy(filas, largo: int = LARGO_SECUENCIA, con_objetivo: bool = False):
    """Pasa las filas recogidas a las matrices que consume PyTorch.

    La secuencia se guarda como uint8 (0/1) y no como float32: son 4 veces menos memoria
    y la conversion a float se hace por lotes dentro del bucle de entrenamiento. Con
    560.000 ejemplos por 12 meses por 24 productos, la diferencia es entre 160 MB y
    640 MB.
    """
    n = len(filas)
    # Tres canales por mes: lo que tenia, lo que acababa de contratar y lo que acababa
    # de cancelar. De ahi el N_PRODUCTOS * CANALES.
    secuencias = np.zeros((n, largo, N_PRODUCTOS * CANALES), dtype=np.uint8)
    estaticas = np.zeros((n, len(ESTATICAS)), dtype=np.float32)
    poseidos = np.zeros((n, N_PRODUCTOS), dtype=np.uint8)
    objetivos = np.full(n, -1, dtype=np.int64)

    columnas_canal = ["trayectoria", "trayectoria_altas", "trayectoria_bajas"]

    for i, fila in enumerate(filas):
        for canal, columna in enumerate(columnas_canal):
            trayectoria = fila[columna] or []
            # Se alinea a la derecha: el mes mas reciente queda siempre en la ultima
            # posicion, para que el relleno de los clientes nuevos quede delante y no
            # desplace la parte que importa.
            recorte = trayectoria[-largo:]
            desplazamiento = largo - len(recorte)
            desfase = canal * N_PRODUCTOS
            for j, bits in enumerate(recorte):
                for b in range(N_PRODUCTOS):
                    if bits >> b & 1:
                        secuencias[i, desplazamiento + j, desfase + b] = 1

        for j, col in enumerate(ESTATICAS):
            valor = fila[col]
            estaticas[i, j] = 0.0 if valor is None else float(valor)

        for b, p in enumerate(PRODUCTOS):
            if fila[f"prev_{p}"] == 1:
                poseidos[i, b] = 1

        # Las filas de validacion no traen objetivo, asi que la columna se pide solo
        # cuando existe. Un Row de PySpark NO tiene .get(): es una subclase de tupla, y
        # acceder a un campo inexistente lanza AttributeError en vez de devolver None.
        if con_objetivo:
            objetivos[i] = PRODUCTOS.index(fila["producto_objetivo"])

    return secuencias, estaticas, poseidos, objetivos


def normalizar(estaticas: np.ndarray, referencia: np.ndarray | None = None):
    """Estandariza las variables estaticas. Las redes no toleran escalas dispares.

    La renta va en decenas de miles y la antiguedad en decenas: sin normalizar, el
    gradiente de la renta aplasta al resto. Los estadisticos se calculan SOLO sobre
    entrenamiento y se aplican a validacion, nunca al reves.
    """
    fuente = referencia if referencia is not None else estaticas
    media = fuente.mean(axis=0)
    desviacion = fuente.std(axis=0)
    desviacion[desviacion < 1e-6] = 1.0
    return (estaticas - media) / desviacion, media, desviacion


# --------------------------------------------------------------------------------------
# Los modelos
# --------------------------------------------------------------------------------------
def definir_modelos():
    """Se importa torch aqui dentro para que el modulo se pueda leer sin tenerlo."""
    import torch
    from torch import nn

    class CodificadorBase(nn.Module):
        """Parte comun: proyectar el estado de cada mes y fusionar con las estaticas."""

        def __init__(self, dim: int, n_estaticas: int):
            super().__init__()
            # Cada mes es un vector de 24 ceros y unos. Una capa lineal lo proyecta a un
            # espacio denso; es el equivalente a sumar las embeddings de los productos
            # que el cliente tiene ese mes.
            self.proyeccion = nn.Linear(N_PRODUCTOS * CANALES, dim)
            self.estaticas = nn.Sequential(
                nn.Linear(n_estaticas, dim), nn.ReLU(), nn.Dropout(0.1)
            )
            self.salida = nn.Sequential(
                nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(dim, N_PRODUCTOS),
            )

        def combinar(self, resumen, estaticas):
            return self.salida(torch.cat([resumen, self.estaticas(estaticas)], dim=-1))

    class ModeloGRU(CodificadorBase):
        """GRU4Rec: una recurrente que arrastra el estado a lo largo de la trayectoria."""

        def __init__(self, dim: int = 64, n_estaticas: int = len(ESTATICAS)):
            super().__init__(dim, n_estaticas)
            self.gru = nn.GRU(dim, dim, num_layers=1, batch_first=True)

        def forward(self, secuencia, estaticas):
            x = self.proyeccion(secuencia)
            salida, _ = self.gru(x)
            return self.combinar(salida[:, -1, :], estaticas)

    class ModeloTransformer(CodificadorBase):
        """SASRec: auto-atencion causal sobre la trayectoria."""

        def __init__(self, dim: int = 64, cabezas: int = 4, capas: int = 2,
                     n_estaticas: int = len(ESTATICAS)):
            super().__init__(dim, n_estaticas)
            # Sin codificacion posicional, la atencion es invariante al orden y el modelo
            # seria una bolsa de meses -- justo lo que se quiere superar.
            self.posicion = nn.Parameter(torch.randn(1, LARGO_SECUENCIA, dim) * 0.02)
            capa = nn.TransformerEncoderLayer(
                d_model=dim, nhead=cabezas, dim_feedforward=dim * 4,
                dropout=0.1, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(capa, num_layers=capas)

        def forward(self, secuencia, estaticas):
            x = self.proyeccion(secuencia) + self.posicion[:, -secuencia.size(1):, :]
            # Mascara triangular: la posicion i solo puede atender a <= i. Sin esto, el
            # modelo veria meses futuros dentro de su propia entrada.
            mascara = nn.Transformer.generate_square_subsequent_mask(
                secuencia.size(1), device=secuencia.device
            )
            salida = self.encoder(x, mask=mascara, is_causal=True)
            return self.combinar(salida[:, -1, :], estaticas)

    return ModeloGRU, ModeloTransformer


def entrenar_red(arquitectura, datos_train, datos_parada, reales_parada, epocas: int,
                 lote: int, lr: float, verbose: bool = True):
    """Bucle de entrenamiento. Devuelve (modelo, historial)."""
    import torch
    from torch import nn

    import copy

    torch.manual_seed(42)
    secuencias, estaticas, poseidos_train, objetivos = datos_train

    modelo = arquitectura()
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=epocas)

    # Mascara de perdida: sin ella, el modelo gasta capacidad aprendiendo a no
    # recomendar productos que el cliente ya tiene -- y luego esa recomendacion se
    # descarta igualmente al construir la lista final (ver ya_tiene() en recommend.py).
    # Es capacidad tirada dos veces. Penalizando con -inf los logits de lo ya poseido
    # ANTES de la softmax, el gradiente deja de gastarse ahi y toda la capacidad del
    # modelo va a discriminar entre los productos que de verdad podria recomendar.
    poseidos_t = torch.from_numpy(poseidos_train).bool()

    def logits_enmascarados(x_sec, x_est, poseidos_lote):
        logits = modelo(x_sec, x_est)
        return logits.masked_fill(poseidos_lote, float("-inf"))

    criterio = nn.CrossEntropyLoss()
    n = len(objetivos)

    # Early stopping con paciencia: si hay datos de validacion, se entrena hasta un
    # maximo de epocas pero se para en cuanto el MAP@7 de validacion deja de mejorar
    # durante `paciencia` epocas seguidas, y se devuelve el MEJOR estado visto, no el
    # ultimo. Sin esto, entrenar un numero de epocas fijo es apostar a que ese numero
    # es el correcto para cada arquitectura, cuando GRU y Transformer no convergen igual.
    mejor_map, mejor_estado, sin_mejora = -1.0, None, 0
    paciencia = 2

    historial = []
    for epoca in range(epocas):
        modelo.train()
        orden = np.random.RandomState(epoca).permutation(n)
        perdida_total = 0.0
        for inicio in range(0, n, lote):
            indices = orden[inicio:inicio + lote]
            x_sec = torch.from_numpy(secuencias[indices]).float()
            x_est = torch.from_numpy(estaticas[indices])
            y = torch.from_numpy(objetivos[indices])
            poseidos_lote = poseidos_t[indices]

            optimizador.zero_grad()
            perdida = criterio(logits_enmascarados(x_sec, x_est, poseidos_lote), y)
            perdida.backward()
            # Recorte de gradiente: las recurrentes son propensas a gradientes que
            # explotan, y con secuencias de 12 pasos ya se nota.
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optimizador.step()
            perdida_total += float(perdida) * len(indices)

        planificador.step()
        media = perdida_total / n
        registro = {"perdida": round(media, 5)}

        if datos_parada is not None:
            map_parada = _map_validacion(modelo, datos_parada, reales_parada)
            registro["map_parada"] = round(map_parada, 5)
            # Guarda: un MAP de parada exactamente 0 en la primera epoca significa que la
            # metrica esta rota, no que el modelo sea malo (un modelo al azar sacaria algo).
            # Si no se avisa, el early stopping corta en la epoca 1 y parece que entreno.
            if epoca == 0 and map_parada == 0.0:
                print("      AVISO: MAP de parada = 0 en la primera epoca. La metrica de "
                      "early stopping esta rota; el entrenamiento se cortaria solo.",
                      flush=True)
            if map_parada > mejor_map:
                mejor_map = map_parada
                mejor_estado = copy.deepcopy(modelo.state_dict())
                sin_mejora = 0
            else:
                sin_mejora += 1

        historial.append(registro)
        if verbose:
            extra = f"  MAP parada {registro.get('map_parada', '-')}" if datos_parada is not None else ""
            print(f"      epoca {epoca + 1}/{epocas}  perdida {media:.5f}{extra}",
                  flush=True)

        if datos_parada is not None and sin_mejora >= paciencia:
            if verbose:
                print(f"      early stopping: sin mejora en {paciencia} epocas "
                      f"(mejor MAP parada: {mejor_map:.5f})", flush=True)
            break

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)

    return modelo, historial


def _map_validacion(modelo, datos_parada, reales_parada) -> float:
    """MAP@7 sobre el mes de parada, con la MISMA definicion que la medicion final.

    La primera version comparaba contra `objetivos`, que en el tramo de parada son todos
    -1: ese tramo se construye con a_numpy(filas_parada) sin con_objetivo=True, porque un
    cliente puede haber contratado varios productos ese mes y no hay un unico objetivo.
    El resultado era que la metrica devolvia 0.0 en todas las epocas, el early stopping
    no veia mejora nunca y se quedaba con los pesos de la epoca 1. Un early stopping roto
    es peor que no tenerlo: no solo no elige bien, sino que corta el entrenamiento.

    Se mide contra `reales` (la lista de productos que el cliente contrato de verdad ese
    mes) reutilizando map_at_k, que es la funcion con la que se reporta el resultado
    final. Que la metrica de parada y la de medicion sean la misma importa: si difieren,
    se esta eligiendo la epoca por un criterio distinto del que luego se publica.
    """
    secuencias, estaticas, poseidos, objetivos = datos_parada
    probs = puntuar_enmascarado(modelo, (secuencias, estaticas, poseidos, objetivos))
    return map_at_k(probs, poseidos, reales_parada)


def puntuar_enmascarado(modelo, datos, lote: int = 4096):
    """Devuelve la matriz de probabilidades [n, 24], con los poseidos a -inf ANTES de
    la softmax -- la misma mascara que se usa durante el entrenamiento (ver
    entrenar_red). Se usa dentro del early stopping, donde interesa el ranking real.
    """
    import torch

    secuencias, estaticas, poseidos, _ = datos
    poseidos_t = torch.from_numpy(poseidos).bool()
    modelo.eval()
    salidas = []
    with torch.no_grad():
        for inicio in range(0, len(estaticas), lote):
            x_sec = torch.from_numpy(secuencias[inicio:inicio + lote]).float()
            x_est = torch.from_numpy(estaticas[inicio:inicio + lote])
            logits = modelo(x_sec, x_est).masked_fill(
                poseidos_t[inicio:inicio + lote], float("-inf")
            )
            salidas.append(torch.softmax(logits, dim=-1).numpy())
    return np.vstack(salidas)


def puntuar(modelo, datos, lote: int = 4096):
    """Devuelve la matriz de probabilidades [n, 24] del modelo, SIN enmascarar.

    Se usa fuera del entrenamiento (por ejemplo al exportar para el ensemble), donde
    interesa la probabilidad "cruda" y el enmascarado de poseidos lo hace map_at_k o el
    consumidor de turno.
    """
    import torch

    secuencias, estaticas, _, _ = datos
    modelo.eval()
    salidas = []
    with torch.no_grad():
        for inicio in range(0, len(estaticas), lote):
            x_sec = torch.from_numpy(secuencias[inicio:inicio + lote]).float()
            x_est = torch.from_numpy(estaticas[inicio:inicio + lote])
            salidas.append(torch.softmax(modelo(x_sec, x_est), dim=-1).numpy())
    return np.vstack(salidas)


def map_at_k(probabilidades, poseidos, reales: list[list[str]], k: int = K) -> float:
    """MAP@7 con la misma definicion que el resto del proyecto."""
    # Los productos que el cliente ya tiene se descartan poniendo su probabilidad a -1:
    # recomendar algo que ya se tiene no es un error del modelo, es un fallo de la capa
    # de negocio, y aqui se evita igual que en los otros enfoques.
    enmascarado = np.where(poseidos == 1, -1.0, probabilidades)
    orden = np.argsort(-enmascarado, axis=1)[:, :k]

    total = 0.0
    for i, indices in enumerate(orden):
        predichos = [PRODUCTOS[j] for j in indices]
        total += average_precision(predichos, reales[i], k)
    return total / len(reales)


# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Modelos de secuencia sobre trayectorias")
    parser.add_argument("--mes-validacion", default=MES_VALIDACION)
    parser.add_argument("--epocas", type=int, default=6)
    parser.add_argument("--lote", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--max-validacion", type=int, default=120_000,
                        help="Clientes de validacion a puntuar. Se evalua sobre los que "
                             "contrataron algo, que son ~28.000; el resto no entra en "
                             "la metrica.")
    parser.add_argument("--modelos", default="gru,transformer")
    parser.add_argument("--distribuido", action="store_true",
                        help="Lanza el entrenamiento con TorchDistributor de Spark")
    args = parser.parse_args()

    spark = crear_sesion("secuencia")
    spark.sparkContext.setLogLevel("ERROR")
    metricas: dict = {
        "mes_validacion": args.mes_validacion,
        "largo_secuencia": LARGO_SECUENCIA,
        "epocas": args.epocas,
        "dim": args.dim,
    }

    print("\n=== FASE 3b: modelos de secuencia ===\n")

    # --- Preparacion en Spark -----------------------------------------------------------
    # Tres tramos, no dos -- el mismo patron que churn.py (train/calibracion/test) y
    # ensemble.py (train/pesos/test). El mes de PARADA sirve solo para decidir cuando
    # dejar de entrenar; el mes de TEST no se toca hasta la medicion final. Usar el mes
    # de test para el early stopping seria elegir la epoca mirando la respuesta -- el
    # mismo tipo de fuga que el resto del proyecto evita en cada fase, aqui colada por
    # la puerta de un hiperparametro en vez de por una variable.
    with cronometro("preparacion_spark", metricas):
        datos = construir_secuencias(spark, args.mes_validacion)
        mes_test = datos.filter(F.col("fecha_dato") == args.mes_validacion).select(
            F.first("mes_idx").alias("idx")
        ).collect()[0]["idx"]
        mes_parada = mes_test - 1

        columnas_alta = F.array(*[
            F.when(F.col(f"alta_{p}") == 1, F.lit(p)).otherwise(F.lit(None))
            for p in PRODUCTOS
        ])

        # Entrenamiento: una fila por producto contratado, igual que en la Fase 3.
        train = (
            datos.filter((F.col("mes_idx") < mes_parada) & (F.col("n_altas") > 0))
            .withColumn("contratados", F.array_compact(columnas_alta))
            .withColumn("producto_objetivo", F.explode("contratados"))
            .drop("contratados")
        )
        # Parada: el mes justo anterior al de test. Solo decide cuando dejar de entrenar.
        parada = (
            datos.filter(F.col("mes_idx") == mes_parada)
            .withColumn("reales", F.array_compact(columnas_alta))
            .filter(F.size("reales") > 0)
        )
        # Test: la medicion final. No participa en ninguna decision de entrenamiento.
        test = (
            datos.filter(F.col("mes_idx") == mes_test)
            .withColumn("reales", F.array_compact(columnas_alta))
            .filter(F.size("reales") > 0)
        )

        filas_train = train.collect()
        filas_parada = parada.collect()
        filas_val = test.collect()

    metricas["ejemplos_entrenamiento"] = len(filas_train)
    metricas["clientes_parada"] = len(filas_parada)
    metricas["clientes_validacion"] = len(filas_val)
    print(f"  Ejemplos de entrenamiento: {len(filas_train):,}")
    print(f"  Clientes de parada (early stopping): {len(filas_parada):,}")
    print(f"  Clientes de test (medicion final):   {len(filas_val):,}")

    with cronometro("conversion_numpy", metricas):
        datos_train = a_numpy(filas_train, con_objetivo=True)
        datos_parada = a_numpy(filas_parada)
        datos_val = a_numpy(filas_val)
        reales = [list(f["reales"]) for f in filas_val]
        # Los productos realmente contratados en el mes de parada. El early stopping los
        # necesita para medir MAP@7 igual que se mide el resultado final (ver
        # _map_validacion): sin ellos la metrica de parada era siempre 0.
        reales_parada = [list(f["reales"]) for f in filas_parada]

        # Normalizacion con los estadisticos del entrenamiento, aplicados a los tres.
        est_train, media, desviacion = normalizar(datos_train[1])
        est_parada = (datos_parada[1] - media) / desviacion
        est_val = (datos_val[1] - media) / desviacion
        datos_train = (datos_train[0], est_train, datos_train[2], datos_train[3])
        datos_parada = (datos_parada[0], est_parada.astype(np.float32),
                        datos_parada[2], datos_parada[3])
        datos_val = (datos_val[0], est_val.astype(np.float32),
                     datos_val[2], datos_val[3])

    memoria_mb = datos_train[0].nbytes / 1024 ** 2
    metricas["memoria_secuencias_mb"] = round(memoria_mb, 1)
    print(f"  Secuencias en memoria: {memoria_mb:,.0f} MB "
          f"(uint8; en float32 serian {4 * memoria_mb:,.0f} MB)\n")

    # --- Entrenamiento ------------------------------------------------------------------
    ModeloGRU, ModeloTransformer = definir_modelos()
    arquitecturas = {
        "gru": lambda: ModeloGRU(dim=args.dim),
        "transformer": lambda: ModeloTransformer(dim=args.dim),
    }

    resultados = {}
    probabilidades_por_modelo = {}
    for nombre in [m.strip() for m in args.modelos.split(",")]:
        if nombre not in arquitecturas:
            continue
        print(f"  --- {nombre.upper()} ---")
        inicio = time.perf_counter()

        if args.distribuido:
            # TorchDistributor reparte el entrenamiento entre procesos usando Spark como
            # planificador. En local_mode con un proceso el resultado es identico a
            # entrenar aqui mismo; el valor es que el codigo ya esta en la forma que
            # aceptaria un cluster con solo cambiar num_processes.
            from pyspark.ml.torch.distributor import TorchDistributor

            distribuidor = TorchDistributor(
                num_processes=1, local_mode=True, use_gpu=False
            )
            modelo, historial = distribuidor.run(
                entrenar_red, arquitecturas[nombre], datos_train, datos_parada,
                reales_parada,
                args.epocas, args.lote, args.lr, False,
            )
        else:
            modelo, historial = entrenar_red(
                arquitecturas[nombre], datos_train, datos_parada, reales_parada,
                args.epocas, args.lote, args.lr,
            )

        segundos = time.perf_counter() - inicio
        probabilidades = puntuar(modelo, datos_val)
        puntuacion = map_at_k(probabilidades, datos_val[2], reales)

        probabilidades_por_modelo[nombre] = probabilidades
        resultados[nombre] = puntuacion
        metricas[f"{nombre}_historial_perdida"] = historial
        metricas[f"{nombre}_segundos"] = round(segundos, 1)
        print(f"    MAP@7 = {puntuacion:.5f}   ({segundos:.0f}s)\n")

    # --- Comparativa contra lo que ya habia ---------------------------------------------
    metricas["map7"] = {k: round(v, 5) for k, v in resultados.items()}

    referencia = None
    ruta_previa = DATA_FEATURES.parent.parent / "outputs" / "recommend_metrics.json"
    if ruta_previa.exists():
        import json
        referencia = json.loads(ruta_previa.read_text(encoding="utf-8")).get("map7", {})

    print("--- Comparativa ---")
    combinado = dict(resultados)
    if referencia:
        combinado.update({f"{k} (Fase 3)": v for k, v in referencia.items()})
    for nombre, valor in sorted(combinado.items(), key=lambda kv: -kv[1]):
        print(f"  {nombre:<24} {valor:.5f}")

    if referencia and "xgboost" in referencia and resultados:
        mejor_red = max(resultados, key=resultados.get)
        diferencia = resultados[mejor_red] - referencia["xgboost"]
        metricas["aporte_secuencia"] = round(diferencia, 5)
        metricas["aporte_secuencia_pct"] = round(
            100 * diferencia / referencia["xgboost"], 2
        )
        veredicto = (
            "El orden de la trayectoria SI aporta sobre el estado actual."
            if diferencia > 0 else
            "El orden de la trayectoria NO aporta sobre el estado actual: con 24 "
            "productos y 16 meses, la foto de lo que el cliente tiene ahora ya resume "
            "casi toda la informacion util de su historia."
        )
        metricas["veredicto_secuencia"] = veredicto
        print(f"\n  Mejor red ({mejor_red}) contra XGBoost: {diferencia:+.5f} MAP "
              f"({metricas['aporte_secuencia_pct']:+.1f}%)")
        print(f"  {veredicto}")

    # Las probabilidades se guardan para que src/ensemble.py pueda combinarlas sin
    # tener que reentrenar nada.
    if probabilidades_por_modelo:
        # El nombre lleva el mes: el ensemble necesita las predicciones de DOS meses
        # distintos -- uno para elegir los pesos de la mezcla y otro para medir -- y con
        # un nombre fijo el segundo pisaba al primero. Ese fallo hacia que la mezcla
        # nunca pudiera usar estas redes: solo existian para el mes de medicion, asi que
        # al elegir los pesos ni aparecian.
        destino = (
            DATA_FEATURES.parent / "export"
            / f"probs_secuencia_{args.mes_validacion}.npz"
        )
        destino.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destino,
            ncodpers=np.array([f["ncodpers"] for f in filas_val]),
            poseidos=datos_val[2],
            **probabilidades_por_modelo,
        )
        print(f"\n  Probabilidades guardadas en {destino}")

    guardar_metricas("secuencia_metrics.json", metricas)
    spark.stop()


if __name__ == "__main__":
    main()
