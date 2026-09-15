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
    DATA_EXPORT,
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

# Tope de longitud. Los tensores se preparan SIEMPRE a este largo y despues se recortan
# por la izquierda con recortar_largo(). Asi la busqueda de hiperparametros puede probar
# varias longitudes sin volver a pasar por Spark, que es la parte cara: preparar cuesta
# cinco minutos y entrenar, cuatro.
LARGO_MAXIMO = 16

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


def a_numpy(filas, largo: int = LARGO_MAXIMO, con_objetivo: bool = False):
    """Pasa las filas recogidas a las matrices que consume PyTorch.

    La secuencia se guarda como uint8 (0/1) y no como float32: son 4 veces menos memoria
    y la conversion a float se hace por lotes dentro del bucle de entrenamiento. Con
    560.000 ejemplos por 12 meses por 24 productos, la diferencia es entre 160 MB y
    640 MB.

    Devuelve tambien `longitudes`: cuantos meses REALES tiene cada fila, frente al
    relleno de ceros de la izquierda. Sin ese dato no se puede supervisar posicion a
    posicion, porque un mes de relleno y un mes real en el que el cliente no contrato
    nada son el mismo vector de ceros -- y entrenar contra el relleno enseñaria al modelo
    a no predecir nada.
    """
    n = len(filas)
    # Tres canales por mes: lo que tenia, lo que acababa de contratar y lo que acababa
    # de cancelar. De ahi el N_PRODUCTOS * CANALES.
    secuencias = np.zeros((n, largo, N_PRODUCTOS * CANALES), dtype=np.uint8)
    estaticas = np.zeros((n, len(ESTATICAS)), dtype=np.float32)
    poseidos = np.zeros((n, N_PRODUCTOS), dtype=np.uint8)
    objetivos = np.full(n, -1, dtype=np.int64)
    longitudes = np.zeros(n, dtype=np.int64)

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
            # Los tres canales salen de la misma ventana, asi que tienen la misma
            # longitud; se apunta una vez.
            if canal == 0:
                longitudes[i] = len(recorte)

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

    return secuencias, estaticas, poseidos, objetivos, longitudes


def recortar_largo(datos, largo: int):
    """Recorta la trayectoria a los `largo` meses mas recientes.

    Los tensores se preparan a LARGO_MAXIMO una sola vez y cada configuracion de la
    busqueda se queda con la cola que necesita. Se recorta por la IZQUIERDA porque las
    secuencias estan alineadas a la derecha: el mes mas reciente siempre es la ultima
    posicion, asi que quedarse con [-largo:] es quedarse con lo mas cercano al objetivo.
    """
    if largo >= datos[0].shape[1]:
        return datos
    return (datos[0][:, -largo:, :], datos[1], datos[2], datos[3],
            np.minimum(datos[4], largo))


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
    from torch.nn import functional as Fn

    class CodificadorBase(nn.Module):
        """Parte comun: proyectar el estado de cada mes y fusionar con las estaticas.

        Dos cabezas, y la razon es de fuga, no de arquitectura:

          `salida`      la cabeza principal. Ve el resumen de la secuencia Y las
                        variables estaticas, y predice el mes objetivo. Es la que se usa
                        para puntuar.

          `cabeza_aux`  la cabeza auxiliar. Ve SOLO el resumen de la secuencia, y predice
                        el mes siguiente desde cada posicion intermedia.

        La auxiliar no puede usar las estaticas. Esas variables (prev_age, prev_renta,
        n_prod_prev, altas_3m...) describen al cliente TAL COMO ESTA en el mes objetivo.
        Para la cabeza principal eso es legitimo, porque predice justo ese mes. Pero la
        posicion j predice un mes del pasado, y darle ahi las estaticas seria darle
        informacion posterior al mes que tiene que adivinar. No contaminaria la metrica
        final -- esa sale solo de la cabeza principal -- pero enseñaria al codificador a
        apoyarse en algo que en esa posicion no deberia existir.
        """

        def __init__(self, dim: int, n_estaticas: int, dropout: float = 0.1,
                     atar_pesos: bool = False):
            super().__init__()
            # Un embedding por canal en vez de una unica capa de 72 entradas. Es la misma
            # funcion (sumar tres productos matriciales equivale a multiplicar por la
            # concatenacion), reparametrizada para poder ATAR la cabeza auxiliar al
            # embedding de altas.
            self.emb_estado = nn.Linear(N_PRODUCTOS, dim, bias=False)
            self.emb_altas = nn.Linear(N_PRODUCTOS, dim, bias=False)
            self.emb_bajas = nn.Linear(N_PRODUCTOS, dim, bias=False)
            self.sesgo_proyeccion = nn.Parameter(torch.zeros(dim))

            self.estaticas = nn.Sequential(
                nn.Linear(n_estaticas, dim), nn.ReLU(), nn.Dropout(dropout)
            )
            self.salida = nn.Sequential(
                nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(dim, N_PRODUCTOS),
            )

            # Weight tying (SASRec, y antes los modelos de lenguaje): la cabeza que
            # PREDICE productos reutiliza la matriz con la que se EMBEBEN los productos,
            # en vez de aprender una segunda tabla de 24 x dim desde cero. Los dos
            # espacios representan lo mismo -- productos --, asi que compartirlos ahorra
            # parametros y obliga a que ambas vistas sean coherentes.
            self.atar_pesos = atar_pesos
            self.cabeza_aux = None if atar_pesos else nn.Linear(dim, N_PRODUCTOS)
            self.sesgo_aux = nn.Parameter(torch.zeros(N_PRODUCTOS)) if atar_pesos else None

        def proyectar(self, secuencia):
            """secuencia: [n, largo, 72] -> [n, largo, dim]."""
            estado = secuencia[..., :N_PRODUCTOS]
            altas = secuencia[..., N_PRODUCTOS:2 * N_PRODUCTOS]
            bajas = secuencia[..., 2 * N_PRODUCTOS:]
            return (self.emb_estado(estado) + self.emb_altas(altas)
                    + self.emb_bajas(bajas) + self.sesgo_proyeccion)

        def logits_auxiliares(self, oculto):
            """Logits por posicion, sin estaticas. oculto: [n, largo, dim]."""
            if self.atar_pesos:
                # emb_altas.weight es [dim, 24]; su traspuesta [24, dim] es justo la
                # matriz que espera F.linear como pesos de salida.
                return Fn.linear(oculto, self.emb_altas.weight.t(), self.sesgo_aux)
            return self.cabeza_aux(oculto)

        def combinar(self, resumen, estaticas):
            return self.salida(torch.cat([resumen, self.estaticas(estaticas)], dim=-1))

        def codificar(self, secuencia):
            """Cada subclase devuelve el estado oculto de TODAS las posiciones."""
            raise NotImplementedError

        def forward(self, secuencia, estaticas, con_auxiliar: bool = False):
            oculto = self.codificar(secuencia)
            principal = self.combinar(oculto[:, -1, :], estaticas)
            if not con_auxiliar:
                return principal
            return principal, self.logits_auxiliares(oculto)

    class ModeloGRU(CodificadorBase):
        """GRU4Rec: una recurrente que arrastra el estado a lo largo de la trayectoria."""

        def __init__(self, dim: int = 64, capas: int = 1, dropout: float = 0.1,
                     n_estaticas: int = len(ESTATICAS), atar_pesos: bool = False,
                     **_ignorados):
            super().__init__(dim, n_estaticas, dropout, atar_pesos)
            # El dropout entre capas de nn.GRU solo se aplica si hay mas de una; con
            # num_layers=1 PyTorch avisa de que el argumento no hace nada.
            self.gru = nn.GRU(dim, dim, num_layers=capas, batch_first=True,
                              dropout=dropout if capas > 1 else 0.0)

        def codificar(self, secuencia):
            salida, _ = self.gru(self.proyectar(secuencia))
            return salida

    class ModeloTransformer(CodificadorBase):
        """SASRec: auto-atencion causal sobre la trayectoria."""

        def __init__(self, dim: int = 64, cabezas: int = 4, capas: int = 2,
                     dropout: float = 0.1, largo: int = LARGO_MAXIMO,
                     n_estaticas: int = len(ESTATICAS), atar_pesos: bool = False,
                     **_ignorados):
            super().__init__(dim, n_estaticas, dropout, atar_pesos)
            # Sin codificacion posicional, la atencion es invariante al orden y el modelo
            # seria una bolsa de meses -- justo lo que se quiere superar.
            self.posicion = nn.Parameter(torch.randn(1, largo, dim) * 0.02)
            capa = nn.TransformerEncoderLayer(
                d_model=dim, nhead=cabezas, dim_feedforward=dim * 4,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(capa, num_layers=capas)

        def codificar(self, secuencia):
            x = self.proyectar(secuencia) + self.posicion[:, -secuencia.size(1):, :]
            # Mascara triangular: la posicion i solo puede atender a <= i.
            #
            # Con supervision solo en la ultima posicion esta mascara no evitaba ninguna
            # fuga (todos los meses de la ventana son anteriores al objetivo de todas
            # formas); solo recortaba capacidad. Lo que habilita es entrenar desde cada
            # posicion contra el mes siguiente, que es como SASRec saca su señal: sin la
            # mascara, la posicion j veria el mes j+1 que tiene que predecir.
            mascara = nn.Transformer.generate_square_subsequent_mask(
                secuencia.size(1), device=secuencia.device
            )
            return self.encoder(x, mask=mascara, is_causal=True)

    return ModeloGRU, ModeloTransformer


def fabricar_arquitectura(nombre: str, **config):
    """Devuelve una funcion sin argumentos que construye el modelo pedido.

    entrenar_red() recibe una fabrica y no un modelo ya construido para poder fijar la
    semilla justo antes de crear los pesos: si el modelo se construyera fuera, dos
    llamadas con la misma semilla partirian de inicializaciones distintas y la
    comparacion entre configuraciones mediria tambien el azar de la inicializacion.
    """
    ModeloGRU, ModeloTransformer = definir_modelos()
    clases = {"gru": ModeloGRU, "transformer": ModeloTransformer}
    if nombre not in clases:
        raise ValueError(f"Arquitectura desconocida: {nombre}")
    clase = clases[nombre]
    return lambda: clase(**config)


def entrenar_red(arquitectura, datos_train, datos_parada, reales_parada, epocas: int,
                 lote: int, lr: float, weight_decay: float = 1e-4, semilla: int = 42,
                 supervision: str = "ultima", lambda_aux: float = 0.3,
                 verbose: bool = True):
    """Bucle de entrenamiento. Devuelve (modelo, historial)."""
    import torch
    from torch import nn

    import copy
    import math

    torch.manual_seed(semilla)
    secuencias, estaticas, poseidos_train, objetivos, longitudes = datos_train

    modelo = arquitectura()
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=weight_decay)

    # Mascara de perdida: sin ella, el modelo gasta capacidad aprendiendo a no
    # recomendar productos que el cliente ya tiene -- y luego esa recomendacion se
    # descarta igualmente al construir la lista final (ver ya_tiene() en recommend.py).
    # Es capacidad tirada dos veces. Penalizando con -inf los logits de lo ya poseido
    # ANTES de la softmax, el gradiente deja de gastarse ahi y toda la capacidad del
    # modelo va a discriminar entre los productos que de verdad podria recomendar.
    poseidos_t = torch.from_numpy(poseidos_train).bool()

    criterio = nn.CrossEntropyLoss()
    criterio_aux = nn.BCEWithLogitsLoss(reduction="none")
    n = len(objetivos)
    largo = secuencias.shape[1]

    # --- Calentamiento + coseno, por PASO y no por epoca --------------------------------
    # AdamW arranca con estimaciones de momento vacias, asi que los primeros pasos van a
    # ciegas; con el learning rate ya a tope eso desordena los pesos justo cuando mas
    # importa. El calentamiento lineal los deja asentarse. Despues, coseno hasta cero.
    #
    # Va por paso y no por epoca porque el numero de epocas aqui es pequeño (entre 8 y
    # 20): un coseno con solo 8 escalones no es una curva, son ocho saltos.
    pasos_por_epoca = max(1, math.ceil(n / lote))
    pasos_totales = pasos_por_epoca * epocas
    pasos_calentamiento = min(500, max(1, pasos_totales // 20))

    def factor_lr(paso: int) -> float:
        if paso < pasos_calentamiento:
            return (paso + 1) / pasos_calentamiento
        avance = (paso - pasos_calentamiento) / max(1, pasos_totales - pasos_calentamiento)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, avance)))

    planificador = torch.optim.lr_scheduler.LambdaLR(optimizador, factor_lr)

    # --- Supervision por posicion -------------------------------------------------------
    # Con supervision="todas" la perdida no se calcula solo sobre el mes objetivo, sino
    # tambien sobre cada posicion intermedia: desde el mes j se predice que se contrata en
    # j+1. El objetivo de esa tarea ya esta dentro del tensor de entrada -- es el canal de
    # altas desplazado una posicion --, asi que no hace falta tocar nada de Spark.
    #
    # Por que compensa: la mascara causal ya impedia mirar hacia delante, pero solo se
    # usaba la ultima posicion, asi que de doce meses de trayectoria salia UNA etiqueta.
    # Supervisando todas las posiciones el mismo forward produce hasta once etiquetas mas
    # por cliente, practicamente gratis.
    #
    # Las tres mascaras que hacen falta, y que son donde esta la dificultad real:
    #   - relleno:  las posiciones de la izquierda de un cliente con poca historia son
    #               ceros inventados. Entrenar contra ellas enseña a no predecir nada.
    #   - ultima:   la posicion final no tiene "mes siguiente" dentro de la secuencia; su
    #               objetivo es el mes real y lo cubre la cabeza principal.
    #   - poseidos: en la posicion j no se puede contratar lo que ya se tenia en j.
    usar_auxiliar = supervision == "todas" and largo >= 2
    if usar_auxiliar:
        # [n, largo-1, 24]: lo que se contrata en el mes siguiente a cada posicion.
        objetivo_aux_np = secuencias[:, 1:, N_PRODUCTOS:2 * N_PRODUCTOS]
        # [n, largo-1, 24]: lo que ya se tenia en la posicion de partida.
        poseido_pos_np = secuencias[:, :-1, :N_PRODUCTOS]
        # [n, largo-1]: posicion valida si tanto ella como la siguiente son meses reales.
        indices_pos = np.arange(largo - 1)[None, :]
        primera_real = (largo - longitudes)[:, None]
        valida_np = (indices_pos >= primera_real).astype(np.float32)

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
        perdida_aux_total = 0.0
        for inicio in range(0, n, lote):
            indices = orden[inicio:inicio + lote]
            x_sec = torch.from_numpy(secuencias[indices]).float()
            x_est = torch.from_numpy(estaticas[indices])
            y = torch.from_numpy(objetivos[indices])
            poseidos_lote = poseidos_t[indices]

            optimizador.zero_grad()
            if usar_auxiliar:
                logits, logits_aux = modelo(x_sec, x_est, con_auxiliar=True)
            else:
                logits = modelo(x_sec, x_est)

            perdida = criterio(logits.masked_fill(poseidos_lote, float("-inf")), y)
            perdida_principal = float(perdida)

            if usar_auxiliar:
                objetivo_aux = torch.from_numpy(objetivo_aux_np[indices]).float()
                # El peso de cada casilla es 1 solo si la posicion es un mes real Y el
                # producto no se tenia ya. Multiplicar mascaras en vez de indexar deja el
                # calculo vectorizado, que con lotes de 512 x 15 x 24 importa.
                peso = (torch.from_numpy(valida_np[indices]).unsqueeze(-1)
                        * (1.0 - torch.from_numpy(poseido_pos_np[indices]).float()))
                bruta = criterio_aux(logits_aux[:, :-1, :], objetivo_aux) * peso
                # Se divide por el peso total y no por el numero de casillas: asi la
                # perdida no se diluye cuando el lote trae clientes de historial corto.
                media_aux = bruta.sum() / peso.sum().clamp(min=1.0)
                perdida = perdida + lambda_aux * media_aux
                perdida_aux_total += float(media_aux) * len(indices)

            perdida.backward()
            # Recorte de gradiente: las recurrentes son propensas a gradientes que
            # explotan, y con secuencias de 12 pasos ya se nota.
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optimizador.step()
            # El planificador avanza por PASO, no por epoca: ver factor_lr arriba.
            planificador.step()
            perdida_total += perdida_principal * len(indices)

        media = perdida_total / n
        registro = {"perdida": round(media, 5)}
        if usar_auxiliar:
            registro["perdida_aux"] = round(perdida_aux_total / n, 5)

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
            # La perdida auxiliar se imprime cuando existe: es la forma de comprobar de un
            # vistazo que la supervision por posicion esta actuando y no se ha quedado
            # apagada por un argumento que no llego.
            aux = f"  aux {registro['perdida_aux']:.5f}" if "perdida_aux" in registro else ""
            extra = (f"{aux}  MAP parada {registro.get('map_parada', '-')}"
                     if datos_parada is not None else aux)
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
    probs = puntuar_enmascarado(modelo, datos_parada)
    return map_at_k(probs, datos_parada[2], reales_parada)


def puntuar_enmascarado(modelo, datos, lote: int = 4096):
    """Devuelve la matriz de probabilidades [n, 24], con los poseidos a -inf ANTES de
    la softmax -- la misma mascara que se usa durante el entrenamiento (ver
    entrenar_red). Se usa dentro del early stopping, donde interesa el ranking real.
    """
    import torch

    secuencias, estaticas, poseidos = datos[0], datos[1], datos[2]
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

    secuencias, estaticas = datos[0], datos[1]
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
def preparar_tramos(spark, mes_validacion: str, largo: int = LARGO_MAXIMO,
                    supervision: str = "ultima", metricas: dict | None = None) -> dict:
    """Construye los tres tramos temporales y los devuelve ya en numpy.

    Tres tramos, no dos -- el mismo patron que churn.py (train/calibracion/test) y
    ensemble.py (train/pesos/test):

        entrenamiento   meses < t-1   ajusta los pesos
        parada          mes t-1       decide cuando dejar de entrenar, y en tuning.py
                                      tambien que configuracion gana
        test            mes t         no participa en ninguna decision

    Usar el mes de test para el early stopping seria elegir un hiperparametro -- el
    numero de epocas -- mirando la respuesta. Es el mismo tipo de fuga que el resto del
    proyecto evita en cada fase, colada por la puerta de atras.

    Esta funcion esta separada de main() porque tuning.py la necesita: preparar los datos
    cuesta unos cinco minutos de Spark y entrenar una configuracion, cuatro. Si la
    busqueda de hiperparametros relanzara el script entero por configuracion, pagaria la
    preparacion doce veces para obtener exactamente los mismos tensores.
    """
    registro = metricas if metricas is not None else {}

    with cronometro("preparacion_spark", registro):
        datos = construir_secuencias(spark, mes_validacion, largo=largo)
        mes_test = datos.filter(F.col("fecha_dato") == mes_validacion).select(
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
        # Parada: el mes justo anterior al de test.
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
        filas_test = test.collect()

    with cronometro("conversion_numpy", registro):
        datos_train = a_numpy(filas_train, largo=largo, con_objetivo=True)
        datos_parada = a_numpy(filas_parada, largo=largo)
        datos_test = a_numpy(filas_test, largo=largo)

        # Normalizacion con los estadisticos del ENTRENAMIENTO, aplicados a los tres.
        # Calcularlos sobre parada o test seria dejar que esos meses influyan en como se
        # representa el entrenamiento.
        est_train, media, desviacion = normalizar(datos_train[1])
        datos_train = (datos_train[0], est_train, *datos_train[2:])
        datos_parada = (datos_parada[0],
                        ((datos_parada[1] - media) / desviacion).astype(np.float32),
                        *datos_parada[2:])
        datos_test = (datos_test[0],
                      ((datos_test[1] - media) / desviacion).astype(np.float32),
                      *datos_test[2:])

    return {
        "train": datos_train,
        "parada": datos_parada,
        "test": datos_test,
        "reales_parada": [list(f["reales"]) for f in filas_parada],
        "reales_test": [list(f["reales"]) for f in filas_test],
        "ncodpers_test": [f["ncodpers"] for f in filas_test],
        "n_estaticas": len(ESTATICAS),
        "supervision": supervision,
        "mes_test": mes_test,
        "mes_parada": mes_parada,
        # Los estadisticos de normalizacion viajan con los tramos porque la exportacion
        # de la cartera completa tiene que aplicar EXACTAMENTE los mismos. Recalcularlos
        # sobre la cartera seria normalizar con datos que el modelo no vio al entrenar.
        "media": media,
        "desviacion": desviacion,
    }


def exportar_prob_compra(spark, modelo, mes_validacion: str, largo: int, media, desviacion,
                         destino, trozo: int = 40_000) -> int:
    """Puntua la cartera COMPLETA del mes y escribe el parquet que consume la Fase 5.

    Por que hace falta esto y no vale con el .npz que ya se guarda: `secuencia.py` solo
    puntuaba a los clientes que contrataron algo (27.875), porque para medir MAP@7 los
    demas no entran. Pero business.py construye una curva de captura sobre la cartera
    entera -- "contactando al 5% de los clientes se alcanza el X% de las contrataciones"
    no significa nada si solo se han puntuado los que iban a contratar. Hacen falta los
    926.663.

    Se escribe con el MISMO esquema que exporta recommend.py (ncodpers, prob_compra,
    contrato, recomendados) para que business.py y la demo no tengan que saber de donde
    vienen los numeros.

    Se procesa con toLocalIterator y en trozos, no con collect(). La cartera completa son
    casi un millon de Row de PySpark con cuarenta campos cada uno; traerlos todos a la vez
    a memoria de Python es la forma mas rapida de tumbar el contenedor. toLocalIterator
    trae una particion cada vez, y cada trozo se escribe como un fichero parquet suelto:
    Spark lee igual de bien un directorio con muchas partes.

    `prob_compra` es la probabilidad del mejor producto que el cliente AUN NO TIENE, con
    los poseidos enmascarados antes de la softmax. El enmascarado no es opcional aqui: el
    modelo se entreno con esa mascara puesta, asi que nunca aprendio a bajar por su cuenta
    el logit de lo que ya se tiene. Sin ella, el producto mas probable de mucha gente
    seria uno que ya tiene contratado.
    """
    import pandas as pd

    datos = construir_secuencias(spark, mes_validacion, largo=largo)
    cartera = datos.filter(F.col("fecha_dato") == mes_validacion).select(
        "ncodpers", "n_altas",
        "trayectoria", "trayectoria_altas", "trayectoria_bajas",
        *ESTATICAS, *[f"prev_{p}" for p in PRODUCTOS],
    )

    destino.mkdir(parents=True, exist_ok=True)
    # Restos de una ejecucion anterior: si no se borran, Spark leeria la union de las dos
    # y saldrian clientes duplicados con probabilidades de modelos distintos.
    for viejo in destino.glob("*.parquet"):
        viejo.unlink()

    estado = {"parte": 0, "total": 0}

    def volcar(filas: list) -> None:
        if not filas:
            return
        secuencias, estaticas, poseidos, _, _ = a_numpy(filas, largo=largo)
        estaticas = ((estaticas - media) / desviacion).astype(np.float32)
        probabilidades = puntuar_enmascarado(modelo, (secuencias, estaticas, poseidos))

        # Los poseidos ya salen con probabilidad ~0 del enmascarado; ponerlos a -1 antes
        # de ordenar evita que un empate a cero los cuele en el top 7.
        ordenables = np.where(poseidos == 1, -1.0, probabilidades)
        orden = np.argsort(-ordenables, axis=1)[:, :K]

        pd.DataFrame({
            "ncodpers": [f["ncodpers"] for f in filas],
            "prob_compra": ordenables.max(axis=1).astype(float),
            "contrato": [1 if (f["n_altas"] or 0) > 0 else 0 for f in filas],
            "recomendados": [[PRODUCTOS[j] for j in fila] for fila in orden],
        }).to_parquet(destino / f"part-{estado['parte']:05d}.parquet", index=False)

        estado["parte"] += 1
        estado["total"] += len(filas)
        print(f"    ... {estado['total']:,} clientes puntuados", flush=True)

    buffer: list = []
    for fila in cartera.toLocalIterator():
        buffer.append(fila)
        if len(buffer) >= trozo:
            volcar(buffer)
            buffer = []
    volcar(buffer)

    return estado["total"]


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
    parser.add_argument("--capas", type=int, default=2,
                        help="Capas del encoder (Transformer) o del GRU")
    parser.add_argument("--cabezas", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay de AdamW")
    parser.add_argument("--largo", type=int, default=LARGO_SECUENCIA,
                        help=f"Meses de trayectoria que ve el modelo (tope {LARGO_MAXIMO})")
    parser.add_argument("--atar-pesos", action="store_true",
                        help="Comparte la matriz de embedding de productos con la cabeza "
                             "auxiliar (weight tying, como en SASRec)")
    parser.add_argument("--supervision", default="ultima", choices=["ultima", "todas"],
                        help="'todas' entrena tambien desde cada posicion intermedia "
                             "contra el mes siguiente")
    parser.add_argument("--lambda-aux", type=float, default=0.3,
                        help="Peso de la perdida auxiliar cuando --supervision todas")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--exportar-parquet", action="store_true",
                        help="Puntua la cartera COMPLETA con el mejor modelo y escribe "
                             "data/export/prob_compra, que es lo que consumen business.py "
                             "y la demo. Sin esto la capa de negocio se queda con XGBoost.")
    parser.add_argument("--distribuido", action="store_true",
                        help="Lanza el entrenamiento con TorchDistributor de Spark")
    args = parser.parse_args()

    if args.largo > LARGO_MAXIMO:
        parser.error(f"--largo no puede pasar de {LARGO_MAXIMO}")

    spark = crear_sesion("secuencia")
    spark.sparkContext.setLogLevel("ERROR")
    metricas: dict = {
        "mes_validacion": args.mes_validacion,
        "largo_secuencia": args.largo,
        "epocas": args.epocas,
        "dim": args.dim,
        "capas": args.capas,
        "cabezas": args.cabezas,
        "dropout": args.dropout,
        "weight_decay": args.wd,
        "atar_pesos": args.atar_pesos,
        "supervision": args.supervision,
        "semilla": args.semilla,
    }

    print("\n=== FASE 3b: modelos de secuencia ===\n")

    # --- Preparacion en Spark -----------------------------------------------------------
    tramos = preparar_tramos(spark, args.mes_validacion, args.largo,
                             supervision=args.supervision, metricas=metricas)
    datos_train = tramos["train"]
    datos_parada = tramos["parada"]
    datos_val = tramos["test"]
    reales = tramos["reales_test"]
    reales_parada = tramos["reales_parada"]

    metricas["ejemplos_entrenamiento"] = len(datos_train[0])
    metricas["clientes_parada"] = len(datos_parada[0])
    metricas["clientes_validacion"] = len(datos_val[0])
    print(f"  Ejemplos de entrenamiento: {len(datos_train[0]):,}")
    print(f"  Clientes de parada (early stopping): {len(datos_parada[0]):,}")
    print(f"  Clientes de test (medicion final):   {len(datos_val[0]):,}")

    memoria_mb = datos_train[0].nbytes / 1024 ** 2
    metricas["memoria_secuencias_mb"] = round(memoria_mb, 1)
    print(f"  Secuencias en memoria: {memoria_mb:,.0f} MB "
          f"(uint8; en float32 serian {4 * memoria_mb:,.0f} MB)\n")

    # --- Entrenamiento ------------------------------------------------------------------
    comun = dict(dim=args.dim, capas=args.capas, cabezas=args.cabezas,
                 dropout=args.dropout, largo=args.largo, atar_pesos=args.atar_pesos,
                 n_estaticas=tramos["n_estaticas"])
    arquitecturas = {
        "gru": fabricar_arquitectura("gru", **comun),
        "transformer": fabricar_arquitectura("transformer", **comun),
    }

    resultados = {}
    probabilidades_por_modelo = {}
    modelos_entrenados = {}
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
                reales_parada, args.epocas, args.lote, args.lr, args.wd,
                args.semilla, args.supervision, args.lambda_aux, False,
            )
        else:
            modelo, historial = entrenar_red(
                arquitecturas[nombre], datos_train, datos_parada, reales_parada,
                epocas=args.epocas, lote=args.lote, lr=args.lr, weight_decay=args.wd,
                semilla=args.semilla, supervision=args.supervision,
                lambda_aux=args.lambda_aux,
            )

        segundos = time.perf_counter() - inicio
        probabilidades = puntuar(modelo, datos_val)
        puntuacion = map_at_k(probabilidades, datos_val[2], reales)

        probabilidades_por_modelo[nombre] = probabilidades
        modelos_entrenados[nombre] = modelo
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
            ncodpers=np.array(tramos["ncodpers_test"]),
            poseidos=datos_val[2],
            **probabilidades_por_modelo,
        )
        print(f"\n  Probabilidades guardadas en {destino}")

    # --- La cartera completa, para la capa de negocio -----------------------------------
    if args.exportar_parquet and modelos_entrenados:
        mejor = max(resultados, key=resultados.get)
        print(f"\n  --- Exportando la cartera completa con '{mejor}' "
              f"(MAP@7 {resultados[mejor]:.5f}) ---", flush=True)
        with cronometro("exportar_prob_compra", metricas):
            total = exportar_prob_compra(
                spark, modelos_entrenados[mejor], args.mes_validacion, args.largo,
                tramos["media"], tramos["desviacion"], DATA_EXPORT / "prob_compra",
            )
        metricas["modelo_exportado"] = mejor
        metricas["clientes_exportados"] = total
        print(f"  {total:,} clientes escritos en {DATA_EXPORT / 'prob_compra'}")

    guardar_metricas("secuencia_metrics.json", metricas)
    spark.stop()


if __name__ == "__main__":
    main()
