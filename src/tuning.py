"""
FASE 3f -- Busqueda de hiperparametros del modelo de secuencia, sin mirar el test.

    python src/tuning.py --mes-validacion 2016-05-28 --configuraciones 12

--------------------------------------------------------------------------------------
Por que un script aparte y no un bucle dentro de secuencia.py
--------------------------------------------------------------------------------------
Porque la parte cara no es entrenar, es preparar los datos. `construir_secuencias` tarda
unos cinco minutos de Spark y produce unos tensores de numpy de 400 MB. Entrenar una
configuracion tarda entre dos y ocho minutos. Si la busqueda se hiciera relanzando
secuencia.py una vez por configuracion, cada prueba pagaria de nuevo los cinco minutos de
Spark: con doce configuraciones son una hora tirada preparando lo mismo doce veces.

Aqui Spark se usa una vez, se cierra la sesion en cuanto los tensores estan en memoria, y
el resto es PyTorch puro sobre los mismos datos.

--------------------------------------------------------------------------------------
La regla que no se rompe
--------------------------------------------------------------------------------------
    Ningun hiperparametro se elige mirando el mes de test.

Se usan tres tramos, como en el resto del proyecto:

    entrenamiento   meses < t-1     ajusta los pesos
    parada          mes t-1         elige epoca Y configuracion
    test            mes t           se mide UNA vez, al final

Esto es mas estricto que lo que se ve en muchos cuadernos de Kaggle, donde se prueban
veinte configuraciones contra el conjunto de test y se reporta la mejor. Eso no mide lo
bien que generaliza el modelo: mide lo bien que se ha ajustado el humano al test. Con
doce configuraciones y un test de 27.000 clientes, la mejor por azar puede sacar varias
milesimas de MAP que no existen.

El coste de hacerlo bien es que el numero final puede ser PEOR que el que saldria
eligiendo contra el test. Se acepta: es el unico que se puede defender.

--------------------------------------------------------------------------------------
Busqueda aleatoria y no rejilla
--------------------------------------------------------------------------------------
Con siete hiperparametros, una rejilla de solo tres valores cada uno son 2.187
combinaciones. Inviable, y ademas desperdiciada: en la practica dos o tres de los siete
explican casi toda la variacion y el resto da igual. Una rejilla gasta el mismo esfuerzo
en todos; una busqueda aleatoria cubre mas valores distintos de los parametros que si
importan con el mismo numero de pruebas (Bergstra y Bengio, 2012).

Se reporta la tabla COMPLETA de configuraciones probadas, no solo la ganadora. Si la
diferencia entre la mejor y la mediana es de milesimas, eso es informacion: significa que
el modelo es insensible a los hiperparametros y que no habia nada que ganar por ahi.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np

from common import OUTPUTS, crear_sesion, cronometro, guardar_metricas
from secuencia import (
    LARGO_MAXIMO,
    MES_VALIDACION,
    entrenar_red,
    fabricar_arquitectura,
    map_at_k,
    preparar_tramos,
    puntuar,
    recortar_largo,
)


# --------------------------------------------------------------------------------------
# El espacio de busqueda
# --------------------------------------------------------------------------------------
# Cada entrada es la lista de valores posibles. Se muestrea uniforme salvo el learning
# rate, que se muestrea log-uniforme: la diferencia entre 0.0005 y 0.001 importa tanto
# como la que hay entre 0.002 y 0.004, y un muestreo lineal apenas visitaria la parte baja.
ESPACIO = {
    "dim": [48, 64, 96, 128],
    "capas": [1, 2, 3],
    "cabezas": [2, 4, 8],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "lote": [256, 512, 1024],
    "wd": [0.0, 1e-5, 1e-4, 1e-3],
    "largo": [8, 12, 16],
    "atar_pesos": [True, False],
    # Peso de la perdida auxiliar frente a la principal. Es un hiperparametro de pleno
    # derecho: con 0 la supervision por posicion se apaga, y con un valor alto el modelo
    # puede acabar optimizando la tarea auxiliar a costa de la que se mide.
    "lambda_aux": [0.1, 0.3, 0.5, 1.0],
}
LR_MINIMO, LR_MAXIMO = 5e-4, 5e-3


def muestrear_configuracion(rng: random.Random) -> dict:
    """Una configuracion al azar del espacio, con las restricciones que impone el modelo."""
    config = {clave: rng.choice(valores) for clave, valores in ESPACIO.items()}

    # El learning rate, log-uniforme.
    config["lr"] = float(
        np.exp(rng.uniform(np.log(LR_MINIMO), np.log(LR_MAXIMO)))
    )

    # Restriccion dura de la atencion multi-cabeza: d_model tiene que ser divisible entre
    # el numero de cabezas, porque cada cabeza se queda un trozo igual del vector. Si no
    # cuadra, PyTorch lanza una excepcion al construir el modelo. Se corrige bajando al
    # divisor valido mas cercano en vez de descartar la configuracion, para no sesgar el
    # muestreo hacia las dimensiones con muchos divisores.
    if config["dim"] % config["cabezas"] != 0:
        validas = [c for c in ESPACIO["cabezas"] if config["dim"] % c == 0]
        config["cabezas"] = max(validas) if validas else 1

    return config


def clave_configuracion(config: dict) -> tuple:
    """Para descartar repetidas: la busqueda aleatoria puede sacar la misma dos veces."""
    return tuple(sorted((k, round(v, 6) if isinstance(v, float) else v)
                        for k, v in config.items()))


# --------------------------------------------------------------------------------------
def submuestrear(datos, fraccion: float, semilla: int):
    """Se queda con una fraccion de los ejemplos de entrenamiento, al azar.

    La busqueda no necesita el entrenamiento completo. Lo que se pide de ella es ORDENAR
    configuraciones, no medir ninguna: si la configuracion A es mejor que la B con el 40%
    de los datos, casi siempre lo sigue siendo con el 100%. Entrenar cada prueba con todo
    multiplicaria por 2,5 el coste de la busqueda para afinar un numero que despues se
    descarta, porque la ganadora se reentrena a fondo de todas formas.

    Es la misma idea que hay detras de successive halving: gastar poco en descartar y
    guardar el presupuesto para las finalistas.
    """
    if fraccion >= 1.0:
        return datos
    n = len(datos[0])
    rng = np.random.RandomState(semilla)
    indices = rng.choice(n, size=int(n * fraccion), replace=False)
    return tuple(parte[indices] for parte in datos)


def evaluar_configuracion(config: dict, tramos: dict, arquitectura_nombre: str,
                          epocas: int, semilla: int, fraccion: float = 1.0,
                          verbose: bool = False) -> dict:
    """Entrena una configuracion y la puntua EN EL MES DE PARADA.

    Devuelve el MAP de parada, el modelo entrenado y cuanto ha tardado. El modelo se
    devuelve para no tener que reentrenar la ganadora despues.
    """
    largo = config["largo"]
    datos_train = submuestrear(recortar_largo(tramos["train"], largo), fraccion, semilla)
    datos_parada = recortar_largo(tramos["parada"], largo)

    fabrica = fabricar_arquitectura(
        arquitectura_nombre,
        dim=config["dim"],
        capas=config["capas"],
        cabezas=config["cabezas"],
        dropout=config["dropout"],
        largo=largo,
        atar_pesos=config["atar_pesos"],
        n_estaticas=tramos["n_estaticas"],
    )

    inicio = time.perf_counter()
    modelo, historial = entrenar_red(
        fabrica, datos_train, datos_parada, tramos["reales_parada"],
        epocas=epocas, lote=config["lote"], lr=config["lr"],
        weight_decay=config["wd"], semilla=semilla,
        # El interruptor de la supervision por posicion vive en entrenar_red, no en la
        # preparacion de datos: preparar_tramos solo transporta el ajuste.
        supervision=tramos["supervision"], lambda_aux=config["lambda_aux"],
        verbose=verbose,
    )
    segundos = time.perf_counter() - inicio

    # El MAP de parada que cuenta es el MEJOR visto durante el entrenamiento, que es el
    # que corresponde a los pesos que entrenar_red devuelve (restaura el mejor estado, no
    # el ultimo). Coger el de la ultima epoca puntuaria unos pesos que no son los que hay
    # dentro del modelo.
    map_parada = max(r.get("map_parada", 0.0) for r in historial)

    return {
        "config": config,
        "map_parada": round(map_parada, 5),
        "epocas_usadas": len(historial),
        "segundos": round(segundos, 1),
        "modelo": modelo,
        "historial": historial,
    }


# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Busqueda de hiperparametros del modelo de secuencia"
    )
    parser.add_argument("--mes-validacion", default=MES_VALIDACION)
    parser.add_argument("--arquitectura", default="transformer",
                        choices=["transformer", "gru"])
    parser.add_argument("--configuraciones", type=int, default=12,
                        help="Cuantas configuraciones probar en el mes de parada")
    parser.add_argument("--epocas-busqueda", type=int, default=8,
                        help="Tope de epocas durante la busqueda. Corto a proposito: la "
                             "busqueda solo tiene que ordenar configuraciones")
    parser.add_argument("--epocas-final", type=int, default=20,
                        help="Tope de epocas de la ganadora. Alto porque el early "
                             "stopping corta solo, y en las pruebas el MAP de parada "
                             "seguia subiendo en la epoca 10")
    parser.add_argument("--fraccion-busqueda", type=float, default=0.4,
                        help="Fraccion del entrenamiento que usa la busqueda. La "
                             "ganadora se reentrena siempre con el 100%%")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--semillas-finales", type=int, default=3,
                        help="Cuantas semillas entrenar con la configuracion ganadora, "
                             "para separar mejora real de ruido de inicializacion")
    parser.add_argument("--supervision", default="todas", choices=["todas", "ultima"])
    args = parser.parse_args()

    metricas: dict = {
        "mes_validacion": args.mes_validacion,
        "arquitectura": args.arquitectura,
        "configuraciones_probadas": args.configuraciones,
        "epocas_busqueda": args.epocas_busqueda,
        "epocas_final": args.epocas_final,
        "fraccion_busqueda": args.fraccion_busqueda,
        "supervision": args.supervision,
    }

    print("\n=== FASE 3f: busqueda de hiperparametros ===\n")
    print("  Regla: todo se elige en el mes de parada (t-1). El mes de test (t) se mide")
    print("  una sola vez, al final, con la configuracion ya cerrada.\n")

    # --- Los datos, una vez ------------------------------------------------------------
    spark = crear_sesion("tuning")
    spark.sparkContext.setLogLevel("ERROR")
    with cronometro("preparacion_spark", metricas):
        tramos = preparar_tramos(spark, args.mes_validacion, LARGO_MAXIMO,
                                 supervision=args.supervision)
    # Spark ya no hace falta: liberar sus 9 GB antes de empezar a entrenar evita pelear
    # por la memoria del contenedor con los tensores de PyTorch.
    spark.stop()

    metricas["ejemplos_entrenamiento"] = len(tramos["train"][0])
    metricas["clientes_parada"] = len(tramos["parada"][0])
    metricas["clientes_test"] = len(tramos["test"][0])
    print(f"  Entrenamiento: {metricas['ejemplos_entrenamiento']:,} ejemplos")
    print(f"  Parada:        {metricas['clientes_parada']:,} clientes")
    print(f"  Test:          {metricas['clientes_test']:,} clientes (intocable hasta el final)\n")

    # --- La busqueda, toda en el mes de parada -----------------------------------------
    rng = random.Random(args.semilla)
    vistas: set = set()
    resultados: list[dict] = []

    print(f"  --- Probando {args.configuraciones} configuraciones en el mes de parada ---\n")
    with cronometro("busqueda", metricas):
        for i in range(args.configuraciones):
            # Hasta 50 intentos de sacar una configuracion no repetida; con un espacio de
            # miles de combinaciones no deberia agotarse nunca, pero un bucle while sin
            # tope es una forma facil de colgar un job de una hora.
            for _ in range(50):
                config = muestrear_configuracion(rng)
                if clave_configuracion(config) not in vistas:
                    break
            vistas.add(clave_configuracion(config))

            resumen = (f"dim={config['dim']} capas={config['capas']} "
                       f"cabezas={config['cabezas']} drop={config['dropout']} "
                       f"lr={config['lr']:.2e} wd={config['wd']:.0e} "
                       f"lote={config['lote']} largo={config['largo']} "
                       f"atadas={'si' if config['atar_pesos'] else 'no'} "
                       f"aux={config['lambda_aux']}")
            print(f"  [{i + 1}/{args.configuraciones}] {resumen}", flush=True)

            resultado = evaluar_configuracion(
                config, tramos, args.arquitectura, args.epocas_busqueda, args.semilla,
                fraccion=args.fraccion_busqueda,
            )
            resultados.append(resultado)
            print(f"        MAP parada = {resultado['map_parada']:.5f}   "
                  f"({resultado['epocas_usadas']} epocas, {resultado['segundos']:.0f}s)\n",
                  flush=True)

    resultados.sort(key=lambda r: r["map_parada"], reverse=True)
    mejor = resultados[0]
    peor = resultados[-1]
    mediana = resultados[len(resultados) // 2]

    # La tabla completa, no solo la ganadora. Si el rango es de milesimas, la conclusion
    # honesta es que los hiperparametros no eran el cuello de botella.
    metricas["busqueda_completa"] = [
        {"config": r["config"], "map_parada": r["map_parada"],
         "epocas_usadas": r["epocas_usadas"], "segundos": r["segundos"]}
        for r in resultados
    ]
    metricas["mejor_config"] = mejor["config"]
    metricas["map_parada_mejor"] = mejor["map_parada"]
    metricas["map_parada_mediana"] = mediana["map_parada"]
    metricas["map_parada_peor"] = peor["map_parada"]
    metricas["rango_busqueda"] = round(mejor["map_parada"] - peor["map_parada"], 5)

    print("\n  --- Resultado de la busqueda (mes de parada) ---")
    print(f"    Ojo: estos MAP salen del {args.fraccion_busqueda:.0%} del entrenamiento y "
          f"un tope de {args.epocas_busqueda} epocas.")
    print("    Sirven para ORDENAR configuraciones, no como medida. El numero que cuenta")
    print("    es el de la medicion final, con el entrenamiento completo.\n")
    print(f"    mejor:    {mejor['map_parada']:.5f}")
    print(f"    mediana:  {mediana['map_parada']:.5f}")
    print(f"    peor:     {peor['map_parada']:.5f}")
    print(f"    rango:    {metricas['rango_busqueda']:.5f}")
    if metricas["rango_busqueda"] < 0.005:
        print("    -> El rango es minimo: el modelo es insensible a estos hiperparametros")
        print("       y no habia mucho que ganar ajustandolos.")
    print(f"\n    Configuracion elegida: {mejor['config']}\n")

    # --- Varias semillas con la ganadora -----------------------------------------------
    # Una sola semilla no distingue "esta configuracion es mejor" de "esta inicializacion
    # tuvo suerte". Se entrena la ganadora con varias semillas y se reporta media y
    # desviacion. La eleccion de CUAL exportar se hace por el MAP de parada, nunca por el
    # de test.
    print(f"  --- Reentrenando la ganadora con {args.semillas_finales} semillas ---\n")
    finales = []
    with cronometro("semillas_finales", metricas):
        for s in range(args.semillas_finales):
            semilla = args.semilla + s * 1000
            # Aqui si: entrenamiento completo y tope de epocas alto. Es la unica
            # configuracion que se va a usar, asi que se le da todo el presupuesto.
            resultado = evaluar_configuracion(
                mejor["config"], tramos, args.arquitectura, args.epocas_final, semilla,
                fraccion=1.0,
            )
            finales.append((semilla, resultado))
            print(f"    semilla {semilla}: MAP parada = {resultado['map_parada']:.5f}",
                  flush=True)

    maps_parada = [r["map_parada"] for _, r in finales]
    metricas["semillas"] = {
        "valores": maps_parada,
        "media_parada": round(float(np.mean(maps_parada)), 5),
        "desviacion_parada": round(float(np.std(maps_parada)), 5),
    }
    print(f"\n    media {np.mean(maps_parada):.5f}  desviacion {np.std(maps_parada):.5f}")
    if np.std(maps_parada) > 0.003:
        print("    -> La desviacion entre semillas es del mismo orden que las diferencias")
        print("       de la busqueda: cuidado al leer la tabla como si fuera una ordenacion.")

    # El modelo que se exporta es el de la semilla con mejor MAP DE PARADA. Elegirlo por
    # el MAP de test seria la misma fuga que todo este script evita.
    semilla_elegida, mejor_final = max(finales, key=lambda par: par[1]["map_parada"])
    metricas["semilla_elegida"] = semilla_elegida

    # --- La medicion final: el test, una sola vez --------------------------------------
    print("\n  --- MEDICION FINAL (mes de test, se toca una sola vez) ---\n")
    largo = mejor["config"]["largo"]
    datos_test = recortar_largo(tramos["test"], largo)
    probabilidades = puntuar(mejor_final["modelo"], datos_test)
    map_test = map_at_k(probabilidades, datos_test[2], tramos["reales_test"])

    metricas["map_test"] = round(map_test, 5)
    print(f"    MAP@7 en test = {map_test:.5f}")

    # Comparacion contra lo que ya habia, si esta disponible.
    ruta_previa = OUTPUTS / "secuencia_metrics.json"
    if ruta_previa.exists():
        previas = json.loads(ruta_previa.read_text(encoding="utf-8"))
        referencia = previas.get("map7", {}).get(args.arquitectura)
        if referencia:
            delta = map_test - referencia
            metricas["referencia_sin_tuning"] = referencia
            metricas["aporte_tuning"] = round(delta, 5)
            metricas["aporte_tuning_pct"] = round(100 * delta / referencia, 2)
            print(f"    Sin tuning era {referencia:.5f}  ->  "
                  f"{'+' if delta >= 0 else ''}{delta:.5f} "
                  f"({'+' if delta >= 0 else ''}{100 * delta / referencia:.2f}%)")
            if delta <= 0:
                print("\n    La busqueda NO ha mejorado el modelo. Se reporta igual: los")
                print("    valores por defecto ya estaban bien elegidos para este problema.")

    guardar_metricas("tuning_metrics.json", metricas)


if __name__ == "__main__":
    main()
