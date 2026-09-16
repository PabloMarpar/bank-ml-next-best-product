# Next best product (banca) — instrucciones para Claude Code

Este fichero lo carga Claude Code automaticamente al abrir cualquier sesion en este repo.
Es la parte ESTABLE: que es el proyecto, como se ejecuta y que reglas respetar. El estado
vivo (que esta hecho, que falta, resultados medidos) esta en `CONTEXTO.md`, importado abajo.

## Que es

Sistema de *next best product* sobre datos reales de banca (Santander Product Recommendation,
Kaggle, 13,6M filas). Decide a que clientes llama el equipo comercial el mes que viene, que
producto ofrecer a cada uno, y a cuales esta a punto de perder. PySpark sobre Docker.

Es un proyecto de **portfolio**: el rigor metodologico importa tanto como la metrica. Un
resultado negativo bien medido vale mas que uno positivo sin controlar.

## Como se ejecuta (leer antes de lanzar nada)

La maquina NO tiene Java ni Python en el host. Todo corre dentro del contenedor.

```bash
MSYS_NO_PATHCONV=1 docker compose run --rm -e PYTHONPATH=/app/src spark \
  python -u src/<script>.py [args] > logs/<nombre>.log 2>&1
```

- `MSYS_NO_PATHCONV=1` es **obligatorio** en Git Bash de Windows. Sin el, MSYS reescribe
  `/app/...` como ruta de Windows y Spark falla con `No FileSystem for scheme "C"`.
- Verificar sintaxis ANTES de lanzar un job largo:
  `docker compose run --rm spark python -c "import ast; ast.parse(open('/app/src/X.py').read())"`
- Prueba rapida sin tocar datos reales (~2 min): `scripts/smoke_test.py`
- Para encadenar varias ejecuciones largas hay scripts de "tanda" en `scripts/` que sirven
  de plantilla (`ejecutar_todo.sh`, `segunda_tanda.sh`, `tercera_tanda.sh`, `cuarta_tanda.sh`).
- Config Spark: `local[*]`, 9 GB de driver, 64 particiones de shuffle.

## Reglas del proyecto

1. **Nada de fuga de datos.** Cualquier decision (umbral, pesos, epocas, reglas de router)
   se fija en un tramo temporal ANTERIOR al de medicion. El patron es siempre de tres
   tramos: train / eleccion / test. Si aparece un hiperparametro elegido mirando el test,
   es un bug, no un detalle.
2. **Respaldar antes de sobrescribir.** Los `.npz` de `data/export/` y los JSON de
   `outputs/` son resultados de horas de computo. Copiar a un temporal antes de re-ejecutar.
3. **Los resultados negativos se reportan.** El ALS no aportaba, el ensemble no aportaba,
   el ranker no mejoraba: eso esta en el README a proposito.
4. **No subir nada a GitHub sin confirmacion explicita del usuario.**
5. **Actualizar `CONTEXTO.md`** al terminar cada fase o hallazgo relevante, para que una
   sesion nueva no tenga que releer nada.

## Ficheros que NO van a GitHub

`DIARIO_PROBLEMAS.md`, `NOTAS_ENTREVISTA.md` (notas privadas de trabajo), `data/`, `logs/`, `*.npz`, credenciales de
Kaggle. Repasar `git status` con los ojos antes de cada commit, no confiar solo en
`.gitignore`.

## Documentos del repo

- `README.md` — narrativa pulida para quien visita el repo
- `RESUMEN.md` — la version sin jerga
- `PERFORMANCE.md` — los 7 experimentos de rendimiento de Spark
- `CONTEXTO.md` — cuaderno de bitacora (estado vivo)
- `DIARIO_PROBLEMAS.md` — notas privadas de trabajo, no se versiona

---

@CONTEXTO.md
