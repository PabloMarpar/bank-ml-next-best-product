"""
Demo: el plan comercial del mes, cliente a cliente.

    streamlit run app/streamlit_app.py

--------------------------------------------------------------------------------------
Por que esta demo NO ejecuta Spark
--------------------------------------------------------------------------------------
Streamlit Cloud da una maquina pequena y sin Java: no puede levantar Spark, y aunque
pudiera, arrancar una sesion de Spark por cada visita para consultar un cliente seria
absurdo -- tardaria mas en arrancar el motor que en responder.

El reparto correcto es el que se usa en produccion de verdad: el calculo pesado ocurre en
lote una vez al mes (el pipeline de PySpark), deja las predicciones escritas, y la capa de
consulta solo lee. Esta demo es esa capa de consulta.

El fichero que lee es una muestra de clientes generada por scripts/exportar_demo.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from common import NOMBRES_PRODUCTO  # noqa: E402

FICHERO_DEMO = Path(__file__).resolve().parent / "demo_data.parquet"

st.set_page_config(page_title="Next Best Product", page_icon="🏦", layout="wide")


@st.cache_data
def cargar() -> pd.DataFrame | None:
    if not FICHERO_DEMO.exists():
        return None
    return pd.read_parquet(FICHERO_DEMO)


def etiqueta(codigo: str) -> str:
    return NOMBRES_PRODUCTO.get(codigo, codigo)


def cuadrante(prob_compra: float, prob_baja: float,
              corte_compra: float, corte_baja: float) -> tuple[str, str, str]:
    """Devuelve (nombre, accion, color) del cuadrante valor x riesgo."""
    alta_compra = prob_compra >= corte_compra
    alto_riesgo = prob_baja >= corte_baja
    if alta_compra and alto_riesgo:
        return ("Retener primero", "Retenerlo antes de ofrecerle nada nuevo", "#b45309")
    if alta_compra:
        return ("Vender", "Ofrecerle el siguiente producto", "#15803d")
    if alto_riesgo:
        return ("Retener", "Campana de retencion", "#b91c1c")
    return ("No tocar", "No gastar contacto este mes", "#64748b")


st.title("Next Best Product")
st.caption(
    "A quien llamar este mes, que ofrecerle, y a quien se esta a punto de perder. "
    "Las predicciones vienen del pipeline de PySpark; esta pagina solo las consulta."
)

datos = cargar()

if datos is None:
    st.warning(
        "No hay datos de demo. Generalos ejecutando el pipeline y despues:\n\n"
        "```\npython scripts/exportar_demo.py\n```"
    )
    st.stop()

corte_compra = datos["prob_compra"].quantile(0.75)
corte_baja = datos["prob_baja"].quantile(0.75)

# ---------------------------------------------------------------------------------------
# Vista de cartera
# ---------------------------------------------------------------------------------------
izquierda, derecha = st.columns([1, 2])

with izquierda:
    st.subheader("La cartera")
    st.metric("Clientes en la muestra", f"{len(datos):,}")

    resumen = datos.apply(
        lambda f: cuadrante(f["prob_compra"], f["prob_baja"], corte_compra, corte_baja)[0],
        axis=1,
    ).value_counts()
    st.dataframe(
        resumen.rename("Clientes").to_frame(),
        use_container_width=True,
    )

    st.caption(
        "Los cortes se fijan en el percentil 75 de cada probabilidad: lo que importa es "
        "separar el cuarto superior, no que la probabilidad pase de 0,5."
    )

with derecha:
    st.subheader("Los primeros de la lista")
    st.caption("Ordenados por propension de compra, que es el orden en que llamaria el equipo.")
    tabla = datos.nlargest(15, "prob_compra")[
        ["ncodpers", "prob_compra", "prob_baja", "n_prod_prev"]
    ].rename(columns={
        "ncodpers": "Cliente",
        "prob_compra": "Propension de compra",
        "prob_baja": "Riesgo de baja",
        "n_prod_prev": "Productos que tiene",
    })
    st.dataframe(
        tabla.style.format({
            "Propension de compra": "{:.1%}", "Riesgo de baja": "{:.1%}",
        }),
        use_container_width=True, hide_index=True,
    )

st.divider()

# ---------------------------------------------------------------------------------------
# Ficha de un cliente
# ---------------------------------------------------------------------------------------
st.subheader("Ficha de cliente")

cliente_id = st.selectbox(
    "Cliente",
    options=datos.nlargest(300, "prob_compra")["ncodpers"].tolist(),
    format_func=lambda x: f"Cliente {x}",
)
fila = datos[datos["ncodpers"] == cliente_id].iloc[0]

nombre_cuadrante, accion, color = cuadrante(
    fila["prob_compra"], fila["prob_baja"], corte_compra, corte_baja
)

c1, c2, c3 = st.columns(3)
c1.metric("Propension de compra", f"{fila['prob_compra']:.1%}")
c2.metric("Riesgo de baja", f"{fila['prob_baja']:.1%}")
c3.metric("Productos contratados", int(fila["n_prod_prev"]))

st.markdown(
    f"<div style='padding:14px 18px;border-radius:8px;background:{color};color:white;'>"
    f"<strong>{nombre_cuadrante}</strong> &mdash; {accion}</div>",
    unsafe_allow_html=True,
)

st.write("")
izq, der = st.columns(2)

with izq:
    st.markdown("**Qué ofrecerle, por orden**")
    recomendados = list(fila.get("recomendados") or [])
    if recomendados:
        for posicion, codigo in enumerate(recomendados[:5], start=1):
            st.write(f"{posicion}. {etiqueta(codigo)}")
    else:
        st.write("Sin recomendación disponible.")

with der:
    st.markdown("**Qué tiene ya contratado**")
    poseidos = list(fila.get("poseidos") or [])
    if poseidos:
        for codigo in poseidos:
            st.write(f"- {etiqueta(codigo)}")
    else:
        st.write("Ningún producto el mes anterior.")

st.divider()
st.caption(
    "Datos: Santander Product Recommendation (competición pública de Kaggle, 2016). "
    "Las probabilidades salen de modelos entrenados con información estrictamente anterior "
    "al mes que se predice."
)
