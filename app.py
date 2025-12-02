import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Módulo de Regresiones", layout="wide")

excel_path = "data/datos.xlsx"

try:
    df = pd.read_excel(excel_path)

    st.subheader("Datos cargados desde Excel")
    st.dataframe(df)

    st.subheader("Estadísticas descriptivas")
    st.write(df.describe())

except FileNotFoundError:
    st.error(f"No se encontró el archivo {excel_path}. Verifica la ruta.")
except Exception as e:
    st.error(f"Ocurrió un error al leer el Excel: {e}")




st.sidebar.title("Navegación")
opcion = st.sidebar.radio(
    "Selecciona el tipo de regresión:",
    ["Regresión Lineal", "Regresión Múltiple", "Regresión Polinomial"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Front")


def mostrar_titulo_y_enunciado(titulo):
    st.header(titulo)
    st.subheader("Enunciado del Problema")
    st.write("""
        *TO-DO*
    """)
    st.markdown("---")


def mostrar_estadisticas():
    st.subheader("📊 Estadísticas del Modelo")
    col1, col2, col3 = st.columns(3)
    col1.metric("R²", "—")
    col2.metric("MAE", "—")
    col3.metric("MSE/RMSE", "—")
    st.markdown("*Metricas*")
    st.markdown("---")


def mostrar_graficas():
    st.subheader("📈 Gráficas del Análisis")

    col1, col2 = st.columns(2)

    fig_placeholder = go.Figure()
    fig_placeholder.add_annotation(
        text="Placeholder de gráfica",
        showarrow=False,
        font=dict(size=18)
    )
    fig_placeholder.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=350
    )

    col1.plotly_chart(fig_placeholder, width="stretch", key="grafica_placeholder_1")
    col2.plotly_chart(fig_placeholder, width="stretch", key="grafica_placeholder_2")

    st.markdown("---")


def mostrar_conclusiones():
    st.subheader("📝 Conclusiones")
    st.write("""
        *TO-DO*
    """)
    st.markdown("---")



if opcion == "Regresión Lineal":
    mostrar_titulo_y_enunciado("Regresión Lineal")
    mostrar_estadisticas()
    mostrar_graficas()
    mostrar_conclusiones()

elif opcion == "Regresión Múltiple":
    mostrar_titulo_y_enunciado("Regresión Múltiple")
    mostrar_estadisticas()
    mostrar_graficas()
    mostrar_conclusiones()

elif opcion == "Regresión Polinomial":
    mostrar_titulo_y_enunciado("Regresión Polinomial (Regresión Compleja)")
    mostrar_estadisticas()
    mostrar_graficas()
    mostrar_conclusiones()
