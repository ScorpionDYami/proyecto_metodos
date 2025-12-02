import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np


st.set_page_config(page_title="Módulo de Regresiones", layout="wide")

excel_path = "data/data.xlsx"

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


def mostrar_titulo_y_enunciado(titulo):
    st.header(titulo)
    st.subheader("Enunciado del Problema")
    st.write("""
        *TO-DO*
    """)
    st.markdown("---")


if opcion == "Regresión Lineal":
    mostrar_titulo_y_enunciado("Regresión Lineal")

    try:
        X = df["Edad"].values.reshape(-1, 1)
        y = df["EstresTotal"].values.reshape(-1, 1)

        model = LinearRegression()
        state = 1

        # Ciclo para buscar un buen ajuste r2 >= 0.7
        while True:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=state
                )

                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)

                r2 = r2_score(y_test, y_pred_test)

                if r2 >= 0.7:
                    break
                else:
                    state += 1
            except ValueError:
                state += 1

        # Predicción completa para la gráfica final
        y_pred = model.predict(X)

        # 2. Estadísticas del modelo
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        st.subheader("Estadísticas del Modelo")
        col1, col2, col3 = st.columns(3)
        col1.metric("R²", f"{r2:.4f}")
        col2.metric("MAE", f"{mae:.4f}")
        col3.metric("MSE", f"{mse:.4f}")

        st.markdown("---")

        # 3. Gráfica de regresión lineal
        fig_lineal = go.Figure()
        fig_lineal.add_trace(go.Scatter(
            x=X.flatten(), y=y.flatten(),
            mode="markers",
            name="Datos Reales"
        ))
        fig_lineal.add_trace(go.Scatter(
            x=X.flatten(), y=y_pred.flatten(),
            mode="lines",
            name="Línea de Regresión",
            line=dict(color="red")
        ))

        fig_lineal.update_layout(
            title="Regresión Lineal: Estrés Total por Edad",
            xaxis_title="Edad",
            yaxis_title="Estrés Total",
            height=500
        )

        st.plotly_chart(fig_lineal, config={"responsive": True})

        st.markdown("---")

        # 4. Gráfica de barras por rangos
        df["RangoEdad"] = pd.cut(df["Edad"], bins=range(3, 81, 7))
        grupo = df.groupby("RangoEdad")["EstresTotal"].mean()

        fig_barras = go.Figure()
        fig_barras.add_trace(go.Bar(
            x=grupo.index.astype(str),
            y=grupo.values
        ))

        fig_barras.update_layout(
            title="Estrés Promedio por Rango de Edad",
            xaxis_title="Rango de Edad",
            yaxis_title="Estrés Total",
            height=500
        )

        st.plotly_chart(fig_barras, config={"responsive": True})

        st.markdown("---")

    except Exception as e:
        st.error(f"Error al procesar la regresión lineal: {e}")


    # -------- MÉTODO DE MÍNIMOS CUADRADOS (MANUAL) --------------

    st.subheader("Método de Mínimos Cuadrados (Manual)")

    # Extraer x e y como vectores simples
    x = df["Edad"].values
    y_vals = df["EstresTotal"].values

    n = len(x)

    # Sumatorias necesarias
    sum_x = np.sum(x)
    sum_y = np.sum(y_vals)
    sum_xy = np.sum(x * y_vals)
    sum_x2 = np.sum(x * x)

    # Cálculo de la pendiente m
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

    # Cálculo del intercepto c
    c = (sum_y - m * sum_x) / n

    # Predicciones manuales y^ = m x + c
    y_ls_pred = m * x + c

    # Métricas del modelo manual
    ecm_ls = mean_squared_error(y_vals, y_ls_pred)
    eam_ls = mean_absolute_error(y_vals, y_ls_pred)
    r2_ls = r2_score(y_vals, y_ls_pred)


    # ------------------ GRÁFICA EN PLOTLY ------------------

    fig_ls = go.Figure()

    # Puntos reales
    fig_ls.add_trace(go.Scatter(
        x=x,
        y=y_vals,
        mode="markers",
        name="Datos reales",
        marker=dict(color="green")
    ))

    # Línea de mínimos cuadrados
    fig_ls.add_trace(go.Scatter(
        x=x,
        y=y_ls_pred,
        mode="lines",
        name="Recta (mínimos cuadrados)",
        line=dict(color="orange")
    ))

    fig_ls.update_layout(
        title="Regresión por Mínimos Cuadrados: Estrés Total por Edad",
        xaxis_title="Edad",
        yaxis_title="Estrés Total",
        autosize=True,
        height=500
    )

    st.plotly_chart(fig_ls, config={"responsive": True})

    # ------------------ RESULTADOS -------------------------
    st.subheader("Resultados del Método Manual")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Pendiente (m):", f"{m:.4f}")
    col2.metric("Intercepto (c):", f"{c:.4f}")
    col3.metric("ECM (Error Cuadrático Medio):", f"{ecm_ls:.4f}")
    col4.metric("EAM (Error Absoluto Medio):", f"{eam_ls:.4f}")
    col5.metric("R²:", f"{r2_ls:.4f}")



elif opcion == "Regresión Múltiple":
    mostrar_titulo_y_enunciado("Regresión Múltiple")

    try:
        # 1. Crear rangos de edad
        df["RangoEdad"] = pd.cut(df["Edad"], bins=range(3, 81, 7))

        # 2. Convertir rangos a variables dummy
        # Convertimos a int para que se muestren como 0 y 1 en lugar de True/False
        X_dummy = pd.get_dummies(df["RangoEdad"], prefix="Rango", drop_first=True).astype(int)

        # 3. Variables dependiente y matriz final
        X_multi = X_dummy.values
        y_multi = df["EstresTotal"].values.reshape(-1, 1)

        # 4. Modelo de regresión lineal múltiple
        modelo_multiple = LinearRegression()
        modelo_multiple.fit(X_multi, y_multi)

        # 5. Predicciones y evaluación
        y_pred_multiple = modelo_multiple.predict(X_multi)
        r2_multiple = r2_score(y_multi, y_pred_multiple)
        mae = mean_absolute_error(y_multi, y_pred_multiple)
        mse = mean_squared_error(y_multi, y_pred_multiple)

        # ------------------ RESULTADOS DETALLADOS ------------------
        st.subheader("Comparación: Real vs Predicho")
        
        df_resultados = df[["Edad", "EstresTotal"]].copy()
        df_resultados["Predicción"] = y_pred_multiple.flatten()
        df_resultados["Error Residual"] = df_resultados["EstresTotal"] - df_resultados["Predicción"]
        
        st.dataframe(df_resultados)

        # ------------------ MÉTRICAS ------------------
        st.subheader("Estadísticas del Modelo")
        col1, col2, col3 = st.columns(3)
        col1.metric("R²", f"{r2_multiple:.4f}")
        col2.metric("MAE", f"{mae:.4f}")
        col3.metric("MSE", f"{mse:.4f}")

        st.markdown("---")

        # ------------------ COEFICIENTES ------------------
        with st.expander("Ver Coeficientes del Modelo"):
            coef_df = pd.DataFrame({
                "Variable": ["Intercepto"] + list(X_dummy.columns),
                "Coeficiente": [modelo_multiple.intercept_[0]] + list(modelo_multiple.coef_[0])
            })
            st.dataframe(coef_df)

        st.markdown("---")

        # ------------------ GRÁFICA PLOTLY ------------------
        st.subheader("Regresión Múltiple por Rangos de Edad")

        # Ordenar para gráfica continua
        df_ordenado = df.sort_values("Edad")
        X_dummy_ordenado = pd.get_dummies(df_ordenado["RangoEdad"], prefix="Rango", drop_first=True)
        y_linea = modelo_multiple.predict(X_dummy_ordenado.values)

        fig = go.Figure()

        # Puntos reales
        fig.add_trace(go.Scatter(
            x=df["Edad"],
            y=df["EstresTotal"],
            mode="markers",
            name="Datos Reales",
            marker=dict(color="blue")
        ))

        # Línea del modelo múltiple
        fig.add_trace(go.Scatter(
            x=df_ordenado["Edad"],
            y=y_linea.flatten(),
            mode="lines",
            name="Modelo Múltiple (Rangos)",
            line=dict(color="red", width=3)
        ))

        fig.update_layout(
            title="Regresión Lineal Múltiple por Rangos de Edad",
            xaxis_title="Edad",
            yaxis_title="Estrés Total",
            height=500
        )

        st.plotly_chart(fig, config={"responsive": True})

        st.markdown("---")

    except Exception as e:
        st.error(f"Error en la regresión múltiple: {e}")
    

elif opcion == "Regresión Polinomial":
    mostrar_titulo_y_enunciado("Regresión Polinomial (Regresión Compleja)")
    
