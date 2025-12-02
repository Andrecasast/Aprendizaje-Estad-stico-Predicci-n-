import streamlit as st
import joblib
import numpy as np

st.title("Predicción del Rendimiento Académico")

st.write("Ingrese los datos para obtener la predicción.")

# Cargar modelo
model = joblib.load("modelo_rendimiento(1).pkl")

# Entradas
horas_uso = st.number_input("Horas de uso diario de redes", 0.0, 24.0, 3.0)
horas_sueno = st.number_input("Horas de sueño", 0.0, 24.0, 7.0)
salud_mental = st.number_input("Puntaje de salud mental (0-100)", 0, 100, 70)
conflictos = st.number_input("Conflictos por redes (0-10)", 0, 10, 0)
adiccion = st.number_input("Puntaje de adicción (0-100)", 0, 100, 30)

if st.button("Predecir"):
    X = np.array([horas_uso, horas_sueno, salud_mental, conflictos, adiccion]).reshape(1, -1)
    pred = model.predict(X)[0]

    resultado = "Afecta al rendimiento académico" if pred == 1 else "NO afecta"
    st.write("### Resultado:", resultado)
