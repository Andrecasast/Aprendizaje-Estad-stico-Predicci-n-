import streamlit as st
import joblib
import numpy as np

st.title("Predicción de Rendimiento Académico")

model = joblib.load("modelo_rendimiento(1).pkl")

horas = st.number_input("Horas promedio de uso diario")
suenio = st.number_input("Horas de sueño")
salud = st.number_input("Puntaje de salud mental")
conflictos = st.number_input("Conflictos por redes sociales")
adiccion = st.number_input("Puntaje de adicción")

if st.button("Predecir"):
    X = np.array([horas, suenio, salud, conflictos, adiccion]).reshape(1, -1)
    pred = model.predict(X)[0]
    st.write("Resultado:", pred)

