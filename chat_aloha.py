import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Verificar si el archivo existe en el directorio
file_path = 'reglamento.xlsx'

if not os.path.isfile(file_path):
    st.error(f"El archivo {file_path} no se encontró.")
else:
    # Cargar el archivo Excel
    @st.cache
    def cargar_datos():
        return pd.read_excel(file_path, usecols="A:B", names=['Pregunta', 'Respuesta'])

    datos = cargar_datos()

    # Inicializar el vectorizador TF-IDF y ajustar con las preguntas
    @st.cache(allow_output_mutation=True)
    def inicializar_vectorizador():
        vectorizer = TfidfVectorizer().fit(datos['Pregunta'])
        return vectorizer

    vectorizer = inicializar_vectorizador()

    # Inicializar el historial de conversación
    if 'historial' not in st.session_state:
        st.session_state.historial = []

    # Función para obtener la respuesta basada en la pregunta del usuario
    def obtener_respuesta(pregunta):
        pregunta_vector = vectorizer.transform([pregunta])
        pregunta_tfidf = vectorizer.transform(datos['Pregunta'])
        similitudes = cosine_similarity(pregunta_vector, pregunta_tfidf).flatten()
        idx_max = similitudes.argmax()
        if similitudes[idx_max] > 0.1:  # Umbral para considerar una respuesta
            return datos['Respuesta'].iloc[idx_max]
        else:
            return "Lo siento, no encontré una respuesta en el reglamento."

    # Configuración de la interfaz de Streamlit
    st.title("ALOHA Virtual")
    st.write("Bienvenido al Chatbot ALOHA Virtual, ¿en qué puedo ayudarte hoy?")

    # Mostrar el historial de conversación
    for entrada in st.session_state.historial:
        st.write(f"**Tú:** {entrada['pregunta']}")
        st.write(f"**ALOHA:** {entrada['respuesta']}")

    # Entrada del usuario
    pregunta = st.text_input("Escribe tu pregunta:")

    # Botón para enviar la pregunta
    if st.button("Enviar"):
        # Obtener la respuesta
        respuesta = obtener_respuesta(pregunta)
        # Guardar en el historial
        st.session_state.historial.append({"pregunta": pregunta, "respuesta": respuesta})
        # Refrescar la página para mostrar el historial actualizado
        st.experimental_rerun()
