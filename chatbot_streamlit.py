import os
import streamlit as st
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Ruta al modelo ajustado
model_dir = "models/llama-2-7b-reglamento"

# Cargar el modelo y el tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Inicializar el historial de la conversación
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Definir función para generar respuestas utilizando el pipeline
def chat_with_model(prompt, model, tokenizer, max_length=100):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_length)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    return result[0]['generated_text']

# Crear la interfaz de usuario con Streamlit
st.title("Chatbot de Reglamento")
st.subheader("Interfaz similar a WhatsApp")

# Mostrar el historial de la conversación
for message in st.session_state['conversation']:
    st.write(message)

# Entrada de texto para el usuario
user_input = st.text_input("Escribe tu mensaje aquí...")

# Botón para enviar el mensaje
if st.button("Enviar"):
    if user_input:
        st.session_state['conversation'].append(f"Tú: {user_input}")
        response = chat_with_model(user_input, model, tokenizer)
        st.session_state['conversation'].append(f"Bot: {response}")

# Botón para limpiar la conversación
if st.button("Limpiar conversación"):
    st.session_state['conversation'] = []
