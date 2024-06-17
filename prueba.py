import streamlit as st
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

# Configuración del modelo ajustado
new_model = "models/llama-2-7b-reglamento"

# Cargar el modelo y el tokenizador
@st.cache_resource
def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(new_model)

# Ignorar advertencias
logging.getLogger("transformers").setLevel(logging.CRITICAL)

def chat_with_model(prompt, max_length=100):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_length)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    return result[0]['generated_text']

# Título de la aplicación
st.title("Chatbot Virtual con Llama 2")

# Entrada de texto
prompt = st.text_input("Introduce tu prompt:")

# Botón para generar texto
if st.button("Enviar"):
    if prompt:
        response = chat_with_model(prompt)
        st.text_area("Respuesta del Chatbot:", value=response, height=300)
    else:
        st.error("Por favor, introduce un prompt.")
