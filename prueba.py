import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configuración del modelo ajustado
new_model = "llama-2-7b-reglamento"

# Cargar el modelo y el tokenizador
model = AutoModelForCausalLM.from_pretrained(new_model)
tokenizer = AutoTokenizer.from_pretrained(new_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Configurar el pipeline de generación de texto
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=100)

# Título de la aplicación
st.title("Generador de Texto con Llama 2")

# Entrada de texto
prompt = st.text_area("Introduce tu prompt:")

# Botón para generar texto
if st.button("Generar texto"):
    if prompt:
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        generated_text = result[0]['generated_text']
        st.text_area("Texto Generado:", value=generated_text, height=300)
    else:
        st.error("Por favor, introduce un prompt.")
