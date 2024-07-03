import os
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# Nombre del modelo base y del modelo fine-tuneado
model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model = "llama-2-7b-reglamento"

# Mapa de dispositivos (modificar según tus necesidades, por ejemplo, 'cpu' o 'cuda')
device_map = {"": "cuda" if torch.cuda.is_available() else "cpu"}

# Cargar el modelo base con las opciones especificadas
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)

# Cargar el modelo con los pesos de LoRA y fusionarlo
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Recargar el tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Inicializar el historial de la conversación
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Definir función para generar respuestas utilizando el pipeline
def chat_with_model(prompt, model, tokenizer, max_length=100):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_length)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    return result[0]['generated_text']

# Crear la interfaz de usuario con Streamlit
st.title("ALOHA VIRTUAL")
st.subheader("Consultas reglamento de docencia")

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
