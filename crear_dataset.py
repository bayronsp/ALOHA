import pandas as pd
import json

# Cargar el archivo de Excel
file_path = 'preguntas_y_respuestas.xlsx'  # Reemplaza con la ruta de tu archivo Excel
df = pd.read_excel(file_path)

# Crear el diccionario con las preguntas y respuestas
data = {
    "Pregunta": df["Pregunta"].tolist(),
    "Respuesta": df["Respuesta"].tolist()
}

# Crear una lista de diccionarios con la nueva estructura deseada
qa_list = [{"text": f'<s>[INST] {pregunta.strip()} [/INST] {respuesta.strip()} </s>'} for pregunta, respuesta in zip(data["Pregunta"], data["Respuesta"])]

# Escribir la lista en un archivo JSONL
with open('preguntas_respuestas_transformadas.jsonl', 'w', encoding='utf-8') as f:
    for qa in qa_list:
        json.dump(qa, f, ensure_ascii=False)
        f.write('\n')