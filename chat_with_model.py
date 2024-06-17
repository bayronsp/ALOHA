import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

# Fine-tuned model name
new_model = "models/llama-2-7b-reglamento"

# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained(new_model)
model.config.use_cache = True

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(new_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

def chat_with_model(prompt, max_length=100):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_length)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    return result[0]['generated_text']

# Example prompt
prompt = "¿Cuáles son los componentes del Plan de Estudios?"
response = chat_with_model(prompt)
print(response)
