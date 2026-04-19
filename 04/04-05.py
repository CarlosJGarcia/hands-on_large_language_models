# Text Classification with Generative Models
# Reinach 19/Apr/2026

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load our data
print()
print("Loading Hugging Face - Rotten Tomatoes dataset")
print("----------------------------------------------")
data = load_dataset("rotten_tomatoes")
print(data)
print()

model_name = "google/flan-t5-small"

# Load our model - Este es el código del libro, pero ya no funciona por un cambio en la librería transformers
# pipe = pipeline("text2text-generation", model="google/flan-t5-small", device=0)


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda:0")

# Prepare input
input_text = "Translate to French: Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")

# Generate
print("Prueba")
print("------")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print()

# Prepare our data
prompt = "Is the following sentence positive or negative? "
data = data.map(lambda example: {"t5": prompt + example['text']})
print("Data")
print("------")
print(data)
print()