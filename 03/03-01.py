# Modelo: Phi-3-Mini-4K-Instruct
# Parámetros 3.8 B
# Memoria: 7.6 GB for weights + 2 GB for context/system overhead = 9.6 GB
# El modelo más potente que se puede cargar en la RTX 3060 12 GB sin usar quantization

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

# Load model Phi-3-mini and tokenizer in the NVIDIA GPU
# IMP: La versión actual de la librería "transformers" ya no soporta el parámetro "trust_remote_code=True" que viene en el libro
print("Loading model.")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="cuda", torch_dtype="auto")
print("Loading tokenizer.")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Create a clean generation config to override the 'max_length=20' hidden in the model files
print("Creating GenerationConfig.")
clean_config = GenerationConfig.from_model_config(model.config)
clean_config.max_new_tokens = 3072  
clean_config.max_length = None # Remove a warning


# Create a pipeline (generator)
# IMP: Quito dos de los parámetros "max_new_tokens=50" y "do_sample=False" que vienen en el libro ya que se han eliminado de la librería
print("Creating pipeline.")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# Prompit (user input / query)
prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
prompt = "<|user|>\n" + prompt + "<|end|>\n<|assistant|>"
print("Prompt: ", prompt)
print()

output = generator(prompt, generation_config=clean_config)
print("Output: ", output[0]['generated_text'])
print()

# Prompit (user input / query)
prompt = "Describe in a paragraph the best way to get started in LLM development. Estimate how much time is needed to do that."
prompt = "<|user|>\n" + prompt + "<|end|>\n<|assistant|>"
print("Prompt: ", prompt)
print()

output = generator(prompt, generation_config=clean_config)
print("Output: ", output[0]['generated_text'])
print()

