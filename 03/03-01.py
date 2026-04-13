# Modelo: Phi-3-Mini-4K-Instruct
# Parámetros 3.8 B
# Memoria: 7.6 GB for weights + 2 GB for context/system overhead = 9.6 GB
# El modelo más potente que se puede cargar en la RTX 3060 12 GB sin usar quantization

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

# Load model Phi-3-mini and tokenizer in the NVIDIA GPU
# IMP: La versión actual de la librería "transformers" ya no soporta el parámetro "trust_remote_code=True" que viene en el libro
print("Loading model.")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="cuda", torch_dtype="auto")

print("Loading tokenizer.")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Create a clean generation config to prevent a warning. Override the 'max_length=20' hidden in the model files
print("Creating GenerationConfig.")
clean_config = GenerationConfig.from_model_config(model.config)
clean_config.max_new_tokens = 3072  
clean_config.max_length = None # Remove the warning


# Create a pipeline (generator)
# IMP: Quito dos de los parámetros "max_new_tokens=50" y "do_sample=False" que vienen en el libro ya que se han eliminado de la librería
print("Creating pipeline.")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# Test Promp (user input / query)
print("Test Prompt #1")
print("-----------------")
prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
prompt = "<|user|>\n" + prompt + "<|end|>\n<|assistant|>"
print("Prompt #1: ", prompt)
print()

output = generator(prompt, generation_config=clean_config)
print("Output #1: ", output[0]['generated_text'])
print()

# Test prompt (user input / query)
print("Test Prompt #2")
print("-----------------")
prompt = "Describe in a paragraph the best way to get started in LLM development. Estimate how much time is needed to do that."
prompt = "<|user|>\n" + prompt + "<|end|>\n<|assistant|>"
print("Prompt #2: ", prompt)
print()

output = generator(prompt, generation_config=clean_config)
print("Output #2: ", output[0]['generated_text'])
print()

print("Print the model variable to see the order or the layers")
print("-------------------------------------------------------")
print(model)
print()

print("Decoding strategy")
print("-----------------")
prompt = "The capital of France is"

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to("cuda")

# Get the output of the model before the lm_head. This is the internal mathematical representation of the text.
model_output = model.model(input_ids)

# Get the output of the lm_head. These are the logits (the raw scores for every possible word in the model's vocabulary).
lm_head_output = model.lm_head(model_output[0])

# This is the logic that picks the single highest score to find the "best" next word.
token_id = lm_head_output[0,-1].argmax(-1)
output = tokenizer.decode(token_id)
print("Prompt #3            : ", prompt)
print("Output #3            : ", output)
print("model_output[0].shape: ", model_output[0].shape)
print("lm_head_output.shape : ", lm_head_output.shape)
print()

print("Cache strategy")
print("--------------")

# Tokenize an input prompt. Para esta prueba, ignoro la 'attention mask' lo que genera un warning "The attention mask is not set and cannot be inferred from input because pad token is same as eos token."
prompt = "Write a very long email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to("cuda")

# Ensure all previous GPU tasks are done, start timer, do task
torch.cuda.synchronize()
start = time.perf_counter()
generation_output = model.generate(input_ids=input_ids, max_new_tokens=100, use_cache=True)

# Wait for the model to actually finish before stopping the clock, stop timer and show the time
torch.cuda.synchronize()
end = time.perf_counter()
tiempo_ena = end - start
print(f"GPU generation time, cache enabled : {tiempo_ena:.4f} seconds")

# Ensure all previous GPU tasks are done, start timer, do task
torch.cuda.synchronize()
start = time.perf_counter()
generation_output = model.generate(input_ids=input_ids, max_new_tokens=100, use_cache=False)

# Wait for the model to actually finish before stopping the clock, stop timer and show the time
torch.cuda.synchronize()
end = time.perf_counter()
tiempo_dis = end - start
print(f"GPU generation time, cache disabled: {tiempo_dis:.4f} seconds")
print(f"Diferencia                         : {tiempo_dis - tiempo_ena:.4f} seconds")
print()
