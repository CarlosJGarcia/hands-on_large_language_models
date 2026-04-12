# Modelo: Mistral-Nemo
# Parámetros 12 B
# Quantization: 4 Bit
# Memoria: 7.5 GB for weights + 1 GB for context/system overhead = 9 GB
# Uno de los modelos más potentes que se pueden cargar en la RTX 3060 12 GB. El uso de quantization disminuye la precisión en un 2% luego es buen compromiso

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, BitsAndBytesConfig

# 1. Configuration for 4-bit quantization (Essential for 12GB VRAM)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # Faster on RTX 30-series
    bnb_4bit_quant_type="nf4"
)

# 2. Load Mistral Nemo 12B
model_id = "mistralai/Mistral-Nemo-Instruct-2407"

print(f"Loading model: {model_id} in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=quant_config, 
    device_map="cuda"
)

print("Loading tokenizer.")
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-Nemo-Instruct-2407", 
    use_fast=True,           # Ensures we use the Rust-based fast tokenizer
    fix_mistral_regex=False   # Avoids the bug in the transformers library
)

# 3. Setup GenerationConfig
print("Creating GenerationConfig.")
clean_config = GenerationConfig.from_model_config(model.config)
clean_config.max_new_tokens = 2048 # Adjusted for memory safety
clean_config.max_length = None 
clean_config.pad_token_id = tokenizer.eos_token_id

# 4. Create the pipeline
print("Creating pipeline.")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# --- PROMPT 1: The Apology ---
# Mistral uses [INST] and [/INST] instead of Phi-3's tags
raw_prompt_1 = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
prompt_1 = f"[INST] {raw_prompt_1} [/INST]"

print("Prompt: ", prompt_1)
output = generator(prompt_1, generation_config=clean_config)
print("Output: ", output[0]['generated_text'])
print("-" * 30)

# --- PROMPT 2: LLM Development ---
raw_prompt_2 = "Describe in a paragraph the best way to get started in LLM development. Estimate how much time is needed to do that."
prompt_2 = f"[INST] {raw_prompt_2} [/INST]"

print("Prompt: ", prompt_2)
output = generator(prompt_2, generation_config=clean_config)
print("Output: ", output[0]['generated_text'])