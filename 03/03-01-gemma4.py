# Modelo: Gemma-4
# Parámetros 4 B
# Quantization: 4 Bit (necesario para RTX 3060 12 GB, puede que no necesario en 5060 16 GB)
# Memoria: 6 GB for weights + 1 GB for context/system overhead = 7 GB
# Uno de los modelos más potentes que se pueden cargar en la RTX 3060 12 GB. El uso de quantization disminuye la precisión en un 2% luego es buen compromiso

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, BitsAndBytesConfig

# 1. We HAVE to use 4-bit quantization because the model is 16GB natively
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)


# 1. Model Selection
# Gemma 4 E4B is designed to punch like a 10B model but fits in 12GB VRAM natively
model_id = "google/gemma-4-e4b-it"

print(f"Loading model: {model_id} natively (unquantized)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=quant_config, # Optimized for RTX 30-series
    device_map="cuda"
)

print("Loading tokenizer.")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Setup GenerationConfig
print("Creating GenerationConfig.")
clean_config = GenerationConfig.from_model_config(model.config)
clean_config.max_new_tokens = 2048 
clean_config.max_length = None 
clean_config.pad_token_id = tokenizer.eos_token_id

# 3. Create the pipeline
print("Creating pipeline.")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# --- PROMPT 1: The Apology ---
# Gemma 4 uses <start_of_turn> and <end_of_turn> tags
raw_prompt_1 = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
prompt_1 = f"<start_of_turn>user\n{raw_prompt_1}<end_of_turn>\n<start_of_turn>model\n"

print("Prompt: ", prompt_1)
output = generator(prompt_1, generation_config=clean_config)
print("Output: ", output[0]['generated_text'])
print("-" * 30)

# --- PROMPT 2: LLM Development ---
raw_prompt_2 = "Describe in a paragraph the best way to get started in LLM development. Estimate how much time is needed to do that."
prompt_2 = f"<start_of_turn>user\n{raw_prompt_2}<end_of_turn>\n<start_of_turn>model\n"

print("Prompt: ", prompt_2)
output = generator(prompt_2, generation_config=clean_config)
print("Output: ", output[0]['generated_text'])