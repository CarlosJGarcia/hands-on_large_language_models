import torch
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
MAX_NEW_TOKENS = 500


# Load Model and Tokenizer using transformers library
# trust_remote_code=False + attn_implementation="eager" para evitar warnings
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda", torch_dtype="auto", trust_remote_code=False, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Define a clean Generation Config to avoid warnings being displayed
gen_cfg = GenerationConfig.from_pretrained(MODEL_ID)
gen_cfg.do_sample = False
gen_cfg.temperature = None
gen_cfg.max_new_tokens = MAX_NEW_TOKENS
model.generation_config.max_length = 4096

# Create the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

print("\nModel loaded successfully!")


# Get user input from the terminal
user_prompt = input("\nUser prompt: ")
user_prompt = [{"role": "user", "content": user_prompt}]
prompt = tokenizer.apply_chat_template(user_prompt, tokenize=False, add_generation_prompt=True)
print(f"Prompt: {prompt}")

# Run inference
print("\nPhi-3 thinking...")
output = pipe(prompt)

# Print result
print(f"\nGenerated reply: {output[0]['generated_text']}")

# 4. PREVENT EXIT: Wait for user signal
input("Press ENTER to close the script and clear GPU memory")