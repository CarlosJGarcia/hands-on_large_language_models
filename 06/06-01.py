import torch
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "microsoft/Phi-3-mini-4k-instruct"

# 1. Load with native transformers implementation (trust_remote_code=False)
# 2. Use attn_implementation="eager" to bypass Flash Attention requirements
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype="auto", trust_remote_code=False, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Define a clean Generation Config
gen_cfg = GenerationConfig.from_pretrained(model_id)
gen_cfg.do_sample = False
gen_cfg.max_new_tokens = 500
gen_cfg.temperature = None
model.generation_config = gen_cfg

# Create the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# Test it
print("\nModel loaded successfully!")