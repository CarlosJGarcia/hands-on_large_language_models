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


# Prompt components
persona = "You are an expert in Large Language models. You excel at breaking down complex papers into digestible summaries.\n"
instruction = "Summarize the key findings of the paper provided.\n"
context = "Your summary should extract the most crucial points that can help researchers quickly understand the most vital information of the paper.\n"
data_format = "Create a bullet-point summary that outlines the method. Follow this up with a concise paragraph that encapsulates the main results.\n"
audience = "The summary is designed for busy researchers that quickly need to grasp the newest trends in Large Language Models.\n"
tone = "The tone should be professional and clear.\n"
text = "All pre-trained models are identified by a model ID. When you create a tokenizer that a pre-trained model requires, it will check with the pre-trained model’s config to instantiate the correct tokenizer object, similarly, for the model. Therefore, you just need to use AutoTokenizer and AutoModel instead of the specific classes, such as BertTokenizer and BertModel. Knowing how a transformer model usually works, you should expect the core model to take the input tokens and output logit tensors. Therefore, you used argmax above to convert the logits to token IDs and convert the IDs to strings using the tokenizer’s decode method. However, you must provide the access token if you want to use a gated model with the above code. The way to set up the access token is to use some environment variables. You can find all environment variables that matter to the transformers library in the documentation; the most important ones are:"
data = f"Text to summarize: {text}"

# The full prompt - remove and add pieces to view its impact on the generated output
query = persona + instruction + context + data_format + audience + tone + data

print(f"\n{query}")


# 4. PREVENT EXIT: Wait for user signal
input(f"\nPress ENTER to close the script and clear GPU memory")