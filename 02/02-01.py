# Librería transformers de Hugging Face
# Basel 10/Apr/2026

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"Transformers version:     {transformers.__version__}")

# Load model Phi-3-mini and tokenizer in the NVIDIA GPU
# IMP: La versión actual de la librería "transformers" ya no soporta el parámetro "trust_remote_code=True" que viene en el libro
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="cuda", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Set the padding token. If the tokenizer doesn't have a pad_token, we assign the eos_token to it. This is required by Phi-3-mini
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prompt (user input / query)
prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>"

# Tokenize the input prompt Generate the attention_mask during tokenization
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

# Pull out the ID's and the Mask
my_input_ids = inputs["input_ids"]
my_attention_mask = inputs["attention_mask"]

# Generate the text. Pass both input_ids and attention_mask using **inputs
generation_output = model.generate(input_ids=my_input_ids, attention_mask=my_attention_mask, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)

# Decode and print
decoded_output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
print(decoded_output)

print(my_input_ids)

for id in my_input_ids[0]:
   print(tokenizer.decode(id))
print()

print(tokenizer.decode(3323))
print(tokenizer.decode(622))
print(tokenizer.decode([3323, 622]))
print(tokenizer.decode(29901))
print()

print("Fin del programa.")
