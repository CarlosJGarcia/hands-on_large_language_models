# Librería transformers de Hugging Face
# Basel 10/Apr/2026

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig


print(f"transformers:     {transformers.__version__}")


# Load model Phi-3-mini and tokenizer in the NVIDIA GPU
# IMP: La versión actual de la librería "transformers" ya no soporta el parámetro "trust_remote_code=True" que viene en el libro
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="cuda", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Create a pipeline (generator)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# Define the generation configuration
gen_config = GenerationConfig(max_new_tokens=500, do_sample=False, num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
)

# Prompt (user input / query)
prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>"

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# Generate the text
generation_output = model.generate(input_ids=input_ids, max_new_tokens=20)

# Print the output
print(tokenizer.decode(generation_output[0]))






"""
messages = [{"role": "user", "content": "Generate a training plan, 4 weeks long, to run 5 Kilometers"}]
output = generator(messages, generation_config=gen_config)
print(output[0]["generated_text"])
"""
