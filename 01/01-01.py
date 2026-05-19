# Librería transformers de Hugging Face
# Basel 10/Apr/2026

import transformers
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig

# Modelo:Microsoft Phi-3-mini 3.8B (billion) parameters, 8 GB VRAM. Text-only, no multimodal. Requires prompt template.
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

console = Console()
console.print(f"\nTransformers", style="gold1")
print(f"transformers:     {transformers.__version__}")


# Load model Phi-3-mini and tokenizer in the NVIDIA GPU
# IMP: La versión actual de la librería "transformers" ya no soporta el parámetro "trust_remote_code=True" que viene en el libro
console.print(f"\nCargando el modelo (en CPU)", style="gold1")
print(f"Loading model {MODEL_ID} in the GPU")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda", torch_dtype="auto")

console.print(f"\nCargando tokenizer", style="gold1")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Create a pipeline (generator)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# Define the generation configuration
gen_config = GenerationConfig(max_new_tokens=500, do_sample=False, num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
)

# Prompit (user input / query)
console.print(f"\nPrompting", style="gold1")
messages = [{"role": "user", "content": "Résumez Alice au pays des merveilles en deux paragraphes."}]
output = generator(messages, generation_config=gen_config)
print(output[0]["generated_text"])
print()

console.print(f"\nPrompting", style="gold1")
messages = [{"role": "user", "content": "Generate a training plan, 4 weeks long, to run 5 Kilometers"}]
output = generator(messages, generation_config=gen_config)
print(output[0]["generated_text"])
print()

console.print(f"\nPrompting", style="gold1")
messages = [{"role": "user", "content": "Dime la receta del pollo al horno"}]
output = generator(messages, generation_config=gen_config)
print(output[0]["generated_text"])
print()

console.print(f"\nPrompting", style="gold1")
messages = [{"role": "user", "content": "An wie vielen Städten kommt der Rhein vorbei?"}]
output = generator(messages, generation_config=gen_config)
print(output[0]["generated_text"])
print()
