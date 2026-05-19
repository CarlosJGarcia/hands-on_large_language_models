import torch
from PIL import Image
from rich.console import Console
from urllib.request import urlopen
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# Modelo: BLIP-2. Es un modelo multimodal
MODEL_ID = "Salesforce/blip2-opt-2.7b"

# Load image of a supercar
console = Console()
console.print(f"\nCargando imagen", style="gold1")
car_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/car.png"
image = Image.open(urlopen(car_path)).convert("RGB")

# Load processor and main model
console.print(f"\nTransformers", style="gold1")
print(f"Loading model {MODEL_ID} in the GPU")
blip_processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Blip2ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16)

# Send the model to GPU to speed up inference
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# TEXT
# ====
console.print(f"\nProcessing text", style="gold1")
print(blip_processor.tokenizer)

# Preprocess the text
text = "Her vocalization was remarkably melodic"
token_ids = blip_processor(image, text=text, return_tensors="pt")
token_ids = token_ids.to(device, torch.float16)["input_ids"][0]

# Convert input ids back to tokens
tokens = blip_processor.tokenizer.convert_ids_to_tokens(token_ids)
print(f"tokens: {tokens}")

# Replace the space token with an underscore
tokens = [token.replace("Ġ", "_") for token in tokens]
print(f"tokens: {tokens}")


# IMAGE
# =====

# Preprocess the image
console.print(f"\nProcessing image", style="gold1")
inputs = blip_processor(image, return_tensors="pt").to(device, torch.float16)
print(f"processed_image pixel_values shape: {inputs['pixel_values'].shape}")



# Fin
print()