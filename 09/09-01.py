import torch
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console

from PIL import Image
from urllib.request import urlopen
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel

# Modelo: OpenAI CLIP. Es un modelo multimodal, para uso académico sobre visión artificial. Es multimodal
MODEL_ID = "openai/clip-vit-base-patch32"

# Load an AI-generated image of a puppy playing in the snow
console = Console()
console.print(f"\nCargando imagen", style="gold1")
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
image = Image.open(urlopen(puppy_path)).convert("RGB")

caption = "a puppy playing in the snow"
print(f"Caption: {caption}")


# Load a tokenizer to preprocess the text
console.print(f"\nCargando tokenizer", style="gold1")
clip_tokenizer = CLIPTokenizerFast.from_pretrained(MODEL_ID)

# Load a processor to preprocess the image
console.print(f"\nCargando image processor", style="gold1")
clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)

# Main model for generating text embeddings and image embeddings
console.print(f"\nCargando el modelo (en CPU)", style="gold1")
model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True)


# TEXT
# ====

# Tokenize the text
console.print(f"\nProcessing text", style="gold1")
inputs = clip_tokenizer(caption, return_tensors="pt")
print(f"\nCaption tokens: {inputs}")

# Convert the tokens back to text
caption_back = clip_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print(f"Caption converted back: {caption_back}")

# Create a text embedding
text_output = model.get_text_features(**inputs)
text_embedding = text_output.pooler_output
print(f"text_embedding.shape: {text_embedding.shape}")


# IMAGE
# =====

# Preprocess image
console.print(f"\nProcessing image", style="gold1")
processed_image = clip_processor(text=None, images=image, return_tensors="pt")["pixel_values"]
print(f"processed_image.shape: {processed_image.shape}")

# Prepare image for visualization
img = processed_image.squeeze(0)
img = img.permute(*torch.arange(img.ndim - 1, -1, -1))
img = np.einsum("ijk->jik", img)

# Clip the values between 0 and 1 to prevent a matplotlib warning
img = np.clip(img, 0, 1)

# Visualize the preprocessed image using matplotlib
plt.imshow(img)
plt.axis("off")
plt.show()

# Create the image embedding
image_output = model.get_image_features(pixel_values=processed_image)
image_embedding = image_output.pooler_output 
print(f"image_embedding.shape: {image_embedding.shape}")


# COMPARAR TEXT EMBEDDING CON IMAGE EMBEDDING
# ===========================================

# Normalize the embeddings
text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

# Calculate their similarity
text_embedding = text_embedding.detach().cpu().numpy()
image_embedding = image_embedding.detach().cpu().numpy()
score = np.dot(text_embedding, image_embedding.T)

console.print(f"\nSimilarity score", style="gold1")
print(f"score: {score}")



# Fin
print()