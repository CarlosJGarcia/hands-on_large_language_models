from PIL import Image
from rich.console import Console
from urllib.request import urlopen
from sentence_transformers import SentenceTransformer, util

# Modelo: SBERT-compatible CLIP
MODEL_ID = "clip-ViT-B-32"

# Load an AI-generated image of a puppy playing in the snow
console = Console()
console.print(f"\nCargando imagen", style="gold1")
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
image = Image.open(urlopen(puppy_path)).convert("RGB")

caption = "a puppy playing in the snow"
print(f"Caption: {caption}")

# Main model for generating text embeddings and image embeddings
console.print(f"\nCargando el modelo (en CPU)", style="gold1")
model = SentenceTransformer(MODEL_ID)



# Encode the caption
console.print(f"\nProcessing text", style="gold1")
text_embedding = model.encode(caption)
print(f"text_embedding.shape: {text_embedding.shape}")



# Encode the image
console.print(f"\nProcessing image", style="gold1")
image_embedding = model.encode(image)
print(f"image_embedding.shape: {image_embedding.shape}")


#Compute cosine similarities
console.print(f"\nSimilarity score", style="gold1")
sim_matrix = util.cos_sim(image_embedding, text_embedding)
print(f"score: {sim_matrix}")

# Fin
print()