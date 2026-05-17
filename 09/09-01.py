import torch
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image
from urllib.request import urlopen
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel

MODEL_ID = "openai/clip-vit-base-patch32"

# Load an AI-generated image of a puppy playing in the snow
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
image = Image.open(urlopen(puppy_path)).convert("RGB")

caption = "a puppy playing in the snow"
print(caption)


# Load a tokenizer to preprocess the text
clip_tokenizer = CLIPTokenizerFast.from_pretrained(MODEL_ID)

# Load a processor to preprocess the images
clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)

# Main model for generating text and image embeddings
model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True)
                                  
# Tokenize our input
inputs = clip_tokenizer(caption, return_tensors="pt")
print(f"Caption tokens: {inputs}")

# Convert our input back to tokens
caption_back = clip_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print(f"Caption converted back: {caption_back}")

# Create a text embedding
text_output = model.get_text_features(**inputs)
text_embedding = text_output.pooler_output
print(f"text_embedding.shape: {text_embedding.shape}")

# Preprocess image
processed_image = clip_processor(text=None, images=image, return_tensors="pt")["pixel_values"]
print(f"processed_image.shape: {processed_image.shape}")

# Prepare image for visualization
img = processed_image.squeeze(0)
img = img.permute(*torch.arange(img.ndim - 1, -1, -1))
img = np.einsum("ijk->jik", img)

# Visualize preprocessed image
plt.imshow(img)
plt.axis("off")