# Loads the Wikipedia dataset from Hugging Face.
# Kaiseraugst 01/May/2026

from rich.console import Console
from datasets import load_dataset

DATASET_ID = "wikimedia/wikipedia"
SUBSET = "20231101.en"              # El 'dump' en inglés más actual en 2026. Tiene 6.410.000 artículos (más de 6 millones)


print(f"Loading Wikipedia dataset '{DATASET_ID}', subset='{SUBSET}'...")
        
# El método load_dataset usa "Apache Arrow" así que en realidad los 6M artículos y 16 GB de datos del dataset no están cargados en RAM, aunque python piensa que sí
dataset = load_dataset(DATASET_ID, SUBSET)

console = Console()
console.print(f"Dataset loaded successfully!\n", style="gold1")
print(f"Dataset structure: {dataset}")
print(f"Train dataset size: {dataset['train'].data.nbytes/(1024*1024):.2f} MB\n")

# Inspecting the first item in the 'train' split
console.print(f"Sample data:", style="gold1")
sample = dataset['train'][0] 
for key, value in sample.items():
    content_preview = str(value)[:200].replace('\n', ' ')
    print(f"{key}: {content_preview}...")

print()
key = input("Press ENTER to exit.")
