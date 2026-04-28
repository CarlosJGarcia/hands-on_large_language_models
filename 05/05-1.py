# Reinach, 28/Apr/2026
from umap import UMAP
from rich.console import Console
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Load dataset from Hugging Face
DATASET_ID = "maartengr/arxiv_nlp"
dataset = load_dataset(DATASET_ID)["train"]

# Extract metadata
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

console = Console()
console.print(f"\nTrain dataset successfully loaded.", style="gold1")
print(f"Abstracts: {abstracts}\n")
print(f"Titles: {titles}\n")


# Create an embedding for each abstract
MODEL_ID = "thenlper/gte-small"
embedding_model = SentenceTransformer(MODEL_ID)
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)

# Check the dimensions of the resulting embeddings
console.print(f"\nDimensions of the resulting embeddings.", style="gold1")
print(f"Shape: {embeddings.shape}")

# We reduce the input embeddings from 384 dimensions to 5 dimensions
umap_model = UMAP(n_components=5, min_dist=0.0, metric='cosine', random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings)