from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Load MNLI dataset from GLUE
# 0 = entailment, 1 = neutral, 2 = contradiction
train_dataset = load_dataset("glue", "mnli", split="train").select(range(50_000))
train_dataset = train_dataset.remove_columns("idx")

# Check one example
print(f"\n{train_dataset[2]}\n")

# Use a base model
embedding_model = SentenceTransformer('bert-base-uncased')