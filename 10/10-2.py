#import mteb
#import warnings
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import losses  
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer


# Load MNLI dataset from GLUE
# 0 = entailment, 1 = neutral, 2 = contradiction
train_dataset = load_dataset("glue", "mnli", split="train").select(range(50_000))
train_dataset = train_dataset.remove_columns("idx")

# Reorder the dataset columns ['premise', 'hypothesis', 'label'] so 'hypothesis' is at index 0, do this:
train_dataset = train_dataset.select_columns(['hypothesis', 'premise', 'label'])

# Check one example
print(f"\nTrain dataset successfully loaded.")
print(f"Example: {train_dataset[2]}\n")

# Create an embedding similarity evaluator for STSB
val_sts = load_dataset("glue", "stsb", split="validation")
evaluator = EmbeddingSimilarityEvaluator(sentences1=val_sts["sentence1"], sentences2=val_sts["sentence2"], scores=[score/5 for score in val_sts["label"]], main_similarity="cosine")
print(f"Evaluator created.\n")

# Load model
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(MODEL_ID)
print(f"\nModel {MODEL_ID} successfully loaded.\n")

# Define the loss function
train_loss = losses.MultipleNegativesRankingLoss(model=embedding_model)
print(f"Loss function defined.\n")

# Define the training arguments
#args = SentenceTransformerTrainingArguments(output_dir="base_embedding_model", num_train_epochs=1, per_device_train_batch_size=32, per_device_eval_batch_size=32, warmup_steps=100, fp16=True, eval_steps=100, logging_steps=100)
args = SentenceTransformerTrainingArguments(output_dir="finetuned_embedding_model", num_train_epochs=1, per_device_train_batch_size=32, per_device_eval_batch_size=32, warmup_steps=100, fp16=True, eval_steps=100, logging_steps=100)

# Train embedding model
#trainer = SentenceTransformerTrainer(model=embedding_model, args=args, train_dataset=train_dataset, loss=train_loss, evaluator=evaluator)
#trainer.train()
trainer = SentenceTransformerTrainer(model=embedding_model, args=args, train_dataset=train_dataset, loss=train_loss, evaluator=evaluator)
print(f"Training starts.\n")
trainer.train()
print(f"Training completed.\n")


# Evaluate our trained model
#evaluator(embedding_model)
eval = evaluator(embedding_model)
print(f"Evaluation: {eval}\n")

