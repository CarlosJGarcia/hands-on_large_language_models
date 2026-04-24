from mteb import MTEB
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

# Check one example
print(f"\nTrain dataset successfully loaded.\n")
print(f"{train_dataset[2]}\n")

# Use a base model
MODEL_ID = "bert-base-uncased"
embedding_model = SentenceTransformer(MODEL_ID)
print(f"\nModel {MODEL_ID} successfully loaded.\n")

# Define the loss function. In softmax loss, we will also need to explicitly set the number of labels.
train_loss = losses.SoftmaxLoss(model=embedding_model, embedding_dimension=embedding_model.get_embedding_dimension(), num_labels=3)

# Create an embedding similarity evaluator for STSB
val_sts = load_dataset("glue", "stsb", split="validation")
evaluator = EmbeddingSimilarityEvaluator(sentences1=val_sts["sentence1"], sentences2=val_sts["sentence2"], scores=[score/5 for score in val_sts["label"]], main_similarity="cosine")
print(f"Evaluator created.\n")

# Define the training arguments
args = SentenceTransformerTrainingArguments(output_dir="base_embedding_model", num_train_epochs=1, per_device_train_batch_size=32, per_device_eval_batch_size=32, warmup_steps=100, fp16=True, eval_steps=100, logging_steps=100)

# Train embedding model
trainer = SentenceTransformerTrainer(model=embedding_model, args=args, train_dataset=train_dataset, loss=train_loss, evaluator=evaluator)
trainer.train()

# Evaluate our trained model
evaluator(embedding_model)
eval = evaluator(embedding_model)
print(f"\nEvaluation: {eval}\n")

# Choose evaluation task
evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(embedding_model)
print(f"Evaluation: {results}\n")