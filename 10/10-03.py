# Basel, 27/Apr/2026
# NVIDIA GeForce RTX 3060: 100% GPU, 83% RAM (10.2GB de 12GB), 169W (max 170W), 74% ventilador

import numpy as np
import pandas as pd
from tqdm import tqdm
from rich.console import Console
from datasets import load_dataset, Dataset
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.sentence_transformer import losses  

# Las siguientes dos líneas sustityuen a from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator


# Prepare a small set of 10000 documents for the cross-encoder
dataset = load_dataset("glue", "mnli", split="train").select(range(10_000))
mapping = {2: 0, 1: 0, 0:1}

# Data loader - Gold Dataset
#gold_examples = [InputExample(texts=[row["premise"], row["hypothesis"]], label=mapping[row["label"]]) for row in tqdm(dataset)]
gold = pd.DataFrame({
    "sentence1": dataset["premise"],
    "sentence2": dataset["hypothesis"],
    "label": [mapping[label] for label in dataset["label"]]
})
train_dataset = Dataset.from_pandas(gold)

args = SentenceTransformerTrainingArguments(
    output_dir="my_model_results",
    per_device_train_batch_size=32,
    batch_sampler=BatchSamplers.NO_DUPLICATES, # This is the magic line that replaces NoDuplicatesDataLoader
    num_train_epochs=1,
)

# Pandas DataFrame for easier data handling
gold = pd.DataFrame({"sentence1": dataset["premise"], "sentence2": dataset["hypothesis"], "label": [mapping[label] for label in dataset["label"]]})


# Train a cross-encoder on the gold dataset
cross_encoder = CrossEncoder("bert-base-uncased", num_labels=2)

# Training Arguments
args = CrossEncoderTrainingArguments(
    output_dir="cross_encoder_output",
    num_train_epochs=1,               # replaces epochs=1
    warmup_steps=100,                 # replaces warmup_steps=100
    fp16=False,                       # replaces use_amp=False (fp16 is the modern equivalent)
    per_device_train_batch_size=32,   # matches your batch_size
    batch_sampler=BatchSamplers.NO_DUPLICATES, # This replaces NoDuplicatesDataLoader
)

# Training on the gold dataset
trainer = CrossEncoderTrainer(model=cross_encoder, args=args, train_dataset=train_dataset)
trainer.train()


# Prepare the silver dataset by predicting labels with the cross-encoder
silver = load_dataset("glue", "mnli", split="train").select(range(10_000, 50_000))
pairs = list(zip(silver["premise"], silver["hypothesis"]))


# Label the sentence pairs using our fine-tuned cross-encoder
output = cross_encoder.predict(pairs, apply_softmax=True, show_progress_bar=True)
silver = pd.DataFrame(
    {
        "sentence1": silver["premise"], 
        "sentence2": silver["hypothesis"],
        "label": np.argmax(output, axis=1)
    }
)

# Combine gold + silver
console = Console()
console.print(f"\nGold + Silver combined.", style="gold1")
data = pd.concat([gold, silver], ignore_index=True, axis=0)
data = data.drop_duplicates(subset=["sentence1", "sentence2"], keep="first")
train_dataset = Dataset.from_pandas(data, preserve_index=False)


# Create an embedding similarity evaluator for stsb
val_sts = load_dataset("glue", "stsb", split="validation")
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score/5 for score in val_sts["label"]], # Normalize 0-5 to 0-1
    main_similarity="cosine",
    name="sts-b-validation"
)
console.print("Evaluator created.\n", style="gold1")


# Load model
MODEL_ID = "bert-base-uncased"
embedding_model = SentenceTransformer(MODEL_ID)
print(f"Model {MODEL_ID} successfully loaded.\n")


# Define the loss function
train_loss = losses.CosineSimilarityLoss(model=embedding_model)
console.print("Loss function defined.\n", style="gold1")

# Define the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="augmented_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100,
)

# Train model
trainer = SentenceTransformerTrainer(model=embedding_model, args=args, train_dataset=train_dataset, loss=train_loss, evaluator=evaluator)
console.print(f"Training starts.\n", style="gold1")
trainer.train()
console.print(f"\nTraining completed.\n", style="gold1")


# Evaluate the trained model
eval = evaluator(embedding_model)
print(f"Evaluation: {eval}\n")
