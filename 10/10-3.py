import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder

# Las siguientes dos líneas sustityuen a from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

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