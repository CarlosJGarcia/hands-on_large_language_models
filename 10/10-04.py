# Unsupervised Learning
# Basel, 27/Apr/2026
# NVIDIA GeForce RTX 3060: 100% GPU, 91% RAM (11GB de 12GB), 153W (max 170W), 74% ventilador

import nltk
from tqdm import tqdm
from rich.console import Console
from datasets import Dataset, load_dataset
from sentence_transformers.sentence_transformer import losses  
from sentence_transformers.sentence_transformer.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import models, SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments



# Download tokenizer
nltk.download("punkt")
nltk.download("punkt_tab")

# Load dataset and create a flat list of sentences
mnli = load_dataset("glue", "mnli", split="train").select(range(25_000))
flat_sentences = list(mnli["premise"]) + list(mnli["hypothesis"])

# Add noise to our input data
damaged_data = DenoisingAutoEncoderDataset(list(set(flat_sentences)))

# Create dataset
train_dataset = {"damaged_sentence": [], "original_sentence": []}
for data in tqdm(damaged_data):
    train_dataset["damaged_sentence"].append(data.texts[0])
    train_dataset["original_sentence"].append(data.texts[1])
train_dataset = Dataset.from_dict(train_dataset)


console = Console()
console.print(f"\nTrain dataset successfully loaded.", style="gold1")
print(f"Example: {train_dataset[0]}\n")


# Create an embedding similarity evaluator for stsb
val_sts = load_dataset("glue", "stsb", split="validation")
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score/5 for score in val_sts["label"]],
    main_similarity="cosine"
)
console.print("Evaluator created.\n", style="gold1")


# Create the embedding model
MODEL_ID = "bert-base-uncased"
word_embedding_model = models.Transformer(MODEL_ID)
pooling_model = models.Pooling(word_embedding_model.get_embedding_dimension(), "cls")
embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
print(f"\nModel {MODEL_ID} successfully loaded.")
console.print("Embedding model created.\n", style="gold1")

# Use the denoising auto-encoder loss
train_loss = losses.DenoisingAutoEncoderLoss(embedding_model, decoder_name_or_path=MODEL_ID, tie_encoder_decoder=False)
train_loss.decoder = train_loss.decoder.to("cuda")
console.print("\nLoss function defined.\n", style="gold1")

# Define the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="tsdae_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
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