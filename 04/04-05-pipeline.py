# Text Classification with Generative Models
# Reinach 19/Apr/2026

import torch
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


# Load our data
print()
print("Loading Hugging Face - Rotten Tomatoes dataset")
print("----------------------------------------------")
data = load_dataset("rotten_tomatoes")

def add_prefix(example):
    return {"t5": "sst2 sentence: " + example["text"]}
data = data.map(add_prefix)

print(data)
print()

model_name = "google/flan-t5-small"

# Load our model - Este es el código del libro, pero ya no funciona por un cambio en la librería transformers
# pipe = pipeline("text2text-generation", model="google/flan-t5-small", device=0)


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda:0")
#pipe = pipeline("text2text-generation", model="google/flan-t5-small", device="cuda:0")
pipe = pipeline("text-generation", model=model_name, device="cuda:0", max_new_tokens=5, max_length=None)



# Run inference
y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"], "t5")), total=len(data["test"])):
    text = output[0]["generated_text"]
    y_pred.append(0 if text == "negative" else 1)


# Evaluación del resultado - Predict previously unseen instances
def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)

evaluate_performance(data["test"]["label"], y_pred)