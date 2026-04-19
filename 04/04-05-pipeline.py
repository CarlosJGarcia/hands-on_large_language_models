# Text Classification with Generative Models
# Reinach 19/Apr/2026

import torch
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

def add_prefix(example):
    return {"t5": "sst2 sentence: " + example["text"]}


# Load our data
print()
print("Loading Hugging Face - Rotten Tomatoes dataset")
print("----------------------------------------------")
data = load_dataset("rotten_tomatoes")
data = data.map(add_prefix)
print(data)
print()



# Load our model - Este es el código del libro, pero ya no funciona por un cambio en la librería transformers
# pipe = pipeline("text2text-generation", model="google/flan-t5-small", device=0)
model_name = "google/flan-t5-small"


# 1. Silence the "Tied Weights" warning by fixing the config first
#config = AutoConfig.from_pretrained(model_name)
#config.tie_word_embeddings = False 

# 2. Load the model and tokenizer explicitly
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda:0")


# Load model
#pipe = pipeline("text-generation", model=model_name, device="cuda:0", max_new_tokens=5, max_length=None)
# pipe = pipeline(task="any-to-any", model=model, tokenizer=tokenizer, device="cuda:0")


# Run inference
y_pred = []

print(f"Predicting {len(data['test'])} reviews...")

model.eval() # Set model to evaluation mode
with torch.no_grad(): # Disable gradient calculation for speed and memory
    # We iterate through the 't5' column we created with the prefix
    for text in tqdm(data["test"]["t5"]):
        # 1. Prepare the input
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        
        # 2. Generate the answer (keeping your max_new_tokens=5 for speed)
        outputs = model.generate(**inputs, max_new_tokens=5)
        
        # 3. Decode the tokens back into a string ("positive" or "negative")
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        
        # 4. Convert text to numerical label
        y_pred.append(0 if "negative" in prediction else 1)


# Evaluación del resultado
def evaluate_performance(y_true, y_pred):
    """
    Create and print the classification report.
    This works perfectly with the y_pred list we just generated.
    """
    performance = classification_report(y_true, y_pred, target_names=["Negative Review", "Positive Review"], zero_division=0)
    print("\nClassification Report:")
    print(performance)

# Run the evaluation
evaluate_performance(data["test"]["label"], y_pred)
