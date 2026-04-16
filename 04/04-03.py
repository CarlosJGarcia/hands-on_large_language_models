import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


# Load our data
print()
print("Loading Hugging Face - Rotten Tomatoes dataset")
print("----------------------------------------------")
data = load_dataset("rotten_tomatoes")
print(data)
print()
print("Look at one example")
print("-------------------")
print(data["train"][0, -1])
print()


# Path to our HF model
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Load model into pipeline. Por un update de seguridad no se puede usar la línea que viene en el libro. En su lugar hay que hacer esto:
# pipe = pipeline(model=model_path, tokenizer=model_path, return_all_scores=True,device="cuda:0")
model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)


# Run inference. Hay que volver a modificar otra vez el código del libro, porque output no es una lista
y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"])):
    # The pipeline sometimes returns a nested list: [[{'label':..., 'score':...}, ...]]
    # We strip the outer list if it exists
    if isinstance(output, list) and isinstance(output[0], list):
        current_output = output[0]
    elif isinstance(output, list):
        current_output = output
    else:
        current_output = [output]

    # Instead of relying on index 0 or 2, find by label name for safety
    # For this model: 0 is Negative, 1 is Neutral, 2 is Positive
    scores = {item['label']: item['score'] for item in current_output}
    
    # Map the labels to the format you need (comparing Negative vs Positive)
    # The labels in this model are 'negative', 'neutral', 'positive'
    neg = scores.get('negative', 0)
    pos = scores.get('positive', 0)
    
    assignment = np.argmax([neg, pos])
    y_pred.append(assignment)


# Evaluación del resultado

def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)

print()
print("Confusion matrix")
print("----------------")
evaluate_performance(data["test"]["label"], y_pred)