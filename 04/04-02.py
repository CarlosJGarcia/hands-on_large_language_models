# Sentiment analysis using a Representation (non generative) model



from datasets import load_dataset
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load our data
print("Loading Hugging Face - Rotten Tomatoes dataset")
print("----------------------------------------------")
data = load_dataset("rotten_tomatoes")
print(data)
print()

# print("Look at one example")
# print("-------------------")
# print(data["train"][0, -1])
# print()

# Path to our HF model
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Load model into pipeline
# pipe = pipeline(model=model_path, tokenizer=model_path, return_all_scores=True, device="cuda:0", use_safetensors=True)

# 1. Load model and tokenizer explicitly
# We use use_safetensors=True to ensure it ignores the .bin file entirely
print("Loading model and tokenizer")
print("---------------------------")
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, 
    use_safetensors=True, 
    device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. Pass the already-loaded objects to the pipeline
pipe = pipeline(
    "sentiment-analysis", 
    model=model, 
    tokenizer=tokenizer, 
    return_all_scores=True
)


# Test the results
print("\nTest Result:")
print("--------------")
print(pipe("The movie was absolutely fantastic, I loved the cinematography!"))