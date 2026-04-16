# Sentiment analysis using a Representation (non generative) model

import time
from datasets import load_dataset
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

TOP = 8500
INFERENCES = 42

# Load our data
print()
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
print(f"Loading model {model_path} and tokenizer")
print("---------------------------")
model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. Pass the already-loaded objects to the pipeline
pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)

# Test the results
print("\nTest Result:")
print("--------------")
print(pipe("The movie was absolutely fantastic, I loved the cinematography!"))
print()

print("\nTest Result:")
print("--------------")
critica = data["train"][0, -1]
print(critica['text'])
print(pipe(critica['text']))
print()

print("\nTest Result:")
print("--------------")
critica = "The rock is destined to be the 21st century's new 'conan' and that he's going to make a splash even greater than arnold schwarzenegger, jean-claud van damme or steven segal. Things really get weird, though not particularly scary: the movie is all portent and no content"
print(critica)
print(pipe(critica))
print()

print("Analyzing 8500 (TOP) movies, one by one:")
print("--------------------------------------")
matrix = [[0, 0] for _ in range(TOP)]

# This is a (bad) example of feeding data too slowly, using a standard Python loop to a powerful GPU. Envío 1 frase 8.500 veces
for n in range(TOP):
    sentence = data["train"][n]["text"]
    matrix[n][0] = sentence
    matrix[n][1] = pipe(critica)

for n in range(TOP):
    sentence = data["train"][n]["text"]
    print(f"{n+1}: {matrix[n][0]}")
    print(f"{n+1}: {matrix[n][1]}")
    print()


print("Analyzing 8500 (TOP) movies, in batches:")
print("----------------------------------------")
def data_generator():
    for text in data["train"]["text"]:
        yield text
results = []

# Use the pipeline as an iterator with a batch_size
for out in pipe(data_generator(), batch_size=64, show_progress_bar=True):
    results.append(out)

print(f"Done! Processed {len(results)} items.")
print(f"Sample result: {results[0]}")
print()

print(f"GPU Loading x{INFERENCES} times with 64 batch size")
print("------------------------------------------------")
start = time.perf_counter()
for n in range(INFERENCES):
    print(f"Inference #{n}")
    results = []
    for out in pipe(data_generator(), batch_size=64, show_progress_bar=True):
        results.append(out)
end = time.perf_counter()
tiempo_ena = (end-start)/60
print(f"GPU inference time: {tiempo_ena:.2f} minutes")
print()


print(f"GPU Loading x{INFERENCES} times with 128 batch size")
print("--------------------------------------------------")
start = time.perf_counter()
for n in range(INFERENCES):
    print(f"Inference #{n}")
    results = []
    for out in pipe(data_generator(), batch_size=128, show_progress_bar=True):
        results.append(out)
end = time.perf_counter()
tiempo_ena = (end-start)/60
print(f"GPU inference time: {tiempo_ena:.2f} minutes")
print()


print(f"GPU Loading x{INFERENCES} times with 256 batch size")
print("---------------------------------------------------")
start = time.perf_counter()
for n in range(INFERENCES):
    print(f"Inference #{n}")
    results = []
    for out in pipe(data_generator(), batch_size=256, show_progress_bar=True):
        results.append(out)
end = time.perf_counter()
tiempo_ena = (end-start)/60
print(f"GPU inference time: {tiempo_ena:.2f} minutes")
print()
