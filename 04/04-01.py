# Load dataset for Sentiment analysis

from datasets import load_dataset

# Load our data
print("Loading Hugging Face - Rotten Tomatoes dataset")
print("----------------------------------------------")
data = load_dataset("rotten_tomatoes")
print(data)
print()

print("Look at one example")
print("-------------------")
print(data["train"][0, -1])
print()

