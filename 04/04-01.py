# Load dataset for Sentiment analysis

from datasets import load_dataset

# Load our data
print("Loading Hugging Face - Rotten Tomatoes dataset")
print("----------------------------------------------")
data = load_dataset("rotten_tomatoes")
print(type(data))
print(data)
print()

print("Look at one example")
print("-------------------")
print(data["train"][0, -1])
print()

print("Printing the first 5 sentences:")
print("-------------------------------")
for n in range(5):
    sentence = data["train"][n]["text"]
    print(f"{n+1}. {sentence}")
