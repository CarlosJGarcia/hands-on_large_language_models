import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load our data
print()
print("Loading Hugging Face - Rotten Tomatoes dataset")
print("----------------------------------------------")
data = load_dataset("rotten_tomatoes")
print(data)
print()

# Load model
print(f"Loading model SentenceTransformer")
print("---------------------------------")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Convert text to embeddings
train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

print()
print(f"Shape:")
print("-------")
print(train_embeddings.shape)


# Train a logistic regression on our train embeddings
clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data["train"]["label"])


# Evaluación del resultado - Predict previously unseen instances
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
y_pred = clf.predict(test_embeddings)
evaluate_performance(data["test"]["label"], y_pred)


# Create embeddings for our labels
label_embeddings = model.encode(["A negative review",  "A positive review"])


# Find the best matching label for each document
sim_matrix = cosine_similarity(test_embeddings, label_embeddings)

print()
print("Confusion matrix")
print("----------------")
y_pred = np.argmax(sim_matrix, axis=1)
evaluate_performance(data["test"]["label"], y_pred)