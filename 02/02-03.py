# Librería gensim 
# Using pretrained Word Embeddings

# glove: Global Vectors for Word Representation. Determina la frecuencia en la que se usan las palabras
# Wiki: Dataset Wikipedia de 2014 
# Gigaword-50: Dataset de noticias (Reuters, etc)
# glove-wiki-gigaword indica que es el algoritmo Glove entrenado en Wikipedia y Gigaword

# Reinach 11/Apr/2026

import gensim.downloader as api

# Download embeddings (66MB, glove, trained on wikipedia, vector size: 50)
# Other options include "word2vec-google-news-300"
# More options at https://github.com/RaRe-Technologies/gensim-data
model = api.load("glove-wiki-gigaword-50")

results = model.most_similar([model['king']], topn=11)

# print the results
print("Top 11 similarities for 'king':")
for word, score in results:
    print(f"{word}: {score:.4f}")

