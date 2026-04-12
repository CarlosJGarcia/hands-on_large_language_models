# Librería gensim (runs on CPU) 
# Training a song embedding model

# Reinach 12/Apr/2026

import numpy as np
import pandas as pd
from urllib import request
from gensim.models import Word2Vec


# Get the playlist dataset file
data = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt')

# Parse the playlist dataset file. Skip the first two lines as
# they only contain metadata
lines = data.read().decode("utf-8").split('\n')[2:]

# Remove playlists with only one song
playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]

# Load song metadata
songs_file = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')
songs_file = songs_file.read().decode("utf-8").split('\n')
songs = [s.rstrip().split('\t') for s in songs_file]
songs_df = pd.DataFrame(data=songs, columns = ['id', 'title', 'artist'])
songs_df = songs_df.set_index('id')

print( 'Playlist #1:\n ', playlists[0])
print()
print( 'Playlist #2:\n ', playlists[1])
print()


# Train our Word2Vec model
print("Entrenando el modelo Word2Vec")
print()
model = Word2Vec(playlists, vector_size=32, window=20, negative=50, min_count=1, workers=4)


# Ask the model for songs similar to song #2172
song_id = 2172
print("Dime canciones que vayan bien con esta:")
print(songs_df.iloc[2172])
print()

prediction = model.wv.most_similar(positive=str(song_id))
print(prediction)
print()


# Muestro los nombres de las las 5 primeras canciones recomendadas
def print_recommendations(song_id):
    similar_songs = np.array(
        model.wv.most_similar(positive=str(song_id),topn=5)
    )[:,0]
    return  songs_df.iloc[similar_songs]

print(print_recommendations(2172))
print()

