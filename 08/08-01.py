import os
import faiss
import cohere
import numpy as np
import pandas as pd
from tqdm import tqdm

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string

# Retrieve the API key from the environment variable
api_key = os.getenv('COHERE_API_KEY')

# Create and retrieve a Cohere API key from os.cohere.ai
co = cohere.Client(api_key)

text = """
Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan. 
It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine. 
Set in a dystopian future where humanity is struggling to survive, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for mankind.

Brothers Christopher and Jonathan Nolan wrote the screenplay, which had its origins in a script Jonathan developed in 2007. 
Caltech theoretical physicist and 2017 Nobel laureate in Physics[4] Kip Thorne was an executive producer, acted as a scientific consultant, and wrote a tie-in book, The Science of Interstellar. 
Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in the Panavision anamorphic format and IMAX 70 mm. 
Principal photography began in late 2013 and took place in Alberta, Iceland, and Los Angeles. 
Interstellar uses extensive practical and miniature effects and the company Double Negative created additional digital effects.

Interstellar premiered on October 26, 2014, in Los Angeles. 
In the United States, it was first released on film stock, expanding to venues using digital projectors. 
The film had a worldwide gross over $677 million (and $773 million with subsequent re-releases), making it the tenth-highest grossing film of 2014. 
It received acclaim for its performances, direction, screenplay, musical score, visual effects, ambition, themes, and emotional weight. 
It has also received praise from many astronomers for its scientific accuracy and portrayal of theoretical astrophysics. Since its premiere, Interstellar gained a cult following,[5] and now is regarded by many sci-fi experts as one of the best science-fiction films of all time.
Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects, and received numerous other accolades"""

# Split into a list of sentences
texts = text.split('.')

# Clean up to remove empty spaces and new lines
texts = [t.strip(' \n') for t in texts]


# Get the embeddings
response = co.embed(
  texts=texts,
  model="embed-english-v3.0",
  input_type="search_document",
).embeddings

embeds = np.array(response)
print(embeds.shape)


dim = embeds.shape[1]
index = faiss.IndexFlatL2(dim)
print(index.is_trained)
index.add(np.float32(embeds))


def search(query, number_of_results=3):
  
  # 1. Get the query's embedding
  query_embed = co.embed(texts=[query], model="embed-english-v3.0",
                input_type="search_query",).embeddings[0]

  # 2. Retrieve the nearest neighbors
  distances , similar_item_ids = index.search(np.float32([query_embed]), number_of_results) 

  # 3. Format the results
  texts_np = np.array(texts) # Convert texts list to numpy for easier indexing
  results = pd.DataFrame(data={'texts': texts_np[similar_item_ids[0]], 
                              'distance': distances[0]})
  
  # 4. Print and return the results
  print(f"Query:'{query}'\nNearest neighbors:")
  return results

query = "how precise was the science"
results = search(query)
print(results)



def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc


tokenized_corpus = []
for passage in tqdm(texts):
    tokenized_corpus.append(bm25_tokenizer(passage))

bm25 = BM25Okapi(tokenized_corpus)

def keyword_search(query, top_k=3, num_candidates=15):
    print("Input question:", query)

    ##### BM25 search (lexical search) #####
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -num_candidates)[-num_candidates:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    
    print(f"Top-3 lexical search (BM25) hits")
    for hit in bm25_hits[0:top_k]:
        print("\t{:.3f}\t{}".format(hit['score'], texts[hit['corpus_id']].replace("\n", " ")))

keyword_search(query = "how precise was the science")


# ==========================================
# PART 3: RERANKING
# ==========================================

print(f"\n--- RERANKING ---")
print(f"Query: '{query}'")

# Use Cohere's Rerank API to re-evaluate the relevance of the documents
rerank_results = co.rerank(
    query=query, 
    documents=texts, # In a huge dataset, you would pass the BM25/FAISS results here instead of all texts
    model="rerank-english-v3.0", # Specify the current model!
    top_n=3, 
    return_documents=True
)

print(f"Top-3 Reranked hits:")
for idx, hit in enumerate(rerank_results.results):
    print(f"\tRank {idx+1} (Score: {hit.relevance_score:.3f}): {hit.document.text}")

# ==========================================
# PART 4: TWO-STAGE PIPELINE (BM25 + RERANK)
# ==========================================

def keyword_and_reranking_search(query, top_k=3, num_candidates=10):
    print("\n--- TWO STAGE SEARCH: BM25 + RERANK ---")
    print("Input question:", query)
    
    ##### BM25 search (lexical search) #####
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -num_candidates)[-num_candidates:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    
    print(f"\nTop-{top_k} lexical search (BM25) hits:")
    for hit in bm25_hits[0:top_k]:
        print("\t{:.3f}\t{}".format(hit['score'], texts[hit['corpus_id']].replace("\n", " ")))
    
    # Add re-ranking
    docs = [texts[hit['corpus_id']] for hit in bm25_hits]
    
    print(f"\nTop-{top_k} hits by rank-API ({len(bm25_hits)} BM25 hits re-ranked):")
    
    # ADDED THE MODEL PARAMETER HERE:
    results = co.rerank(
        query=query, 
        documents=docs, 
        model="rerank-english-v3.0", 
        top_n=top_k, 
        return_documents=True
    )
    
    for hit in results.results:
        print("\t{:.3f}\t{}".format(hit.relevance_score, hit.document.text.replace("\n", " ")))

# Call the function so it actually runs when you execute the script!
keyword_and_reranking_search(query="how precise was the science")

# ==========================================
# PART 5: GROUNDED GENERATION (RAG)
# ==========================================

print("\n--- GROUNDED GENERATION (RAG) ---")
query = "income generated"

# 1- Retrieval
# We'll use embedding search. But ideally we'd do hybrid
print(f"Searching for: '{query}'...")
results = search(query)

# 2- Grounded Generation
# Format the retrieved texts into a dictionary format Cohere expects
docs_dict = [{'text': text} for text in results['texts']]

# Pass the query and the retrieved documents to the chat model
response = co.chat(
    message=query,
    documents=docs_dict,
    model="command-a-03-2025" # ADDED THE MODEL PARAMETER HERE!
)

print("\nFinal AI Answer:")
print(response.text)