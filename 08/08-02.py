from rich.console import Console
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate


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


# Modelo: Microsoft Phi-3-mini, version fp16 (full precision) 3.8B (billion) parameters, 8 GB VRAM. Text-only, no multimodal. Requires prompt template.
MODEL_PATH = "../models/Phi-3-mini-4k-instruct-fp16.gguf"
CONTEXT_WINDOW_SIZE = 4096     # Limite de memoria de contexto para cargar el Phi-3-mini en la RTX 3060 12 GB
EMBEDDING_MODEL = "thenlper/gte-small"

# Cargando el modelo en la GPU a partir del fichero GGUF
console = Console()
console.print(f"\nLangChain & Llama-cpp-python", style="gold1")
print(f"Loading model {MODEL_PATH} in the GPU")
model = LlamaCpp(model_path=MODEL_PATH, n_gpu_layers=-1, max_tokens=1024, n_ctx=CONTEXT_WINDOW_SIZE, seed=42, verbose=False)
print()


# Embedding model for converting text to numerical representations
console.print(f"LangChain embedding model from Huggingface", style="gold1")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# Create a local vector database
db = FAISS.from_texts(texts, embedding_model)




# Create a prompt template
template = """<|user|>
Relevant information:
{context}

Provide a concise answer the following question using the relevant information provided above:
{question}<|end|>
<|assistant|>"""
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)



"""
# RAG pipeline
rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(),
    chain_type_kwargs={
        "prompt": prompt
    },
    verbose=True
)
"""


# Libero recursos manualmente. Si no, la librería muestra un warning al cerrar; Exception ignored in: <function Llama.__del__ at 0x7f99fb00a700> error when exiting the python script
del model
print()

