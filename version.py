import sys
import torch
import openai
import tiktoken
from importlib.metadata import version

# Hugging Face
import tokenizers
import transformers

# llama.cpp
import llama_cpp
import langchain_core
import langchain_community


print()
print("Apple Silicon acceleration:", torch.backends.mps.is_available())
print("CUDA enabled              :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Compute Platform:", torch.version.cuda)
print(f"Python version     : {sys.version_info.major}.{sys.version_info.minor}")
print(f"PyTorch            : {torch.__version__}")
print(f"Tiktoken           : {version('tiktoken')}")
print(f"Tokenizers         : {tokenizers.__version__}")
print(f"Bitsandbytes       : {version('bitsandbytes')}")
print(f"llama-cpp-python   : {llama_cpp.__version__}")
print(f"langchain-core     : {langchain_core.__version__}") # Add this line
print(f"langchain-community: {langchain_community.__version__}")
print(f"Transformers       : {transformers.__version__}")
print(f"OpenAI             : {openai.__version__}")
print()
