import sys
import torch
import tiktoken
from importlib.metadata import version

# Hugging Face
import transformers
import tokenizers

print()
print(f"Python version : {sys.version_info.major}.{sys.version_info.minor}")
print("Torch version  :", torch.__version__)
print("Apple Silicon acceleration:", torch.backends.mps.is_available())
print("CUDA enabled   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Compute Platform:", torch.version.cuda)
print(f"Tiktoken       : {version('tiktoken')}")
print(f"Transformers   : {transformers.__version__}")
print(f"Tokenizers     : {tokenizers.__version__}")
print()
