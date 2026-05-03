from rich.console import Console
from langchain_community.llms import LlamaCpp

# Modelo: Microsoft Phi-3-mini version fp16 (full precision) 3.8B (Billion) Parameters, 8 GB VRAM. Es text-only, no multimodal.
MODEL_PATH = "../models/Phi-3-mini-4k-instruct-fp16.gguf"

# Cargando el modelo
print(f"Loading model {MODEL_PATH} in the GPU")
model = LlamaCpp(model_path=MODEL_PATH, n_gpu_layers=-1, max_tokens=1024, n_ctx=4096, seed=42, verbose=False)

# Local, GPU-accelerated LLM inference. Uso el método invoke() del objeto llm que devuelve LlamaCpp al cargar el LLM
question = "What are the advantages of using 16-bit precision in AI?"
prompt = "<|user|>\n" + question + "<|end|>\n<|assistant|>\n"
console = Console()
console.print(f"\n--- Sending Prompt ---", style="gold1")
print(prompt)
    
response = model.invoke(prompt)
console.print(f"\n--- Response ---", style="gold1")
print(response)

# Pause 0
print()
key = input("Press ENTER to exit.")

# Manual cleanup to prevernt an Exception ignored in: <function Llama.__del__ at 0x7f99fb00a700> error when exiting the python script
del model
print()


"""
Idea for experiment 

1. Text Chunking: Use your 14-core Xeon to split that Wikipedia text into small pieces. (based on 01/01-3-wiki.py)

2. Embeddings: Use your GPU to turn those text pieces into vectors (mathematical representations of meaning).

3. Local RAG: Create a script where you ask a question, the code finds the right Wikipedia chunk, and sends it to Phi-3 to answer based only on that data.

RAG: Retrieval Augmented Generation. The most common architectural pattern in AI development in 2026

"""