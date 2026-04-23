import warnings
# Elimina un warning de librería huggingface_hub (Llama la usa para descargar el modelo desde Hugging Face Hub )
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# Llama (mayúscula) es la clase y llama (minúscula) es el módulo (fichero)
from llama_cpp.llama import Llama

# Phi-3 version llama.cpp
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"

# Load Model
llm = Llama.from_pretrained(repo_id=MODEL_ID, filename="*fp16.gguf", n_gpu_layers=-1, n_ctx=4096, verbose=False)

print(f"\nEl modelo {MODEL_ID} se ha cargado y descargado correctamente.")
