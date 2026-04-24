# Uso del modelo Phi-3 con la librería llama-cpp-python

import json

# Elimina un warning de la librería huggingface_hub (Llama la usa para descargar el modelo desde Hugging Face Hub )
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# llama (minúscula) es el módulo (fichero) y Llama (mayúscula) es la clase
from llama_cpp.llama import Llama

# Versión de Phi-3 para llama-cpp
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"

# Load Model
llm = Llama.from_pretrained(repo_id=MODEL_ID, filename="*fp16.gguf", n_gpu_layers=-1, n_ctx=4096, verbose=False)
print(f"\nEl modelo {MODEL_ID} se ha cargado correctamente.\n")


# Generate output
query = [{"role": "user", "content": "Create a warrior for an RPG in JSON format."},]
output = llm.create_chat_completion(messages=query, response_format={"type": "json_object"}, temperature=0,)['choices'][0]['message']["content"]
print(f"Query: {query}\n")
print(f"Respuesta: {output}\n")

# Check that the output is JSON
output_json = json.dumps(json.loads(output), indent=4)
print(f"Respuesta en JSON: {output_json}\n")


# Generate output
query = [{"role": "user", "content": "The sky is"},]
output = llm.create_chat_completion(messages=query, response_format={"type": "json_object"}, temperature=0,)['choices'][0]['message']["content"]
print(f"Query: {query}\n")
print(f"Respuesta: {output}\n")
