from rich.console import Console
from langchain_community.llms import LlamaCpp

console = Console()

# Modelo: Microsoft Phi-3-mini version fp16 (full precision) 3.8B (Billion) Parameters, 8 GB VRAM. Es text-only, no multimodal.
MODEL_PATH = "../models/Phi-3-mini-4k-instruct-fp16.gguf"

# Cargando el modelo
console.print(f"Llama-cpp-python", style="gold1")
print(f"Loading model {MODEL_PATH} in the GPU")
model = LlamaCpp(model_path=MODEL_PATH, n_gpu_layers=-1, max_tokens=1024, n_ctx=4096, seed=42, verbose=False)

# Inferencia local en GPU. Uso el método invoke() del objeto model. Phi-3-mini requiere template.
question = "What are the advantages of using 16-bit precision in AI?"
prompt = "<|user|>\n" + question + "<|end|>\n<|assistant|>\n"

console.print(f"\n--- Sending Prompt ---", style="gold1")
print(prompt)
    
response = model.invoke(prompt)
console.print(f"\n--- Response ---", style="gold1")
print(response)

# Pause 0
print()
key = input("Press ENTER to exit.")

# Libero recursos manualmente. Si no, la librería muestra un warning ar cerrar; Exception ignored in: <function Llama.__del__ at 0x7f99fb00a700> error when exiting the python script
del model
print()