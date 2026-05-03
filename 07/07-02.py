# Reinach, 03/May/2026
# NVIDIA GeForce RTX 3060: x% GPU, 72% VRAM (8.8GB de 12GB), xW (max 170W), x% ventil

# Engine: LlamaCpp hosting Phi-3-mini 
# Bridge: llama-cpp-python (library langchain_community)
# Orchestrator: LangChain (library langchain-core)

from rich.console import Console
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

# Modelo: Microsoft Phi-3-mini  version fp16 (full precision) 3.8B (Billion) Parameters, 8 GB VRAM. Es text-only, no multimodal.
MODEL_PATH = "../models/Phi-3-mini-4k-instruct-fp16.gguf"

# Cargando el modelo
print(f"Loading model {MODEL_PATH} in the GPU")
model = LlamaCpp(model_path=MODEL_PATH, n_gpu_layers=-1, max_tokens=1024, n_ctx=4096, seed=42, verbose=False)


# Define the prompt template string. LangChain will internally replace the {input_prompt} part
template = "<|endoftext|><|user|>\n{texto}<|end|>\n<|assistant|>\n"
prompt = PromptTemplate(template=template, input_variables=["texto"])

# Define the chain
# The pipe operator (|) in Python is OR, but LangChain "overloads" the pipe operator to work like a Unix pipe
basic_chain = prompt | model




# Create a Chain that creates our business' name
# question = "Create a funny name for a business that sells {product}."
# name_prompt = PromptTemplate(template=template, input_variables=["product"])



# Inferencia local en GPU. Uso el método invoke() del objeto model. Phi-3-mini requiere template.
question = "What are the advantages of using 16-bit precision in AI?"
prompt = "<|endoftext|><|user|>\n" + question + "<|end|>\n<|assistant|>\n"
console = Console()
console.print(f"\n--- Sending Prompt ---", style="gold1")
print(prompt)
    
response = model.invoke(prompt)
console.print(f"\n--- Response ---", style="gold1")
print(response)


# Inferencia local en GPU. Uso el método invoke() del objeto basic_chain. Ya no tengo que aplicar la template a mano, lo hace basic_chain automáticamente
question = "What are the advantages of using 16-bit precision in AI?"
console.print(f"\n--- Sending Prompt ---", style="gold1")
print(question)
    
response = basic_chain.invoke({"texto": question,})
console.print(f"\n--- Response ---", style="gold1")
print(response)







# Pause 0
print()
key = input("Press ENTER to exit.")

# Libero recursos manualmente. Si no, la librería muestra un warning ar cerrar; Exception ignored in: <function Llama.__del__ at 0x7f99fb00a700> error when exiting the python script
del basic_chain  # As basic_chain contains model
del model
print()