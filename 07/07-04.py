# Memory - Helping LLMs to remember conversations

# Kaiseraugst, 07/May/2026
# Engine: LlamaCpp hosting Phi-3-mini 
# Bridge: llama-cpp-python (library langchain_community)
# Orchestrator: LangChain (library langchain-core) - Chain with Multiple Prompts

from rich.console import Console
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

console = Console()

# Modelo: Microsoft Phi-3-mini, version fp16 (full precision) 3.8B parameters, 8 GB VRAM.
MODEL_PATH = "../models/Phi-3-mini-4k-instruct-fp16.gguf"

# Cargando el modelo en la GPU a partir del fichero GGUF
console.print(f"Llama-cpp-python", style="gold1")
print(f"Loading model {MODEL_PATH} in the GPU")
model = LlamaCpp(
    model_path=MODEL_PATH, 
    n_gpu_layers=-1, 
    max_tokens=1024, 
    n_ctx=4096, 
    seed=42, 
    verbose=False
)


# Define the chain
# The pipe operator (|) in Python is OR, but LangChain "overloads" the pipe operator to work like a Unix pipe
template = "<|endoftext|><|user|>\n{texto}<|end|>\n<|assistant|>\n"
prompt = PromptTemplate(template=template, input_variables=["texto"])
chain = prompt | model


# Inferencia local en GPU. 
question = "Hi! My name is Carlos. What is 1 + 1? What's my name?"
console.print(f"\n--- Sending Prompt with chain ---", style="gold1")
print(question)
    
response = chain.invoke({"texto": question,})
console.print(f"\n--- Response ---", style="gold1")
print(response)

# Nueva inferencia local en GPU. Entre la pregunta anterior y esta no hay contexto común.
question = "What is my name?"
console.print(f"\n--- Sending Prompt with chain ---", style="gold1")
print(question)
    
response = chain.invoke({"texto": question,})
console.print(f"\n--- Response ---", style="gold1")
print(response)



del chain
del model
print()
