# A Chain with Multiple Prompts (Fixed Version)

# Basel, 06/May/2026
# NVIDIA GeForce RTX 5060 Ti: Optimized GPU Inference
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

# ---------------------------------------------------------------------
# Define the templates
# ---------------------------------------------------------------------
template_title = "<|endoftext|><|user|>\nCreate a title for a story about {summary}. Only return the title.<|end|>\n<|assistant|>\n"
title_prompt = PromptTemplate(template=template_title, input_variables=["summary"])

template_char = "<|endoftext|><|user|>\nDescribe the main character of a story about {summary} with the title '{title}'. Use only two sentences.<|end|>\n<|assistant|>\n"
character_prompt = PromptTemplate(template=template_char, input_variables=["summary", "title"])

template_story = "<|endoftext|><|user|>\nCreate a story about {summary} with the title '{title}'. The main character is: {character}. Only return the story and it cannot be longer than one paragraph.<|end|>\n<|assistant|>\n"
story_prompt = PromptTemplate(template=template_story, input_variables=["summary", "title", "character"])

# ---------------------------------------------------------------------
# Standalone execution test (Title only)
# ---------------------------------------------------------------------
title_chain = title_prompt | model | StrOutputParser()

question = "a girl that lost her mother"
console.print(f"\n--- Sending Prompt with chain (title_chain) ---", style="gold1")
print(question)
    
response = title_chain.invoke({"summary": question})
console.print(f"\n--- Response ---", style="gold1")
print(response)

# ---------------------------------------------------------------------
# Build the sequential, thread-safe pipeline
# ---------------------------------------------------------------------
title_generator = title_prompt | model | StrOutputParser()
character_generator = character_prompt | model | StrOutputParser()
story_generator = story_prompt | model | StrOutputParser()

# We use RunnablePassthrough.assign to pass dictionaries forward sequentially
full_chain = (
    # Step 1: Receives {"summary": ...} -> Appends "title" to the dict
    RunnablePassthrough.assign(title=title_generator)
    
    # Step 2: Receives {"summary": ..., "title": ...} -> Appends "character" to the dict
    | RunnablePassthrough.assign(character=character_generator)
    
    # Step 3: Receives {"summary": ..., "title": ..., "character": ...} -> Generates final story string
    | story_generator
)

# ---------------------------------------------------------------------
# Execution of the entire chain
# ---------------------------------------------------------------------
question = "a girl that lost her mother"
console.print(f"\n--- Processing Chain ---", style="gold1")
print(f"Input: {question}")

# Invoking the full pipeline safely on a single thread
response = full_chain.invoke({"summary": question})

console.print(f"\n--- Final Result ---", style="gold1")
print(response)

# Pause to view output before exit
print()
key = input("Press ENTER to exit.")

# Libero recursos manualmente para prevenir warnings o crashes de CUDA al terminar el script
del title_chain
del title_generator
del character_generator
del story_generator
del full_chain
del model
print()
