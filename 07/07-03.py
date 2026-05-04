# A Chain with Multiple Prompts

# Basel, 04/May/2026
# NVIDIA GeForce RTX 3060: x% GPU, 72% VRAM (8.8GB de 12GB), xW (max 170W), x% ventil
# Engine: LlamaCpp hosting Phi-3-mini 
# Bridge: llama-cpp-python (library langchain_community)
# Orchestrator: LangChain (library langchain-core)

# Out of memory en la tarjeta de 12 GB VRAM
# /tmp/pip-install-hxsm9wca/llama-cpp-python_515500c6b7a64f63a222e7a60f398abf/vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:97: CUDA error
# Aborted (core dumped)
# Unable to attach: program terminated with signal SIGABRT, Aborted.


from rich.console import Console
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

console = Console()

# Modelo: Microsoft Phi-3-mini, version fp16 (full precision) 3.8B (billion) parameters, 8 GB VRAM. Text-only, no multimodal. Requires prompt template.
MODEL_PATH = "../models/Phi-3-mini-4k-instruct-fp16.gguf"

# Cargando el modelo en la GPU a partir del fichero GGUF
console.print(f"Llama-cpp-python", style="gold1")
print(f"Loading model {MODEL_PATH} in the GPU")
model = LlamaCpp(model_path=MODEL_PATH, n_gpu_layers=-1, max_tokens=1024, n_ctx=4096, seed=42, verbose=False)



# Define the chain for the title
# The pipe operator (|) in Python is OR, but LangChain "overloads" the pipe operator to work like a Unix pipe
template = "<|endoftext|><|user|>\nCreate a title for a story about {summary}. Only return the title.<|end|>\n<|assistant|>\n"
title_prompt = PromptTemplate(template=template, input_variables=["summary"])
title_chain = title_prompt | model


# Inferencia local en GPU. Método "automático con chain". Uso el método invoke() del objeto chain. Ya no tengo que aplicar la template a mano, lo hace chain automáticamente
question = "a girl that lost her mother"
console.print(f"\n--- Sending Prompt with chain (title_chain) ---", style="gold1")
print(question)
    
response = title_chain.invoke({"summary": question,})
console.print(f"\n--- Response ---", style="gold1")
print(response)


# Define the chain for the character description
template_char = "<|endoftext|><|user|>\nDescribe the main character of a story about {summary} with the title '{title}'. Use only two sentences.<|end|>\n<|assistant|>\n"
character_prompt = PromptTemplate(template=template_char, input_variables=["summary", "title"])

# Define the chain for the story
template_story = "<|endoftext|><|user|>\nCreate a story about {summary} with the title '{title}'. The main character is: {character}. Only return the story and it cannot be longer than one paragraph.<|end|>\n<|assistant|>\n"
story_prompt = PromptTemplate(template=template_story, input_variables=["summary", "title", "character"])

# Generate the title
title_chain = title_prompt | model | StrOutputParser()


# Generate character description (needs summary + title)
# We use a dictionary to "carry forward" the summary while adding the title result
character_chain = (
    {"summary": lambda x: x["summary"], "title": title_chain}
    | character_prompt 
    | model 
    | StrOutputParser()
)
    

# Generate the final story (needs summary + title + character)
full_chain = (
    {
        "summary": lambda x: x["summary"],
        "title": title_chain,
        "character": character_chain
    }
    | story_prompt
    | model
    | StrOutputParser()
)

# --- Execution ---

question = "a girl that lost her mother"
console.print(f"\n--- Processing Chain ---", style="gold1")
print(f"Input: {question}")

# We only need to pass the first variable; the chain handles the rest
response = full_chain.invoke({"summary": question})

console.print(f"\n--- Final Result ---", style="gold1")
print(response)

# Pause 0
print()
key = input("Press ENTER to exit.")

# Libero recursos manualmente. Si no, la librería muestra un warning al cerrar; Exception ignored in: <function Llama.__del__ at 0x7f99fb00a700> error when exiting the python script
del title_chain  # As title_chain contains model
del character_chain
del full_chain
del model
print()