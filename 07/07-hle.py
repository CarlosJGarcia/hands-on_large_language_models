# Loads the HLE (Humanity Last Exam) dataset from Hugging Face.
# Reinach 13/May/2026z

from rich.console import Console
from datasets import load_dataset
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate


# Dataset HLE
DATASET_ID = "cais/hle"
SPLIT = "test"              # El único 'split'. Contiene 2.500 preguntas, algunas con imágenes

# Modelo: Microsoft Phi-3-mini, version fp16 (full precision) 3.8B (billion) parameters, 8 GB VRAM. Text-only, no multimodal. Requires prompt template.
MODEL_PATH = "../models/Phi-3-mini-4k-instruct-fp16.gguf"

# El valor por defecto es 4096 y si lo superamos da un warning, pero evita que el programa se cuelgue en la pregunta 78 que supera los 4096 tokens
CONTEXT_WINDOW_SIZE = 8192

print(f"\nLoading dataset '{DATASET_ID}', split='{SPLIT}'...")
        
# El método load_dataset usa "Apache Arrow" así que en realidad los 6M artículos y 16 GB de datos del dataset no están cargados en RAM, aunque python piensa que sí
dataset = load_dataset(DATASET_ID, split=SPLIT)

console = Console()
console.print(f"Dataset loaded successfully!\n", style="gold1")
print(f"Dataset structure: {dataset}")
print(f"Number of rows: {len(dataset)}")
#print(f"Train dataset size: {dataset['num_rows'].data.nbytes/(1024*1024):.2f} MB\n")


# Cargando el modelo en la GPU a partir del fichero GGUF
console.print(f"Llama-cpp-python", style="gold1")
print(f"Loading model {MODEL_PATH} in the GPU")
model = LlamaCpp(model_path=MODEL_PATH, n_gpu_layers=-1, max_tokens=1024, n_ctx=CONTEXT_WINDOW_SIZE, seed=42, verbose=False)


# Método "automático". Define the chain
# The pipe operator (|) in Python is OR, but LangChain "overloads" the pipe operator to work like a Unix pipe
template = "<|endoftext|><|user|>\n{texto}<|end|>\n<|assistant|>\n"
prompt = PromptTemplate(template=template, input_variables=["texto"])
chain = prompt | model


# Define the empty list
m = 0
lista = []
for n in range(0, len(dataset)):
# for n in range(0, 3):

    row = dataset[n]

    id_val = row['id']
    question_val = row['question']
    image_val = row['image']

    if image_val == "":
        # Row is text-only

        # Inferencia
        prompt = "<|endoftext|><|user|>\n" + question_val + "<|end|>\n<|assistant|>\n"
        response = model.invoke(prompt)

        # Añado pregunta y respuesta a mi lista
        new_row = {"id": m, "question": question_val, "answer": response}
        lista.append(new_row)
        m = m+1    

        print(f"Question {m}/{len(dataset)}")
    
for n in range(0, len(lista)):
    print(lista[n])


# Inspecting the first item in the 'train' split
"""
console.print(f"Sample data:", style="gold1")
sample = dataset['train'][0] 
for key, value in sample.items():
    content_preview = str(value)[:200].replace('\n', ' ')
    print(f"{key}: {content_preview}...")
"""
print()
key = input("Press ENTER to exit.")

# Libero recursos manualmente. Si no, la librería muestra un warning al cerrar; Exception ignored in: <function Llama.__del__ at 0x7f99fb00a700> error when exiting the python script
del chain  # As basic_chain contains model
del model
print()
