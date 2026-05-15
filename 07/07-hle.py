# Dataset: HLE (Humanity Last Exam) from Hugging Face.
# Model  : Microsoft Phi-3-mini, version fp16 (full precision) 3.8B (billion) parameters, 8 GB VRAM. Text-only, no multimodal. Requires prompt template.
# Librerías: datasets, Hugging Face
#            LangChain (que no es de Hugging Face), que a su vez usa:
#                   - llama-cpp-python para cargar el modelo en la GPU/VRAM a partir de un fichero GGUF (GPT Generated Unified Format)
#                   - PromptTemplate       

# Reinach 13/May/2026z

from rich.console import Console
from datasets import load_dataset
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate


# Dataset HLE
DATASET_ID = "cais/hle"
SPLIT = "test"              # El único 'split'. Contiene 2.500 preguntas, algunas con imágenes. Indica que este es un dataset de test, no de training ni de validación

# Modelo: Microsoft Phi-3-mini, version fp16 (full precision) 3.8B (billion) parameters, 8 GB VRAM. Text-only, no multimodal. Requires prompt template.
MODEL_PATH = "../models/Phi-3-mini-4k-instruct-fp16.gguf"

# Context window. Contiene la pregunta y la respuesta generada. Si se supera el tamaño asignado, la inferencia falla y el programa se interrumpe
# El valor por defecto de Phi-3-mini es 4096. Si asignamos más, da un warning, pero evita que el programa se interrumpa en las preguntas que superan los 4096 tokens
# El hecho de tener que triplicar el tamaño de la ventana de contexto por defecto indica que las preguntas tienen una complejidad muy superior a la capacidad del modelo

# 4096 (4K), 8129 (8 K) y 12.288 (12 K) son tokens. 
# Si 1 palabra = 0'75 token, 8.192 tokens = 6.144 palabras (12 - 15 páginas de texto)
# En precisión FP16 (2 Bytes cada valor), 8.192 tokens ocupan 768 MB
# En precisión FP16 (2 Bytes cada valor), 12.288 tokens ocupan 1.152 MB

# CONTEXT_WINDOW_SIZE = 12032     # Limite de memoria de contexto para cargar el Phi-3-mini en la RTX 3060 12 GB
CONTEXT_WINDOW_SIZE = 12288       # Requiere 16GB VRAM

# Uso de VRAM RTX 3060
# Capacidad          12000 MB 100%
# Modelo Phi-3 FP16   7600 MB  62%
# Context 12032 token 1152 MB   9%
# CUDA Overhead       2300 MB  19%
# Libre               1200 MB  10%
# El siguiente nivel de ampliación de la ventana de contexto son 12288 tokens, que ya no caben en la targeta de 12 GB

console = Console()
console.print(f"\nDataset", style="gold1")
print(f"Loading dataset '{DATASET_ID}', split='{SPLIT}'...")
        
# El método load_dataset usa "Apache Arrow" así que en realidad las 2.500 preguntas/respuestas del dataset no están cargados en RAM, aunque python piensa que sí
dataset = load_dataset(DATASET_ID, split=SPLIT)

print(f"- Dataset structure: {dataset}")
print(f"- Number of rows: {len(dataset)}")


# Cargando el modelo en la GPU a partir del fichero GGUF
console.print(f"\nLangChain & Llama-cpp-python", style="gold1")
print(f"Loading model {MODEL_PATH} in the GPU")
model = LlamaCpp(model_path=MODEL_PATH, n_gpu_layers=-1, max_tokens=1024, n_ctx=CONTEXT_WINDOW_SIZE, seed=42, verbose=False)
print()

# Método "automático" de enviar prompts usando template y de recoger la respuesta. Define the chain.
# The pipe operator (|) in Python is OR, but LangChain "overloads" the pipe operator to work like a Unix pipe
template = "<|endoftext|><|user|>\n{texto}<|end|>\n<|assistant|>\n"
prompt = PromptTemplate(template=template, input_variables=["texto"])
chain = prompt | model

# Define the empty list
m = 0
lista = []
correct_count = 0
LIMIT = len(dataset)
#LIMIT = 10            # Versión reducida para pruebas

# Bucle principal para recorrer el dataset
for n in range(0, LIMIT):

    row = dataset[n]

    id_val = row['id']
    question_val = row['question']
    image_val = row['image']
    true_answer_val = str(row['answer'])    # La respuesta correcta según el dataset

    # Check if the row (question) is text-only if yes, I ask it to the model, otherwise I ignore it as Phi-3 is text-based
    if image_val == "":
        
        # Inferencia
        prompt = "<|endoftext|><|user|>\n" + question_val + "<|end|>\n<|assistant|>\n"
        response = model.invoke(prompt)

        # Evaluación de la respueta por inferencia vs respuesta verdadera
        correct = true_answer_val.lower().strip() in response.lower()
        if correct:
            correct_count += 1

        # Guardo la pregunta y la respuesta en mi lista
        new_row = {"id": m, "question": question_val, "answer": response, "correct": correct}
        lista.append(new_row)
        print(f"Question {m}/{LIMIT} completed.")
        m += 1    

        
    
# Una vez terminada la sesión de preguntas/respuestas muestra el resultado
for n in range(0, len(lista)):
    print(lista[n])

print (f"\nNúmero de respuestas acertadas: {correct_count}")
correct_percent = correct_count / LIMIT * 100
print (f"Porcentaje de respuestas acertadas: {correct_percent:.0f}%")

# Pause 0
print()
key = input("Press ENTER to exit.")

# Libero recursos manualmente. Si no, la librería muestra un warning al cerrar; Exception ignored in: <function Llama.__del__ at 0x7f99fb00a700> error when exiting the python script
del chain  # As basic_chain contains model
del model
print()

# Analizando el resultado de las primeras 50 preguntas/respuestas con Gemini Pro, el 100% de las respuestas de Phi-3 son incorrectas
# The dataset HLE is not a standard AI benchmark. Was created specifically to be the hardest test in the world.
# You essentially put a brilliant middle-school student into a PhD-level theoretical physics defense and stretched his brain to the breaking point.
# Small models (3.8 B) like Phi-3 are highly optimized for tasks like summarizing emails, writing basic Python scripts or acting as a simple conversational agent.
# When a small model encounters a concept it doesn't understand (like computing the Poincaré polynomial of a Lie algebra), it doesn't know how to say "I don't know." 
# Instead, it uses its excellent language skills to confidently string together advanced-sounding words that mathematically mean absolutely nothing (hallucination)