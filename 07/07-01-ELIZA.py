from rich.console import Console
from langchain_community.llms import LlamaCpp

# Modelo: Phi-3-mini version fp16 (full precision) 3.8B (Billion) Parameters, 8 GB VRAM
MODEL_PATH = "../models/Phi-3-mini-4k-instruct-fp16.gguf"

def print_eliza_welcome():
    # Eliminamos los espacios antes de 'Welcome' para que pegue al borde izquierdo
    # Pero mantenemos la sangría interna del arte ASCII
    welcome_text = """Welcome to
             EEEEEE   LL       IIII  ZZZZZZZZ    AAAAA
             EE       LL        II         ZZ   AA   AA
             EEEEE    LL        II       ZZZ    AAAAAAA
             EE       LL        II      ZZ      AA   AA
             EEEEEE   LLLLLL   IIII  ZZZZZZZZ   AA   AA

ELIZA is a mock (Rogerian) psychotherapist."""
    print(welcome_text)

# Cargando el modelo
print(f"Loading model {MODEL_PATH} in the GPU")
try:
    model = LlamaCpp(model_path=MODEL_PATH, n_gpu_layers=-1, max_tokens=1024, n_ctx=4096, seed=42, verbose=False)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

console = Console()

print_eliza_welcome()


while True:
    try:
        user_input = input("User: ")
    except EOFError:
        break

    # Exit conditions
    if user_input.strip().lower() in ["goodbye", "bye", "quit", "exit"]:
        print("ELIZA: GOODBYE.")
        break

    # Prompt construction to enforce ELIZA persona
    # Using Phi-3 instruction format
    prompt = (
        "<|system|>\n"
        "You are ELIZA, a Rogerian psychotherapist. Your goal is to reflect the user's "
        "statements back to them, often as questions. Keep your responses very brief "
        "and always use ALL CAPS.<|end|>\n"
        "<|user|>\n" + user_input + "<|end|>\n"
        "<|assistant|>\n"
    )
    
    try:
        response = model.invoke(prompt)
        #response = llm(prompt, stop=["===", "User:", "\n\n"])
        # The user wants the session to look like: ELIZA: IN WHAT WAY?
        # So we strip and uppercase the response.
        clean_response = response.strip().upper()
        print(f"ELIZA: {clean_response}")
    except Exception as e:
        print(f"ELIZA: ERROR IN PROCESSING: {str(e).upper()}")

# Manual cleanup
del model
print()
