# Versión del código 04-06.py para el modelo Gemma4 de Google corriendo sobre Ollama en la workstation rkauw0007.kau.roche.com
# Sigo usando la misma librería y API de OpenAI pero en lugar de usar el modelo "gpt-3.5-turbo-0125" en sus servidores y pagar por uso, uso el modelo Gemma4 en mi servidor local, gratis
# La librería y el API para acceder al modelo, siguen siendo exactamente los mismos en ambos casos, solo varía el método de conexión

import openai

# 1. Configuration for the remote Roche server
# Ensure you are on the Roche VPN or internal network to reach this host
SERVER_URL = "http://rkauw0007.kau.roche.com:11434/v1"
MODEL_NAME = "gemma4:4b" # Or "gemma4:latest" depending on what's pulled on the server

# 2. Create client pointing to the remote server
print(f"Connecting to Ollama on {SERVER_URL}...")
client = openai.OpenAI(
    base_url=SERVER_URL,
    api_key="ollama"  # Required field, but ignored by Ollama
)

def gemma_generation(prompt, document, model=MODEL_NAME):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.replace("[DOCUMENT]", document)}
    ]
    
    # This now executes on the remote rkauw0007 server
    chat_completion = client.chat.completions.create(
        messages=messages, 
        model=model, 
        temperature=0
    )
    return chat_completion.choices[0].message.content

# --- TEST CONNECTION ---
try:
    # List models to verify the server is up and gemma4 is available
    available_models = [m.id for m in client.models.list().data]
    print(f"Connection successful! Available models: {available_models}")
    
    if MODEL_NAME not in available_models:
        print(f"Warning: {MODEL_NAME} not found on server. You might need to run 'ollama pull {MODEL_NAME}' on rkauw0007.")
except Exception as e:
    print(f"Connection failed. Ensure the server is reachable and Ollama is running: {e}")

# --- EXECUTION ---
prompt = """Predict whether the following document is a positive or negative movie review: [DOCUMENT] 
If it is positive return 1 and if it is negative return 0. Do not give any other answers."""

document = "unpretentious , charming , quirky , original"

result = gemma_generation(prompt, document)
print(f"\nDocument: {document}")
print(f"Prediction: {result}")