import os
from openai import OpenAI

# The model name to use in Ollama
MODEL_ID = "gemma4:26b-agent"
MAX_NEW_TOKENS = 500

# Get the Ollama host from environment variables
# The user specified OLLAMA_HOST=rbauw0007.kau.roche.com:11434
ollama_host = os.environ.get("OLLAMA_HOST", "localhost:11434")
base_url = f"http://{ollama_host}/v1"

# Initialize the OpenAI client pointing to the Ollama server
client = OpenAI(
    base_url=base_url,
    api_key="ollama"  # Ollama doesn't require a real key, but the client needs one
)

print(f"Connected to Ollama at: {base_url}")
print("Model ready!")

# Get user input from the terminal
user_prompt = input("\nUser prompt: ")

# Run inference using the OpenAI client
print("\nGemma 4 thinking...")
try:
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=MAX_NEW_TOKENS,
        temperature=0  # Equivalent to do_sample=False
    )

    # Print result
    print(f"\nGenerated reply: {response.choices[0].message.content}")
except Exception as e:
    print(f"\nAn error occurred: {e}")

# 4. PREVENT EXIT: Wait for user signal
input("\nPress ENTER to close the script")
