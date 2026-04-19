import openai

# Generate an output based on a prompt and an input document
def chatgpt_generation(prompt, document, model="gpt-3.5-turbo-0125"):
    messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content":   prompt.replace("[DOCUMENT]", document)}]
    chat_completion = client.chat.completions.create(messages=messages, model=model, temperature=0)
    return chat_completion.choices[0].message.content


# Create client. Environoment variable OPENAI_API_KEY with API key must be defined in the system (~/.bashrc)
print("Open connection to ChatGPT API")
client = openai.OpenAI()
response = client.models.list()
print("Connection successful!")

# Define a prompt template as a base
prompt = """Predict whether the following document is a positive or negative movie review: [DOCUMENT] If it is positive return 1 and if it is negative return 0. Do not give any other answers."""

# Predict the target using GPT
document = "unpretentious , charming , quirky , original"
chatgpt_generation(prompt, document)