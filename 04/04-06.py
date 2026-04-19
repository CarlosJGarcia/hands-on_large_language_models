import openai

# Create client. Environoment variable OPENAI_API_KEY with API key must be defined in the system (~/.bashrc)
client = openai.OpenAI()

response = client.models.list()
print("Connection successful!")