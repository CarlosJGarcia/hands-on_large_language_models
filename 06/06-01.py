import torch
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
MAX_NEW_TOKENS = 500


# Load Model and Tokenizer using transformers library
# trust_remote_code=False + attn_implementation="eager" para evitar warnings
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda", torch_dtype="auto", trust_remote_code=False, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Define a clean Generation Config to avoid warnings being displayed
gen_cfg = GenerationConfig.from_pretrained(MODEL_ID)
gen_cfg.do_sample = False
gen_cfg.temperature = None
gen_cfg.max_new_tokens = MAX_NEW_TOKENS
model.generation_config.max_length = 4096

# Create the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

print(f"\nModel {MODEL_ID} loaded successfully!")


# Get user input from the terminal
user_prompt = input("\nUser prompt: ")
user_prompt = [{"role": "user", "content": user_prompt}]
prompt = tokenizer.apply_chat_template(user_prompt, tokenize=False, add_generation_prompt=True)
print(f"Prompt: {prompt}")

# Run inference adnd print result
print("Phi-3 thinking...")
output = pipe(prompt)
print(f"\nGenerated reply: {output[0]['generated_text']}")


# Prompt components
persona = "You are an expert in Large Language models. You excel at breaking down complex papers into digestible summaries.\n"
instruction = "Summarize the key findings of the paper provided.\n"
context = "Your summary should extract the most crucial points that can help researchers quickly understand the most vital information of the paper.\n"
data_format = "Create a bullet-point summary that outlines the method. Follow this up with a concise paragraph that encapsulates the main results.\n"
audience = "The summary is designed for busy researchers that quickly need to grasp the newest trends in Large Language Models.\n"
tone = "The tone should be professional and clear.\n"
text = "All pre-trained models are identified by a model ID. When you create a tokenizer that a pre-trained model requires, it will check with the pre-trained model’s config to instantiate the correct tokenizer object, similarly, for the model. Therefore, you just need to use AutoTokenizer and AutoModel instead of the specific classes, such as BertTokenizer and BertModel. Knowing how a transformer model usually works, you should expect the core model to take the input tokens and output logit tensors. Therefore, you used argmax above to convert the logits to token IDs and convert the IDs to strings using the tokenizer’s decode method. However, you must provide the access token if you want to use a gated model with the above code. The way to set up the access token is to use some environment variables. You can find all environment variables that matter to the transformers library in the documentation; the most important ones are:"
data = f"Text to summarize: {text}"

# The full prompt - remove and add pieces to view its impact on the generated output
query = persona + instruction + context + data_format + audience + tone + data
print(f"\n{query}")


# Use a single example of using the made-up word in a sentence
one_shot_prompt = [
    {
        "role": "user",
        "content": "A 'Gigamuru' is a type of Japanese musical instrument. An example of a sentence that uses the word Gigamuru is:"
    },
    {
        "role": "assistant",
        "content": "I have a Gigamuru that my uncle gave me as a gift. I love to play it at home."
    },
    {
        "role": "user",
        "content": "To 'screeg' something is to swing a sword at it. An example of a sentence that uses the word screeg is:"
    }
]
query = tokenizer.apply_chat_template(one_shot_prompt, tokenize=False)
print(f"\nPrompt: {query}")

# Generate the output
print("Phi-3 thinking...")
output = pipe(one_shot_prompt)
print(f"\nGenerated reply: {output[0]['generated_text']}")


# Create name and slogan for a product
product_prompt = [{"role": "user", "content": "Create a name and slogan for a chatbot that leverages LLMs."}]
print("\nPhi-3 thinking...")
outputs = pipe(product_prompt)
product_description = outputs[0]["generated_text"]
print(f"\nGenerated reply: {product_description}")


# Based on a name and slogan for a product, generate a sales pitch
sales_prompt = [
    {"role": "user", "content": f"Generate a very short sales pitch for the following product: '{product_description}'"}
]
print("\nPhi-3 thinking...")
outputs = pipe(sales_prompt)
sales_pitch = outputs[0]["generated_text"]
print(f"\nGenerated reply: {sales_pitch}")


# Answering with chain-of-thought
cot_prompt = [
    {"role": "user", "content": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"},
    {"role": "assistant", "content": "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."},
    {"role": "user", "content": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"}
]

# Generate the output
print("\nPhi-3 thinking...")
outputs = pipe(cot_prompt)
print(f"\nGenerated reply: {outputs[0]['generated_text']}")


# Zero-shot chain-of-thought
zeroshot_cot_prompt = [
    {"role": "user", "content": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have? Let's think step-by-step."}
]

# Generate the output
print("\nPhi-3 thinking...")
outputs = pipe(zeroshot_cot_prompt)
print(f"\nGenerated reply: {outputs[0]['generated_text']}")


# Zero-shot tree-of-thought
zeroshot_tot_prompt = [
    {"role": "user", "content": "Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realizes they're wrong at any point then they leave. The question is 'The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?' Make sure to discuss the results."}
]
# Generate the output
print("\nPhi-3 thinking...")
outputs = pipe(zeroshot_tot_prompt)
print(f"\nGenerated reply: {outputs[0]['generated_text']}")



# Zero-shot learning: Providing no examples
zeroshot_prompt = [
    {"role": "user", "content": "Create a character profile for an RPG game in JSON format."}
]
# Generate the output
print("\nPhi-3 thinking...")
outputs = pipe(zeroshot_prompt)
print(f"\nGenerated reply: {outputs[0]['generated_text']}")


# One-shot learning: Providing an example of the output structure
one_shot_template = """Create a short character profile for an RPG game. Make sure to only use this format:

{
  "description": "A SHORT DESCRIPTION",
  "name": "THE CHARACTER'S NAME",
  "armor": "ONE PIECE OF ARMOR",
  "weapon": "ONE OR MORE WEAPONS"
}
"""
one_shot_prompt = [
    {"role": "user", "content": one_shot_template}
]

# Generate the output
print("\nPhi-3 thinking...")
outputs = pipe(one_shot_prompt)
print(f"\nGenerated reply: {outputs[0]['generated_text']}")


# 4. PREVENT EXIT: Wait for user signal
input(f"\nPress ENTER to close the script and clear GPU memory")