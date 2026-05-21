# Carga desde Hugging Face y usa el modelo GPT-2 de OpenAI
# Basel, 29/Apr/2026
# NVIDIA GeForce RTX 3060 : 67% GPU, 52% RAM (6GB de 12GB), 84W (max 170W), 0% ventilador
# GPT: Generative Pre-trained Transformer
# En lugar de mecanismos de recurrencia o convolución, utiliza "atención", lo que supuso una revolución en 2019

import re
import transformers
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig

def text_cleanup(text):
    # 1. Remove leading/trailing garbage
    text = text.strip()
    
    # 2. Fix tokenization artifacts (e.g., "Hello , world" -> "Hello, world")
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    
    # 3. Fix quotes that are separated from words (e.g., '" Hello "' -> '"Hello"')
    text = re.sub(r'(\s)"\s+', r'\1"', text)
    text = re.sub(r'\s+"(\s)', r'"\1', text)
    
    # 4. Collapse multiple newlines into professional paragraph breaks
    # This turns 3+ newlines into exactly 2.
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 5. Remove "The Nolan Artifacts" (Random punctuation strings like ,,, or ...)
    text = re.sub(r',{2,}', ',', text) # Fixes ,,,
    text = re.sub(r'\.{4,}', '...', text) # Fixes ......
    
    # 6. The "Cliffhanger" Fix: 
    # Base models often stop mid-sentence because they hit max_new_tokens.
    # We find the last sentence-ending punctuation and cut off the trailing fragment.
    last_punctuation = max(text.rfind('.'), text.rfind('?'), text.rfind('!'))
    if last_punctuation != -1:
        text = text[:last_punctuation + 1]
        
    return text


print(f"transformers: {transformers.__version__}")

# Load model OpenAI GPT-2 and tokenizer in the NVIDIA GPU
#MODEL_ID = "openai-community/gpt2-medium"
MODEL_ID = "openai-community/gpt2-xl"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda", torch_dtype="auto")
print(f"\nModel {MODEL_ID} successfully loaded.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print(f"Tokenizer {MODEL_ID} successfully loaded.")

# Create a pipeline (generator)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)
console = Console()
console.print(f"Generator successfully created.", style="gold1")

# Define the generation configuration
gen_config = GenerationConfig(
    max_new_tokens=500,
    # 1. Enable Sampling (Adds randomness to break loops)
    do_sample=True, 
    # 2. Temperature (Higher = more creative, Lower = more focused. 0.7 is a sweet spot)
    temperature=0.7,
    # 3. Top-P / Nucleus Sampling (Ignores low-probability "junk" words)
    top_p=0.9,
    # 4. Repetition Penalty (The most direct fix for your specific issue)
    repetition_penalty=1.2, 
    # 5. No Repeat N-Gram Size (Prevents any 3-word sequence from appearing twice)
    no_repeat_ngram_size=3,
    
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
)
console.print(f"Generation configuration successfully created.", style="gold1")

# GPT-2 is a text-completion model (it just tries to predict the next word)
# To use it, I pass the prompt as a simple string instead of a "chat" list

#prompt = "The sky is"

# Few-shot prompt
prompt = "Fact: The grass is, completion: green. Fact: The sun, completion: shines. Fact: the sky is"

#prompt = "Generate a training plan, 4 weeks long, to run 5 Kilometers"

output = generator(prompt, generation_config=gen_config)

# Clean the output text (Double quotes, newlines, etc. that come from GPT-2 being trained on Web text)
#output_cleaned = output[0]['generated_text'].strip()
#output_cleaned = re.sub(r'\n\s*\n', '\n', output_cleaned)
output_cleaned = text_cleanup(output[0]['generated_text'])

print(f"\nPrompt: {prompt}")
print(f"Generated text: {output_cleaned}")
print()

# Prompting using the "chat" schema doesn't work with GPT-2
#messages = [{"role": "user", "content": "Generate a training plan, 4 weeks long, to run 5 Kilometers"}]
#output = generator(messages, generation_config=gen_config)
#print(output[0]["generated_text"])
#print()