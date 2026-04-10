# Librería transformers de Hugging Face
# Modelo DeBERTa v3
# DeBERTa = Decoded BERT with attention - (De)coded BERT with (a)ttention
# Basel 10/Apr/2026

import transformers
from transformers import AutoModel, AutoTokenizer

print(f"Transformers version:     {transformers.__version__}")

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")

# Load a language model
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall", use_safetensors=True)

# Tokenize the sentence
tokens = tokenizer('Hello world', return_tensors='pt')

# Process the tokens
output = model(**tokens)[0]

print("\n--- ¡Éxito! ---")
# Forma del tensor de salida: debería imprimir algo como [1, 4, 384] que significa: (1 frase, 4 tokens, 384 dimensiones del modelo xsmall)
print(f"Forma del tensor de salida: {output.shape}")
print()

for token in tokens['input_ids'][0]:
    print(tokenizer.decode(token))
