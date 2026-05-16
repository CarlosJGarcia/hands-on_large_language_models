# Modificación de 12-01.py para usar Phi3 en vez de tiny-llama

# GGUF Cannot Be Fine-Tuned directly via SFTTrainer
# You cannot use SFTTrainer / get_peft_model) with a .gguf file loaded via LlamaCpp.

# GGUF is a highly compressed, static format designed exclusively for inference (running models). 
# Hugging Face's SFTTrainer and PyTorch require unquantized, floating-point weights (like safetensors or bin formats) to compute gradients and perform backpropagation.
# Trying to attach a PyTorch LoRA adapter to a LlamaCpp instance will fail instantly.