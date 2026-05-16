import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    pipeline
)

# ENVIRONMENT & HARDWARE OPTIMIZATIONS
# Disable weights & biases logging if you don't have an account
os.environ["WANDB_DISABLED"] = "true" 

# Verify RTX 5060 can use bfloat16 (Ampere/Blackwell supports this natively)
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"Using compute dtype: {compute_dtype}")

# LOAD AND PREPROCESS DATASET (GLUE MNLI)
print("Loading GLUE MNLI dataset...")
# Slicing down to 50,000 samples to keep training fast while learning
dataset = load_dataset("glue", "mnli", split="train").select(range(50_000))
dataset = dataset.remove_columns("idx")

# LOAD THE BASE MODEL & TOKENIZER IN PURE 16-BIT
model_id = "microsoft/Phi-3-mini-4k-instruct"

print(f"Loading tokenizer and model: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Standard for sequence classification

# Phi-3 has 3 labels for MNLI: 0=entailment, 1=neutral, 2=contradiction
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=3,
    torch_dtype=compute_dtype,
    device_map="auto",
    trust_remote_code=True
)

# CONFIGURING PURE LORA (NO QUANTIZATION)
# Rank=16 keeps optimizer memory overhead small enough to fit into 16GB VRAM
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS", # Sequence Classification task type
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"] # Explicit Phi-3 layers
)

# Inject the LoRA adapters into the 16-bit model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# CONFIGURE THE TRAINING ARGUMENTS into 16GB VRAM
output_dir = "./lora_phi3_results"

sft_config = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,     # Strict limit of 2 to avoid OOM
    gradient_accumulation_steps=4,    # Simulates a total batch size of 8 (2x4)
    optim="adamw_torch",              # Clean standard optimizer since we aren't using k-bit/paged
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    logging_steps=50,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,      # CRITICAL: Saves immense VRAM by recalculating activation layers
    max_seq_length=512,               # Strict context cap to protect your 16GB ceiling
)

# INITIALIZE TRAINER AND RUN Fine-Tuning
def formatting_func(example):
    # Combines premise and hypothesis into a structured text string for the classifier
    return f"Premise: {example['premise']} Hypothesis: {example['hypothesis']}"

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    packing=False,
    formatting_func=formatting_func,
)

print("🚀 Starting 16-bit LoRA training loop on Phi-3...")
trainer.train()

# 7. SAVE THE LORA ADAPTER WEIGHTS
adapter_path = "./phi3-mnli-lora-adapter"
trainer.model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"✅ LoRA adapter successfully saved to {adapter_path}")

# 8. MERGE ADAPTER AND BASE MODEL FOR INFERENCE
print("🔄 Merging LoRA weights back into the base model...")
# Unload current training graph, reload base model cleanly, and merge
final_merged_model = trainer.model.merge_and_unload()

# 9. RUN A TEST PREDICTION
print("\n🔮 Running verification test on the newly trained model...")
classifier = pipeline(task="text-classification", model=final_merged_model, tokenizer=tokenizer)

# Example text that should trigger 'contradiction' (Label_2)
test_prompt = "Premise: A man is sleeping on a couch. Hypothesis: A man is running a marathon outside."
result = classifier(test_prompt)

print(f"Test Input: {test_prompt}")
print(f"Model Prediction: {result}")