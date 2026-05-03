# Elimina un warning al usar LoRA adapter (alta precisión float16) con el modelo Tinyllama cuantizado 4-bit (NF4)
# Idem al "apilar" dos modelos LoRA en la misma variable model 
import warnings
warnings.filterwarnings("ignore", message="Merge lora module to 4-bit linear")
warnings.filterwarnings("ignore", message="Already found a .peft_config. attribute")

import sys
import types
import torch


# 1. Create the fake package 'torch.distributed.fsdp'
if "torch.distributed.fsdp" not in sys.modules:
    mock_fsdp = types.ModuleType("torch.distributed.fsdp")
    sys.modules["torch.distributed.fsdp"] = mock_fsdp
    
    # Link it to the existing distributed module if possible
    try:
        import torch.distributed
        torch.distributed.fsdp = mock_fsdp
    except:
        pass

# 2. Create the fake sub-module 'fully_sharded_data_parallel' inside it
path = "torch.distributed.fsdp.fully_sharded_data_parallel"
if path not in sys.modules:
    mock_sub = types.ModuleType(path)
    
    # Define the dummy classes TRL is hunting for
    class DummyClass: pass
    
    # Fill both the package and sub-module with these dummies
    for target in [sys.modules["torch.distributed.fsdp"], mock_sub]:
        setattr(target, "FullyShardedDataParallel", DummyClass)
        setattr(target, "FSDPModule", DummyClass)
        setattr(target, "StateDictType", DummyClass) # TRL often looks for this too
    
    # Link them so 'fsdp.fully_sharded_data_parallel' works as dot notation
    sys.modules["torch.distributed.fsdp"].fully_sharded_data_parallel = mock_sub
    sys.modules[path] = mock_sub

# NOW import TRL
from trl import DPOConfig, DPOTrainer

from rich.console import Console
from datasets import load_dataset
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

def format_prompt(example):
    """Format the prompt to using the <|user|> template TinyLLama is using"""

    # Format answers
    system = "<|system|>\n" + example["system"] + "</s>\n"
    prompt = "<|user|>\n" + example["input"] + "</s>\n<|assistant|>\n"
    chosen = example["chosen"] + "</s>\n"
    rejected = example["rejected"] + "</s>\n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

# Apply formatting to the dataset and select relatively short answers
dpo_dataset = load_dataset(
    "argilla/distilabel-intel-orca-dpo-pairs", split="train"
)
dpo_dataset = dpo_dataset.filter(
    lambda r: 
        r["status"] != "tie" and 
        r["chosen_score"] >= 8 and 
        not r["in_gsm8k_train"]
)
dpo_dataset = dpo_dataset.map(
    format_prompt,  remove_columns=dpo_dataset.column_names
)

console = Console()
console.print(f"\nTrain dataset successfully loaded.", style="gold1")
print(f"Structure: {dpo_dataset}\n")


# 4-bit quantization configuration - Q in QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit precision model loading
    bnb_4bit_quant_type="nf4",  # Quantization type
    bnb_4bit_compute_dtype="bfloat16",  # Compute dtype
    bnb_4bit_use_double_quant=True,  # Apply nested quantization
)

# Merge LoRA and base model
model = AutoPeftModelForCausalLM.from_pretrained(
    "TinyLlama-1.1B-qlora",
    low_cpu_mem_usage=True,
    device_map="auto",
    quantization_config=bnb_config,
)
merged_model = model.merge_and_unload()

# Load LLaMA tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Align config to prevent a warning
merged_model.config.pad_token_id = tokenizer.pad_token_id
if hasattr(merged_model, "generation_config"):
    merged_model.generation_config.pad_token_id = tokenizer.pad_token_id

# Prepare LoRA configuration
peft_config = LoraConfig(
    lora_alpha=32,  # LoRA Scaling
    lora_dropout=0.1,  # Dropout for LoRA Layers
    r=64,  # Rank
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=  # Layers to target
     ["k_proj", "gate_proj", "v_proj", "up_proj", "q_proj", "o_proj", "down_proj"]
)

# prepare model for training
model = prepare_model_for_kbit_training(model)
model_for_dpo = prepare_model_for_kbit_training(merged_model)
# model = get_peft_model(model, peft_config)

output_dir = "./results-2"

# Training arguments
training_arguments = DPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    max_steps=200,
    logging_steps=10,
    bf16=True,                # <--- Set this to True
    fp16=False,               # <--- Set this to False
    gradient_checkpointing=True,
    warmup_steps=0.1,
    beta=0.1,
)

# Create DPO trainer
dpo_trainer = DPOTrainer(
    model=model_for_dpo,
    args=training_arguments,
    train_dataset=dpo_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

# Fine-tune model with DPO
console.print(f"Training starts.\n", style="gold1")
dpo_trainer.train()
console.print(f"\nTraining completed.\n", style="gold1")

# Save adapter
dpo_trainer.model.save_pretrained("TinyLlama-1.1B-dpo-qlora")

# Merge LoRA and base model
model = AutoPeftModelForCausalLM.from_pretrained(
    "TinyLlama-1.1B-qlora",
    low_cpu_mem_usage=True,
    device_map="auto",
)
sft_model = model.merge_and_unload()

# Merge DPO LoRA and SFT model
dpo_model = PeftModel.from_pretrained(
    sft_model,
    "TinyLlama-1.1B-dpo-qlora",
    device_map="auto",
)
dpo_model = dpo_model.merge_and_unload()