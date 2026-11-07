from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 1. Load the Model
max_seq_length = 4096
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-4-26B-A4B-it",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    dtype = torch.bfloat16,
)

# 2. Attach the LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. Format the Dataset
tokenizer = get_chat_template(
    tokenizer, 
    chat_template = "gemma-4"
)

def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False)
        for convo in examples["conversations"]
    ]
    return {"text": texts}

dataset = load_dataset("json", data_files={"train": "dataset.jsonl"}, split="train")

dataset = dataset.map(formatting_prompts_func, batched = True)

# 4. Configure the Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = SFTConfig(
        per_device_train_batch_size = 2, 
        gradient_accumulation_steps = 4,
        warmup_steps = 2,
        num_train_epochs = 3,
        learning_rate = 5e-5,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 5. Start Training
print("Starting training...")
trainer_stats = trainer.train()

# 6. Save and Export
print("Merging weights and exporting to GGUF...")
model.save_pretrained_gguf("gemma4_native_agent", tokenizer, quantization_method = "q4_k_m")
