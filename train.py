from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import os
import argparse
import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

def main(args):
    # Authenticate with Hugging Face early to fail fast if missing
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set. Please set it in a .env file or environment.")
    login(token=hf_token)

    # 2. Load the Model
    print(f"Loading model: {args.base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.base_model,
        max_seq_length = args.max_seq_length,
        load_in_4bit = False,
        dtype = torch.bfloat16,
    )

    # 3. Attach the LoRA Adapters
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

    # 4. Format the Dataset
    tokenizer = get_chat_template(
        tokenizer, 
        chat_template = args.chat_template
    )

    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False)
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    dataset = load_dataset("json", data_files={"train": args.dataset}, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True)

    # 5. Configure the Trainer
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,  # <-- FIXED: Renamed from 'tokenizer' to 'processing_class'
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        dataset_num_proc = 2,
        args = SFTConfig(
            per_device_train_batch_size = 2, 
            gradient_accumulation_steps = 4,
            warmup_steps = 2,
            num_train_epochs = args.epochs,
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

    # 6. Start Training
    print("Starting training...")
    trainer.train()

    # 7. Save and Export
    print(f"Merging weights and exporting to GGUF in ./{args.output_dir}...")
    model.save_pretrained_gguf(args.output_dir, tokenizer, quantization_method = args.quantization)

    # 8. Upload to Hugging Face
    repo_id = f"{args.hf_username}/{args.hf_repo}"
    api = HfApi()

    print(f"Creating repository: {repo_id}...")
    api.create_repo(
        repo_id=repo_id, 
        repo_type="model", 
        private=True, 
        exist_ok=True
    )

    print(f"Uploading the contents of ./{args.output_dir} to Hugging Face...")
    api.upload_folder(
        folder_path=args.output_dir,
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"\nSuccess! Your model is now securely stored at:")
    print(f"https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and upload a GGUF model")
    
    # Model & Dataset arguments
    parser.add_argument("--base_model", type=str, default="unsloth/gemma-4-26B-A4B-it", help="Base model to train")
    parser.add_argument("--chat_template", type=str, default="gemma-4", help="Chat template to apply")
    parser.add_argument("--dataset", type=str, default="dataset.jsonl", help="Path to the training dataset")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    # Export arguments
    parser.add_argument("--output_dir", type=str, default="gemma4_native_agent", help="Local folder to save GGUF files")
    parser.add_argument("--quantization", type=str, default="q4_k_m", help="GGUF quantization method")
    
    # Hugging Face arguments
    parser.add_argument("--hf_username", type=str, required=True, help="Your Hugging Face username")
    parser.add_argument("--hf_repo", type=str, required=True, help="Destination repository name")
    
    args = parser.parse_args()
    main(args)
