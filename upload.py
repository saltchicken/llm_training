import os
from huggingface_hub import HfApi, login

# 1. Authenticate with your Hugging Face token
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN environment variable is not set. Please set it before running the script.")

login(token=hf_token)

# 2. Define your repository details
username = "saltchicken"         # Replace with your Hugging Face username
repo_name = "first"      # The name you want for the model repository
folder_to_upload = "./gemma4_native_agent_gguf" # The local folder Unsloth saved to

repo_id = f"{username}/{repo_name}"
api = HfApi()

print(f"Creating repository: {repo_id}...")
# Create a private repository (exist_ok=True prevents errors if it already exists)
api.create_repo(
    repo_id=repo_id, 
    repo_type="model", 
    private=True, 
    exist_ok=True
)

print(f"Uploading the contents of {folder_to_upload} to Hugging Face...")
print("This may take a few minutes for a 16GB folder...")

# 3. Upload the entire folder
api.upload_folder(
    folder_path=folder_to_upload,
    repo_id=repo_id,
    repo_type="model",
)

print(f"\nSuccess! Your model is now securely stored at:")
print(f"https://huggingface.co/{repo_id}")
