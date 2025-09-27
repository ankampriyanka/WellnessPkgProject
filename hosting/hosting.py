from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN")) # Use HF_TOKEN from environment variables

repo_id = "Priyanka-Ankam/WellnessPkgProject" # Your Hugging Face Space repo ID
repo_type = "space" # Repo type for Hugging Face Spaces

print(f"Uploading deployment files to Hugging Face Space '{repo_id}'...")

try:
    api.upload_folder(
        folder_path="tourism_project/deployment",     # the local folder containing your deployment files
        repo_id=repo_id,          # the target repo
        repo_type=repo_type,                      # dataset, model, or space
        path_in_repo="",                          # optional: subfolder path inside the repo
    )
    print("Deployment files uploaded successfully.")
except Exception as e:
    print(f"Error uploading deployment files: {e}")
    print("Please ensure your Hugging Face Space exists and your HF_TOKEN is correctly set in Colab secrets with write access.")
