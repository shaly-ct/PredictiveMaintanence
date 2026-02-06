from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os
import sys

repo_id = "Shalyn/predictiveMaintanence"
repo_type = "dataset"
data_path = "mlops/data"

token = os.getenv("HF_TOKEN")


api = HfApi(token=token)

# Optional but very helpful
api.whoami()

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        exist_ok=True,
        token=token,
    )
    print(f"Dataset '{repo_id}' created.")

api.upload_folder(
    folder_path=data_path,
    repo_id=repo_id,
    repo_type=repo_type,
)
