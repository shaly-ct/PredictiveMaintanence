from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="mlops/deployment",
    repo_id="Shalyn/PredictiveMaintanence",
    repo_type = "space",
)
