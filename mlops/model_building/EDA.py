# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Shalyn/predictiveMaintanence/engine_data.csv"
engine_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
data = engine_dataset.copy()
