from datasets import load_dataset
import os
from pathlib import Path

project_root = os.path.abspath(os.path.join(Path.cwd(), "..", ".."))

configs = ["query", "corpus_clean", "corpus_s2orc"]


# Loop through and download each if not already saved
for config in configs:
    folder_name = f"LitSearch_{config}"
    save_path = os.path.join(project_root, "dataset", folder_name)
    
    if os.path.exists(save_path):
        print(f" {config} already downloaded at {save_path}, skipping.\n")
        continue  # Skip download if directory already exists
    
    print(f" Downloading configuration: {config}")
    dataset = load_dataset("princeton-nlp/LitSearch", config)
    dataset.save_to_disk(save_path)
    print(f" Saved {config} to {save_path}\n")