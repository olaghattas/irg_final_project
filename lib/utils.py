
import os
from datasets import load_dataset, load_from_disk, DatasetDict

def get_dataset(config):
    # returns the dataset. if its doesnt already exist it downloads it in dataset folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(project_root, "dataset")
    save_dir = os.path.join(dataset_dir, f"LitSearch_{config}")

    os.makedirs(dataset_dir, exist_ok=True)

    try:
        dataset = load_from_disk(save_dir)
        print(f"Loaded {config} from disk: {save_dir}")
    except Exception:
        print(f"{save_dir} not found; downloading from Hugging Face...")
        dataset = load_dataset("princeton-nlp/LitSearch", config)
        dataset.save_to_disk(save_dir)
        print(f"Saved dataset to {save_dir}")

    return dataset


def create_subset_dataset(config="LitSearch_query", size=100):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(project_root, f"dataset/{config}")

    # Load dataset (could be Dataset or DatasetDict)
    dataset = load_from_disk(dataset_path)

    # Detect if it's a DatasetDict or Dataset
    ## to preserve same format of dataset
    if isinstance(dataset, DatasetDict):
        split_name = list(dataset.keys())[0]
        data_split = dataset[split_name]

        # Random sample
        subset = data_split.shuffle(seed=42).select(range(min(size, len(data_split))))

        # Keep same format
        subset_to_save = DatasetDict({split_name: subset})
    else:
        subset = dataset.shuffle(seed=42).select(range(min(size, len(dataset))))
        subset_to_save = subset

    # Save in same format as original
    save_path = os.path.join(project_root, f"dataset/{config}_subset{size}")
    subset_to_save.save_to_disk(save_path)

    print(f"Saved subset ({len(subset)} rows) in same format as {config} to {save_path}")
    return subset_to_save


# python3 -c "from lib.utils import create_subset_dataset; create_subset_dataset('LitSearch_corpus_clean')"