from torch.utils.data import Dataset
from PIL import Image
import os, json
import datasets as hf_datasets
import pyarrow.parquet as pq
import numpy as np

class FlexibleDataset(Dataset):
    """
    Flexible dataset loader:
    - local: directory of images
    - json: directory + metadata.json mapping inputs to targets
    - huggingface: load from HF datasets
    - arrow: load from Arrow/Parquet file
    """

    def __init__(self, config, transform=None):
        self.config = config
        self.transform = transform
        self.source = config["dataset"]["source"]

        if self.source == "local":
            self.root_dir = config["dataset"]["path"]
            self.image_files = sorted(os.listdir(self.root_dir))

        elif self.source == "json":
            self.root_dir = config["dataset"]["path"]
            with open(config["dataset"]["json_file"], "r") as f:
                self.metadata = json.load(f)

        elif self.source == "huggingface":
            hf_id = config["dataset"]["huggingface_id"]
            split = config["dataset"].get("split", "train")
            self.dataset = hf_datasets.load_dataset(hf_id, split=split)
            self.image_col = config["dataset"].get("image_column", "image")
            self.target_col = config["dataset"].get("target_column", None)

        elif self.source == "arrow":
            self.table = pq.read_table(config["dataset"]["arrow_file"])
            self.image_col = config["dataset"].get("image_column", "image")
            self.target_col = config["dataset"].get("target_column", None)
            self.table_np = self.table.to_pandas()

        else:
            raise ValueError(f"Unknown dataset source: {self.source}")

    def __len__(self):
        if self.source == "local":
            return len(self.image_files)
        elif self.source == "json":
            return len(self.metadata)
        elif self.source == "huggingface":
            return len(self.dataset)
        elif self.source == "arrow":
            return len(self.table_np)

    def __getitem__(self, idx):
        if self.source == "local":
            img_path = os.path.join(self.root_dir, self.image_files[idx])
            image = Image.open(img_path).convert("RGB")
            target = image  # identity if no labels

        elif self.source == "json":
            entry = self.metadata[idx]
            img_path = os.path.join(self.root_dir, entry["input"])
            tgt_path = os.path.join(self.root_dir, entry.get("target", entry["input"]))
            image = Image.open(img_path).convert("RGB")
            target = Image.open(tgt_path).convert("RGB")

        elif self.source == "huggingface":
            item = self.dataset[idx]
            image = item[self.image_col]
            target = item[self.target_col] if self.target_col else image
            if hasattr(image, "convert"):  # PIL
                image = image.convert("RGB")
            if hasattr(target, "convert"):
                target = target.convert("RGB")

        elif self.source == "arrow":
            row = self.table_np.iloc[idx]
            image = row[self.image_col]
            target = row[self.target_col] if self.target_col else image
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image)).convert("RGB")
            if isinstance(target, bytes):
                target = Image.open(io.BytesIO(target)).convert("RGB")

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target