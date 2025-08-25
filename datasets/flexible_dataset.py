from torch.utils.data import Dataset
from PIL import Image
import os, json
import datasets as hf_datasets
import pyarrow as pa
import pyarrow.dataset as pa_ds
import io

class FlexibleDataset(Dataset):
    """
    Supports:
    - Arrow files
    - Hugging Face datasets
    - Local image dirs (plus JSON metadata)
    """
    def __init__(self, config, transform=None):
        self.config = config
        self.transform = transform
        self.src = config["dataset"]["source"]

        if self.src == "arrow":
            path = config["dataset"]["arrow_path"]
            self.arrow_dataset = pa_ds.dataset(path, format="arrow")
            self.image_col = config["dataset"]["image_column"]
            self.instruction_col = config["dataset"].get("instruction_column", "instruction")
            self.key_col = config["dataset"].get("key_column", "key")

            # Preload a table
            self.table = self.arrow_dataset.to_table()

        elif self.src == "huggingface":
            hf_id = config["dataset"]["huggingface_id"]
            split = config["dataset"].get("split", "train")
            self.hf_ds = hf_datasets.load_dataset(hf_id, split=split, trust_remote_code=True)
            self.image_col = config["dataset"]["image_column"]
            self.instruction_col = config["dataset"].get("instruction_column", "instruction")
            self.key_col = config["dataset"].get("key_column", "key")

        else:
            raise NotImplementedError("Support still needed for other sources")

    def __len__(self):
        if self.src in ("arrow",):
            return self.table.num_rows
        elif self.src == "huggingface":
            return len(self.hf_ds)

    def __getitem__(self, idx):
        if self.src == "arrow":
            row = self.table.slice(idx, 1).to_pydict()
            images = row[self.image_col][0]  # list of bytes
            instruction = row[self.instruction_col][0]
            key = row[self.key_col][0]

        else:  # huggingface
            item = self.hf_ds[idx]
            images = item[self.image_col]  # list of PIL images
            instruction = item[self.instruction_col]
            key = item[self.key_col]

        # Convert image bytes to PIL
        pil_images = []
        for img in images:
            if isinstance(img, bytes):
                pil_images.append(Image.open(io.BytesIO(img)).convert("RGB"))
            else:
                pil_images.append(img.convert("RGB"))

        if self.transform:
            pil_images = [self.transform(im) for im in pil_images]

        return pil_images, instruction, key