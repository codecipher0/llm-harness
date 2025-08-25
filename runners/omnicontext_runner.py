import os
import io
from datasets import load_dataset
from PIL import Image

from models.omnigen2_model import OmniGen2Model
from utils.config_parser import load_config


def load_dataset_from_yaml(cfg):
    ds_cfg = cfg["dataset"]
    source = ds_cfg.get("source")
    path = ds_cfg.get("huggingface_id")
    split = ds_cfg.get("split", "test")
    num_samples = ds_cfg.get("num_samples", 5)
    trust_remote_code = ds_cfg.get("trust_remote_code", False)

    if source == "huggingface":
        ds = load_dataset(
            path, 
            split=split,
            trust_remote_code=trust_remote_code
        )
        return ds.select(range(min(num_samples, len(ds))))
    elif source == "arrow":
        ds = load_dataset("arrow", data_files=path, split=split)
        return ds.select(range(min(num_samples, len(ds))))
    elif source == "jsondir":
        json_path = os.path.join(path, ds_cfg["json_file"])
        ds = load_dataset("json", data_files=json_path, split="test")
        return ds.select(range(min(num_samples, len(ds))))
    elif source == "local":
        raise NotImplementedError("Direct local image dir loader not implemented yet")
    else:
        raise ValueError(f"Unknown dataset source: {source}")


def run_inference(model_cfg_path, dataset_cfg_path, output_dir="./outputs/"):
    # Load model
    model_cfg = load_config(model_cfg_path)
    model = OmniGen2Model()
    model.load(model_cfg)

    # Load dataset
    ds_cfg = load_config(dataset_cfg_path)
    ds = load_dataset_from_yaml(ds_cfg)
    print(f"Loaded dataset with {len(ds)} samples.")

    for idx, item in enumerate(ds):
        images = item.get("input_images")
        instruction = item.get("instruction", "")
        key = item.get("key", f"sample_{idx}")

        print(f"[{idx}] Key: {key}, Instruction: {instruction}")

        generated = model.infer(images, instruction)

        # Save output
        if isinstance(generated, (bytes, bytearray)):
            img = Image.open(io.BytesIO(generated)).convert("RGB")
        elif isinstance(generated, Image.Image):
            img = generated
        else:
            img = generated  # fallback

        os.makedirs(output_dir, exist_ok=True)
        for i,image in enumerate(img):
            out_path = os.path.join(output_dir, f"{key}_{i}.png")
            img.save(out_path)
            print(f" → Saved to {out_path}")

    model.unload()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/")
    args = parser.parse_args()

    run_inference(
        model_cfg_path=args.model_config,
        dataset_cfg_path=args.dataset_config,
        output_dir=args.output_dir,
    )