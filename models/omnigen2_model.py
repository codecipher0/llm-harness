from .base_model import BaseModel
import torch

import sys
import os

from accelerate import Accelerator

sys.path.append(os.path.abspath("third_party/OmniGen2"))

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline, OmniGen2Transformer2DModel

class OmniGen2Model(BaseModel):
    def load(self, config: dict):
        weight_dtype = torch.float32
        precision = config["model"].get("precision", "fp32")
        accelerator = Accelerator(mixed_precision=precision if precision != 'fp32' else 'no')

        if precision == 'fp16':
            weight_dtype = torch.float16
        elif precision == 'bf16':
            weight_dtype = torch.bfloat16

        #device=config["model"].get("device", "cuda"),
        self.pipeline = OmniGen2Pipeline.from_pretrained(
            config["model"]["hf_model_id"],
            torch_dtype=weight_dtype,
            trust_remote_code=True 
        )
        self.pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            config["model"]["hf_model_id"],
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )
        self.pipeline.to(accelerator.device)

    def infer(self, input_images, instruction=None):
        result = self.pipeline(prompt=instruction, input_images=input_images, output_type="pil")
        #return result
        #result = self.pipeline(images=input_images, prompt=instruction, return_image=True)
        return result.images

    def unload(self):
        del self.pipeline
        torch.cuda.empty_cache()