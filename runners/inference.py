from torch.utils.data import DataLoader
import os

class InferenceRunner:
    def __init__(self, model, dataset, output_dir="results/"):
        self.model = model
        self.dataset = dataset
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self):
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        for idx, (input_img, _) in enumerate(dataloader):
            output_img = self.model.infer(input_img)
            # Save or visualize (here we just print for simplicity)
            print(f"[INFO] Processed image {idx}, output shape: {output_img.shape}")