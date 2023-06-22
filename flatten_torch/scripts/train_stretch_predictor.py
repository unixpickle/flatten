"""
Train a model to predict the original aspect ratio of photos given a square resize of them.
"""

import argparse
import glob
import os
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, ToTensor

from flatten_torch.model import StretchPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, default="stretch_predictor.pt")
    parser.add_argument("image_dir", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = StretchDataset(args.image_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    model = StretchPredictor(device=device)
    print(f"total of {sum(x.numel() for x in model.parameters())} parameters.")
    opt = Adam(model.parameters(), lr=args.lr)

    i = 0
    while True:
        for inputs, targets in loader:
            i += 1
            loss = model.losses(inputs, targets).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(f"step {i}: loss={loss.item():.05}")
            if i % args.save_interval == 0:
                print(f"saving to {args.checkpoint}...")
                with open(args.checkpoint, "wb") as f:
                    torch.save(model.state_dict(), f)


class StretchDataset(Dataset):
    def __init__(self, directory: str):
        self.image_paths = glob.glob(os.path.join(directory, "*.jpg"))
        self.resize = Resize((64, 64))
        self.torchify = ToTensor()

    def __getitem__(self, index: int):
        image = Image.open(self.image_paths[index])
        orig_shape = self.torchify(image).shape
        img = self.torchify(self.resize(image))
        size = orig_shape[1] / orig_shape[0]
        return img, torch.tensor(size, dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    main()
