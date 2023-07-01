"""
Train a model to predict the original aspect ratio of photos given a square resize of them.
"""

import argparse
import glob
import math
import os
import random
from typing import Iterator, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
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

    test_data = iterate_data(
        StretchDataset(args.image_dir, valid=True), batch_size=args.batch_size
    )
    train_data = iterate_data(
        StretchDataset(args.image_dir), batch_size=args.batch_size
    )

    model = StretchPredictor(device=device)
    print(f"total of {sum(x.numel() for x in model.parameters())} parameters.")
    opt = Adam(model.parameters(), lr=args.lr)

    i = 0
    while True:
        train_x, train_y = next(train_data)
        test_x, test_y = next(test_data)
        i += 1
        loss = model.losses(train_x.to(device), train_y.to(device)).mean()
        with torch.no_grad():
            test_loss = model.losses(test_x.to(device), test_y.to(device)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {i}: loss={loss.item():.05} test={test_loss.item():.05}")
        if i % args.save_interval == 0:
            print(f"saving to {args.checkpoint}...")
            with open(args.checkpoint, "wb") as f:
                torch.save(model.state_dict(), f)


def iterate_data(
    dataset: Dataset, batch_size: int
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=2, worker_init_fn=_seed_worker
    )
    while True:
        yield from iter(loader)


def _seed_worker(worker_id: int):
    seed = worker_id + np.random.randint(2**30)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class StretchDataset(Dataset):
    def __init__(self, directory: str, valid: bool = False, num_valid: int = 10000):
        self.image_paths = sorted(glob.glob(os.path.join(directory, "*.jpg")))
        random.Random(1337).shuffle(self.image_paths)
        if valid:
            self.image_paths = self.image_paths[:num_valid]
        else:
            self.image_paths = self.image_paths[num_valid:]
        self.resize = Resize((64, 64))
        self.torchify = ToTensor()

    def __getitem__(self, index: int):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        image = Image.open(self.image_paths[index]).convert("RGB")
        image = random_crop(image)

        orig_shape = self.torchify(image).shape
        img = self.torchify(self.resize(image))
        size = orig_shape[1] / orig_shape[2]
        return img, torch.tensor(size, dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)


def random_crop(image: Image.Image, min_frac: float = 0.9) -> Image.Image:
    w, h = image.size
    new_h = np.random.randint(math.ceil(h * min_frac), h + 1)
    new_w = np.random.randint(math.ceil(w * min_frac), w + 1)
    new_y = np.random.randint(0, h - new_h + 1)
    new_x = np.random.randint(0, w - new_w + 1)
    return image.crop((new_x, new_y, new_x + new_w, new_y + new_h))


if __name__ == "__main__":
    main()
