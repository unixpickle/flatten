"""
Train a model to predict the original aspect ratio of photos given a square resize of them.
"""

import argparse
import glob
import math
import os
import random
from typing import Iterator, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter, GaussianBlur, Resize, ToTensor

from flatten_torch.model import StretchPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default="stretch_predictor.pt")
    parser.add_argument("image_dir", type=str, nargs="+")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = StretchDataset(args.image_dir, valid=True)
    train_ds = StretchDataset(args.image_dir)
    test_data = iterate_data(test_ds, batch_size=args.batch_size)
    train_data = iterate_data(train_ds, batch_size=args.batch_size)
    print(f"train images: {len(train_ds)}")
    print(f"test images: {len(test_ds)}")

    model = StretchPredictor(device=device)
    if os.path.exists(args.checkpoint):
        print(f"loading from {args.checkpoint}...")
        torch.save(model.state_dict(), args.checkpoint)

    print(f"total of {sum(x.numel() for x in model.parameters())} parameters.")
    opt = Adam(model.parameters(), lr=args.lr)

    i = 0
    test_losses = []
    best_test_loss = None

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
        print(
            f"step {i}: loss={loss.item():.05} test={test_loss.item():.05}"
            f" best_test={(best_test_loss if best_test_loss is not None else test_loss):.05}"
        )
        test_losses.append(test_loss.item())
        if len(test_losses) > args.save_interval:
            del test_losses[0]
        elif len(test_losses) < args.save_interval:
            continue
        mean_test_loss = np.mean(test_losses)
        if best_test_loss is None or mean_test_loss < best_test_loss:
            best_test_loss = mean_test_loss
            test_losses.clear()  # don't save too frequently; reset the counter.
            print(f"saving to {args.checkpoint}...")
            torch.save(model.state_dict(), args.checkpoint)


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
    def __init__(
        self, directories: Sequence[str], valid: bool = False, num_valid: int = 10000
    ):
        self.image_paths = []
        for directory in directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    if os.path.splitext(file)[1].lower() in {".jpg", ".jpeg"}:
                        self.image_paths.append(os.path.join(root, file))
        self.image_paths.sort()
        random.Random(1337).shuffle(self.image_paths)
        if valid:
            self.image_paths = self.image_paths[:num_valid]
        else:
            self.image_paths = self.image_paths[num_valid:]
        self.torchify = ToTensor()
        self.augment = (
            nn.Identity()
            if valid
            else nn.Sequential(
                GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.0)),
                ColorJitter(brightness=0.2, hue=0.2),
            )
        )

    def __getitem__(self, index: int):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        image = Image.open(self.image_paths[index]).convert("RGB")
        image = random_crop(image)

        orig_shape = self.torchify(image).shape
        ratio = orig_shape[1] / orig_shape[2]

        img = self.augment(self.torchify(image))
        # Resizing with torch is and then average pooling is
        # easier to emulate in JavaScript than resizing with
        # Pillow's more sophisticated antialiasing.
        img = F.interpolate(img[None], (128, 128), mode="bilinear")
        img = F.avg_pool2d(img, 2, 2)[0]

        return img, torch.tensor(ratio, dtype=torch.float32)

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
