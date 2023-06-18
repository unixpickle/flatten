import argparse

import torch
import torch.nn as nn
from torch.optim import Adam

from flatten_torch.camera import Camera, euler_rotation
from flatten_torch.data import Batch, corners_on_zplane


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("corners", type=float, nargs="+")
    args = parser.parse_args()

    assert len(args.corners) == 8, "must pass exactly 8 numerical arguments"
    targets = torch.tensor([float(x) for x in args.corners], device=device).view(4, 2)

    batch = Batch.sample_batch(args.batch_size, device=device)

    origin = nn.Parameter(batch.origin)
    size = nn.Parameter(batch.size)
    rotation = nn.Parameter(batch.rotation)
    translation = nn.Parameter(batch.translation)

    opt = Adam([origin, size, rotation, translation], lr=args.lr)

    for i in range(args.iters):
        corners = corners_on_zplane(origin, size)
        camera = Camera(rotation=euler_rotation(rotation), translation=translation)
        proj = camera.project(corners).projected
        losses = (proj - targets).pow(2).flatten(1).sum(-1)
        loss = losses.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {i}: loss={loss.item()} best={losses.min().item()}")

    best_idx = torch.argmin(losses)
    print(f"best loss: {losses[best_idx].item()}")
    print(
        f"origin={origin[best_idx].tolist()}"
        f" size={size[best_idx].tolist()}"
        f" rotation={rotation[best_idx].tolist()}"
        f" translation={translation[best_idx].tolist()}"
    )


if __name__ == "__main__":
    main()
