import argparse

import torch
import torch.nn as nn
from torch.optim import Adam

from flatten_torch.camera import Camera, euler_rotation
from flatten_torch.data import Batch, corners_on_zplane
from flatten_torch.gaussian_diffusion import diffusion_from_config
from flatten_torch.model import DiffusionPredictor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--diffusion-checkpoint", type=str, default=None)
    parser.add_argument("corners", type=float, nargs="+")
    args = parser.parse_args()

    assert len(args.corners) == 8, "must pass exactly 8 numerical arguments"
    targets = torch.tensor([float(x) for x in args.corners], device=device).view(4, 2)

    if args.diffusion_checkpoint is None:
        batch = Batch.sample_batch(args.batch_size, device=device)
        origin = nn.Parameter(batch.origin)
        size = nn.Parameter(batch.size)
        rotation = nn.Parameter(batch.rotation)
        translation = nn.Parameter(batch.translation)
        post_translation = nn.Parameter(batch.post_translation)
    else:
        model = DiffusionPredictor(device=device)
        diffusion = diffusion_from_config(
            dict(
                schedule="linear",
                timesteps=1024,
                respacing="128",
            )
        )
        with open(args.diffusion_checkpoint, "rb") as f:
            obj = torch.load(f, map_location=device)
            model.load_state_dict(obj["model"])

        sample = diffusion.p_sample_loop(
            model,
            shape=(args.batch_size, model.d_input),
            clip_denoised=False,
            model_kwargs=dict(cond=targets.view(1, -1).repeat(args.batch_size, 1)),
        )
        origin, size, rotation, translation = [
            nn.Parameter(x) for x in torch.split(sample, [3, 2, 3, 3], dim=-1)
        ]

    origin = nn.Parameter(origin[:, :2].detach())

    opt = Adam([origin, size, rotation, translation, post_translation], lr=args.lr)

    for i in range(args.iters):
        corners = corners_on_zplane(
            torch.cat([origin, torch.zeros_like(origin[:, :1])], dim=-1), size
        )
        camera = Camera(
            rotation=euler_rotation(rotation),
            translation=translation,
            post_translation=post_translation,
        )
        proj = camera.project(corners).projected
        losses = (proj - targets).pow(2).flatten(1).sum(-1)
        loss = losses.sum()
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
        f" post_translation={post_translation[best_idx].tolist()}"
    )


if __name__ == "__main__":
    main()
