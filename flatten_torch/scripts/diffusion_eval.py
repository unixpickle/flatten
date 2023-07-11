import os

import torch
import torch.nn as nn
import torch.optim as optim

from flatten_torch.camera import Camera, euler_rotation
from flatten_torch.data import Batch, corners_on_zplane
from flatten_torch.gaussian_diffusion import diffusion_from_config
from flatten_torch.model import DiffusionPredictor

LOAD_PATH = "diffusion_model.pt"
BATCH_SIZE = 32


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionPredictor(device=device)
    diffusion = diffusion_from_config(
        dict(
            schedule="linear",
            timesteps=1024,
            respacing="128",
        )
    )
    with open(LOAD_PATH, "rb") as f:
        obj = torch.load(f, map_location=device)
        model.load_state_dict(obj["model"])

    batch = Batch.sample_batch(BATCH_SIZE, device=device)

    sample = diffusion.p_sample_loop(
        model,
        shape=(BATCH_SIZE, model.d_input),
        clip_denoised=False,
        model_kwargs=dict(cond=batch.proj_corners.flatten(1)),
    )
    origin, size, rotation, translation, post_translation = torch.split(
        sample, [3, 2, 3, 3, 2], dim=-1
    )
    camera = Camera(
        rotation=euler_rotation(rotation),
        translation=translation,
        post_translation=post_translation,
    )
    corners = corners_on_zplane(origin, size)
    proj = camera.project(corners)
    reproj_err = (proj.projected - batch.proj_corners).pow(2).mean()
    print(f"mse={reproj_err.item()}")


if __name__ == "__main__":
    main()
