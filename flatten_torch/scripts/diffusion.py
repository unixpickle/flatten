import os

import torch
import torch.nn as nn
import torch.optim as optim

from flatten_torch.data import Batch
from flatten_torch.gaussian_diffusion import diffusion_from_config
from flatten_torch.model import DiffusionPredictor

BATCH_SIZE = 50000
SAVE_INTERVAL = 5000
SAVE_PATH = "diffusion_model.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionPredictor(device=device)
    diffusion = diffusion_from_config(
        dict(
            schedule="linear",
            timesteps=1024,
        )
    )
    opt = optim.Adam(params=model.parameters(), lr=1e-3)
    gen = torch.Generator(device=device)
    iter = 0

    if os.path.exists(SAVE_PATH):
        print(f"loading from {SAVE_PATH}")
        with open(SAVE_PATH, "rb") as f:
            obj = torch.load(f, map_location=device)
            gen.set_state(obj["gen"])
            iter = obj["iter"]
            opt.load_state_dict(obj["opt"])
            model.load_state_dict(obj["model"])

    while True:
        batch = Batch.sample_batch(BATCH_SIZE, generator=gen, device=device)
        model_kwargs = dict(cond=batch.proj_corners.flatten(1))
        target = torch.cat(
            [batch.origin, batch.size, batch.rotation, batch.translation], dim=-1
        )
        losses = diffusion.training_losses(
            model=model,
            x_start=target,
            t=torch.randint(
                low=0,
                high=diffusion.num_timesteps,
                size=(len(batch),),
                generator=gen,
                device=device,
            ),
            noise=torch.randn(target.shape, device=device, generator=gen),
            model_kwargs=model_kwargs,
        )
        loss = losses["loss"].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"iter={iter} loss={loss.item()}")
        iter += 1
        if iter % SAVE_INTERVAL == 0:
            with open(SAVE_PATH, "wb") as f:
                torch.save(
                    dict(
                        opt=opt.state_dict(),
                        model=model.state_dict(),
                        gen=gen.get_state(),
                        iter=iter,
                    ),
                    f,
                )


if __name__ == "__main__":
    main()
