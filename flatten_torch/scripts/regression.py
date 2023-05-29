import torch
import torch.nn as nn
import torch.optim as optim

from flatten_torch.data import Batch

BATCH_SIZE = 10000


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(8, 256),
        nn.GELU(),
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, 11),  # origin(3) + size(2) + rotation(3) + translation(3)
    ).to(device)
    opt = optim.Adam(params=model.parameters(), lr=1e-3)

    iter = 0
    while True:
        batch = Batch.sample_batch(BATCH_SIZE, device=device)
        output = model(batch.proj_corners.flatten(1))
        origin, size, rotation, translation = torch.split(output, [3, 2, 3, 3], dim=-1)
        mse = (
            (origin - batch.origin).pow(2).sum(-1)
            + (size - batch.size).pow(2).sum(-1)
            + (rotation - batch.rotation).pow(2).sum(-1)
            + (translation - batch.translation).pow(2).sum(-1)
        ).mean()
        opt.zero_grad()
        mse.backward()
        opt.step()
        print(f"iter={iter} loss={mse.item()}")
        iter += 1


if __name__ == "__main__":
    main()
