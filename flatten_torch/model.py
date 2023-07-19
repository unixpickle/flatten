import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from flatten_torch.data import Batch


class DiffusionPredictor(nn.Module):
    def __init__(
        self,
        device: torch.device,
        d_cond: int = 8,
        d_input: int = 13,
        d_model: int = 192,
        pos_emb_feats: int = 0,
    ):
        super().__init__()
        self.device = device
        self.d_cond = d_cond
        self.d_input = d_input
        self.d_model = d_model
        self.pos_emb_feats = pos_emb_feats
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model, device=device),
            nn.ReLU(),
            nn.Linear(d_model, d_model, device=device),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(d_cond * (1 + pos_emb_feats), d_model, device=device),
            nn.ReLU(),
            nn.Linear(d_model, d_model, device=device),
        )
        self.input_embed = nn.Sequential(
            nn.Linear(d_input, d_model, device=device),
            nn.ReLU(),
            nn.Linear(d_model, d_model, device=device),
        )
        self.backbone = nn.Sequential(
            nn.Linear(d_model, d_model, device=device),
            nn.ReLU(),
            nn.Linear(d_model, d_model, device=device),
            nn.ReLU(),
            nn.Linear(d_model, d_model, device=device),
            nn.ReLU(),
            nn.Linear(d_model, d_model, device=device),
            nn.ReLU(),
            nn.Linear(d_model, d_model, device=device),
            nn.ReLU(),
            nn.Linear(d_model, d_input * 2, device=device),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, *, cond: torch.Tensor
    ) -> torch.Tensor:
        time_emb = self.time_embed(timestep_embedding(t, self.d_model))
        input_emb = self.input_embed(x)
        cond_emb = self.cond_embed(frequency_pos_embedding(cond, self.pos_emb_feats))
        return self.backbone((time_emb + input_emb + cond_emb) / math.sqrt(3))


@dataclass
class DiffusionPrediction:
    origin: torch.Tensor  # [N x 3]
    size: torch.Tensor  # [N x 2]
    rotation: torch.Tensor  # [N x 3]
    translation: torch.Tensor  # [N x 3]
    post_translation: torch.Tensor  # [N x 2]

    def __repr__(self) -> str:
        return (
            f"DiffusionPrediction({self.origin=} {self.size=} {self.rotation=}"
            f" {self.translation=} {self.post_translation=})"
        )

    @classmethod
    def from_vec(cls, vec: torch.Tensor) -> "DiffusionPrediction":
        origin, size, rotation, translation, post_translation = torch.split(
            vec, [3, 2, 3, 3, 2], dim=-1
        )
        return cls(
            origin=origin,
            size=size,
            rotation=rotation,
            translation=translation,
            post_translation=post_translation,
        )

    @classmethod
    def from_batch(cls, batch: Batch) -> "DiffusionPrediction":
        return cls(
            origin=batch.origin,
            size=batch.size,
            rotation=batch.rotation,
            translation=batch.translation,
            post_translation=batch.post_translation,
        )

    def to_vec(self) -> torch.Tensor:
        return torch.cat(
            [
                self.origin,
                self.size,
                self.rotation,
                self.translation,
                self.post_translation,
            ],
            dim=-1,
        )


def frequency_pos_embedding(
    x: torch.Tensor, num_feats: int, max_arg: float = 1000.0
) -> torch.Tensor:
    if num_feats == 0:
        return x
    assert num_feats % 2 == 0
    coeffs = torch.linspace(
        0, math.log(max_arg), num_feats // 2, device=x.device, dtype=x.dtype
    ).exp()
    args = (x[..., None] * coeffs).flatten(-2)
    return torch.cat([x, args.cos(), args.sin()], dim=-1)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class StretchPredictor(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.register_buffer("ratios", torch.linspace(-3, 3, 25, device=device).exp())
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, device=device),
            nn.ReLU(),
            nn.AvgPool2d(8, 8),
            nn.Flatten(),
            nn.Linear(256, len(self.ratios), device=device),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)

    def losses(self, images: torch.Tensor, ratios: torch.Tensor) -> torch.Tensor:
        log_ratios = ratios.log()
        indices = torch.argmin(
            (log_ratios[:, None] - self.ratios.log()).abs(), dim=-1, keepdim=True
        )
        log_probs = self(images).log_softmax(-1).gather(1, indices)
        return -log_probs.view(-1)
