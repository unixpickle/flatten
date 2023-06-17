import math
from dataclasses import dataclass, fields
from typing import Callable, Optional

import torch

from .camera import Camera, euler_rotation


@dataclass
class Batch:
    origin: torch.Tensor  # [N x 3] batch of source origins
    size: torch.Tensor  # [N x 2] batch of width+height
    rotation: torch.Tensor  # [N x 3] batch of Euler angles
    translation: torch.Tensor  # [N x 3] batch of translations
    proj_corners: torch.Tensor  # [N x 4 x 2] batch of projected corners

    def cat(self, other: "Batch") -> "Batch":
        kwargs = dict()
        for field in fields(Batch):
            kwargs[field.name] = torch.cat(
                [getattr(self, field.name), getattr(other, field.name)]
            )
        return Batch(**kwargs)

    def __getitem__(self, *args) -> "Batch":
        return self.map(lambda x: x.__getitem__(*args))

    def to(self, *args, **kwargs) -> "Batch":
        return self.map(lambda x: x.to(*args, **kwargs))

    def map(self, f: Callable[[torch.Tensor], torch.Tensor]) -> "Batch":
        kwargs = dict()
        for field in fields(Batch):
            kwargs[field.name] = f(getattr(self, field.name))
        return Batch(**kwargs)

    def __len__(self) -> int:
        return len(self.origin)

    @classmethod
    def sample_batch(
        cls,
        size: int,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
        margin: float = 0.1,
        z_near: float = 0.1,
    ) -> "Batch":
        sample_size = size * 10
        res: Optional[Batch] = None
        while not res or len(res) < size:
            sub_batch = cls._sample_batch(
                max_batch=sample_size,
                device=device,
                generator=generator,
                margin=margin,
                z_near=z_near,
            )
            if res is None:
                res = sub_batch
            else:
                res = res.cat(sub_batch)
        return res[:size]

    @classmethod
    def _sample_batch(
        cls,
        *,
        max_batch: int,
        device: torch.device,
        generator: Optional[torch.Generator],
        margin: float,
        z_near: float,
    ) -> "Batch":
        euler_angles = torch.randn(
            size=(max_batch, 3), generator=generator, device=device
        ) * torch.tensor([0.75, 0.75, 0.2], device=device)
        origin = (
            torch.rand(size=(max_batch, 3), generator=generator, device=device) * 10 - 5
        )
        origin[..., 2] = 0
        size = (
            torch.rand(size=(max_batch, 2), generator=generator, device=device) * 5
            + 0.01
        )
        translation = torch.rand(
            size=(max_batch, 3), generator=generator, device=device
        )
        translation[..., :2] *= 10
        translation[..., :2] -= 5
        translation[..., 2] = -(z_near + translation[..., 2] * 10)

        corners = corners_on_zplane(origin, size)
        camera = Camera(rotation=euler_rotation(euler_angles), translation=translation)
        proj = camera.project(corners)
        areas = areas_of_corners(proj.projected)
        area_thresh = torch.randn(
            size=areas.shape, generator=generator, device=device
        ).pow(0.75)
        valid = (
            (proj.projected[..., 0] >= -margin)
            & (proj.projected[..., 0] <= 1 + margin)
            & (proj.projected[..., 1] >= -margin)
            & (proj.projected[..., 1] <= 1 + margin)
            & (proj.z[..., 0] < -z_near)
            & (areas > area_thresh)[..., None]
        ).all(-1)

        return cls(
            origin=origin[valid],
            size=size[valid],
            rotation=euler_angles[valid],
            translation=translation[valid],
            proj_corners=proj.projected[valid],
        )


def areas_of_corners(corners: torch.Tensor) -> torch.Tensor:
    """
    Get the area of N quadrilaterals passed as an [N x 4 x 2] tensor.
    """

    def tri_area(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        return 0.5 * (v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]).abs()

    v1 = corners[:, 1] - corners[:, 0]
    v2 = corners[:, 3] - corners[:, 0]
    v3 = corners[:, 1] - corners[:, 2]
    v4 = corners[:, 3] - corners[:, 2]
    return tri_area(v1, v2) + tri_area(v3, v4)


def corners_on_zplane(origin: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """
    Get an [N x 4 x 3] grid of 3D corner points.
    """
    zero = torch.zeros_like(size[..., 0])
    return torch.stack(
        [
            origin,
            origin + torch.stack([size[..., 0], zero, zero], dim=-1),
            origin + torch.stack([size[..., 0], size[..., 1], zero], dim=-1),
            origin + torch.stack([zero, size[..., 1], zero], dim=-1),
        ],
        dim=1,
    )
