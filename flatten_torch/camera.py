from dataclasses import dataclass

import torch


@dataclass
class Projection:
    projected: torch.Tensor  # [N x K x 2]
    z: torch.Tensor  # [N x K x 1]


@dataclass
class Camera:
    rotation: torch.Tensor  # [N x 3 x 3] rotation matrix
    translation: torch.Tensor  # [N x 3] offset vector

    def project(self, coords: torch.Tensor) -> Projection:
        """
        :param coords: an [N x K x 3] batch of vector batches.
        :return: an [N x K x 2] batch of projected coordinates and the z for
                 each coordinate (to perform clipping).
        """
        p = torch.einsum("bjk,bnk->bnj", self.rotation, coords)
        p = self.translation[:, None] + p
        z = p[..., 2:]
        return Projection(p[..., :2] / z, z)


def euler_rotation(xyz: torch.Tensor) -> torch.Tensor:
    """
    :param xyz: an [N x 3] batch of Euler angles.
    :return: an [N x 3 x 3] batch of rotation matrices.
    """
    theta_x, theta_y, theta_z = xyz.unbind(1)
    zero = torch.zeros_like(theta_x)
    one = torch.ones_like(theta_x)

    cos_x = theta_x.cos()
    sin_x = theta_x.sin()
    rot_x = torch.stack(
        [
            torch.stack([one, zero, zero], dim=1),
            torch.stack([zero, cos_x, -sin_x], dim=1),
            torch.stack([zero, sin_x, cos_x], dim=1),
        ],
        dim=1,
    )

    cos_y = theta_y.cos()
    sin_y = theta_y.sin()
    rot_y = torch.stack(
        [
            torch.stack([cos_y, zero, sin_y], dim=1),
            torch.stack([zero, one, zero], dim=1),
            torch.stack([-sin_y, zero, cos_y], dim=1),
        ],
        dim=1,
    )

    cos_z = theta_z.cos()
    sin_z = theta_z.sin()
    rot_z = torch.stack(
        [
            torch.stack([cos_z, -sin_z, zero], dim=1),
            torch.stack([sin_z, cos_z, zero], dim=1),
            torch.stack([zero, zero, one], dim=1),
        ],
        dim=1,
    )

    return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))
