import torch

from flatten_torch.gaussian_diffusion import diffusion_from_config
from flatten_torch.model import DiffusionPredictor

LOAD_PATH = "diffusion_model.pt"


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

    # Test input, should be from origin (0.3, 0.3, 0), size 0.15, rotation y 0.1, camera x -1
    input = torch.tensor(
        [
            0.28982110038313036,
            0.2912762684788842,
            0.42850143149999603,
            0.28710193481469054,
            0.42850143149999603,
            0.43065290222203584,
            0.28982110038313036,
            0.4369144027183263,
        ],
        device=device,
    )[None]

    sample = diffusion.p_sample_loop(
        model,
        shape=(len(input), model.d_input),
        clip_denoised=False,
        model_kwargs=dict(cond=input),
    )

    origin, size, rotation, translation, post_translation = torch.split(
        sample, [3, 2, 3, 3, 2], dim=-1
    )
    print(f"{origin=} {size=} {rotation=} {translation=} {post_translation=}")


if __name__ == "__main__":
    main()
