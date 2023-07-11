import argparse
import json
from typing import List, Union

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    args = parser.parse_args()

    sd = torch.load(args.input_path, map_location="cpu")
    if "model" in sd:
        sd = sd["model"]

    result = {k: round_floats(v.tolist(), args.precision) for k, v in sd.items()}
    with open(args.output_path, "w") as f:
        json.dump(result, f)


def round_floats(floats: Union[List[float], float], prec: int):
    if isinstance(floats, list) and isinstance(floats[0], float):
        return [round(x, prec) for x in floats]
    elif isinstance(floats, list):
        return [round_floats(xs, prec) for xs in floats]
    else:
        return round(floats, prec)


if __name__ == "__main__":
    main()
