import argparse
import io
import json
import struct
from typing import List, Union

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    args = parser.parse_args()

    sd = torch.load(args.input_path, map_location="cpu")
    if "model" in sd:
        sd = sd["model"]

    metadata = bytes(json.dumps([(k, v.shape) for k, v in sd.items()]), "utf-8")
    data = io.BytesIO()

    with open(args.output_path, "wb") as f:
        f.write(struct.pack("<I", len(metadata)))
        f.write(metadata)
        for v in sd.values():
            data = v.reshape(-1).float().tolist()
            f.write(struct.pack(f"<{len(data)}f", *data))


def round_floats(floats: Union[List[float], float], prec: int):
    if isinstance(floats, list) and isinstance(floats[0], float):
        return [round(x, prec) for x in floats]
    elif isinstance(floats, list):
        return [round_floats(xs, prec) for xs in floats]
    else:
        return round(floats, prec)


if __name__ == "__main__":
    main()
