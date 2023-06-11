import argparse
import json

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    args = parser.parse_args()

    sd = torch.load(args.input_path, map_location="cpu")
    result = {k: v.tolist() for k, v in sd["model"].items()}
    with open(args.output_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
