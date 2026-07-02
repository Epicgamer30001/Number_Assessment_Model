"""Export model.pth weights to webapp/weights.json for the in-browser forward pass.

Run once from the Number_Assessment_Model/ directory:
    python webapp/export_weights.py
"""
import json
import os
import sys

import torch

# Allow importing CrossEntropy.py regardless of where this is run from.
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(HERE)  # Number_Assessment_Model/
sys.path.insert(0, MODEL_DIR)

from CrossEntropy import CrossEntropyModel  # noqa: E402

MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")
OUT_PATH = os.path.join(HERE, "weights.json")


def r_matrix(t):
    return [[round(v, 5) for v in row] for row in t.tolist()]


def r_vec(t):
    return [round(v, 5) for v in t.tolist()]


def main():
    model = CrossEntropyModel([784, 128, 64, 10], 0.02)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    sd = model.state_dict()

    out = {
        "layers": [
            {"W": r_matrix(sd["network.0.weight"]), "b": r_vec(sd["network.0.bias"])},
            {"W": r_matrix(sd["network.2.weight"]), "b": r_vec(sd["network.2.bias"])},
            {"W": r_matrix(sd["network.4.weight"]), "b": r_vec(sd["network.4.bias"])},
        ]
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f)

    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"Wrote {OUT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
