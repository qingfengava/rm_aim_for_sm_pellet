#!/usr/bin/env python3
"""Export model by delegating to train/train_tiny_CNN.py."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root_dir = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Export pellet classifier to ONNX")
    parser.add_argument(
        "--config",
        default=str(root_dir / "train" / "config" / "train.yaml"),
        help="Path to training config yaml",
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(root_dir / "train" / "train_tiny_CNN.py"),
        "--config",
        str(Path(args.config)),
        "--mode",
        "export",
    ]

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
