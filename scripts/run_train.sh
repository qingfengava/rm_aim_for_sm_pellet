#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/train/config/train.yaml"

python3 "${ROOT_DIR}/train/train_tiny_CNN.py" --config "${CONFIG_PATH}" --mode all
