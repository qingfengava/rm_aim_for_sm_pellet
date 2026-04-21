#!/usr/bin/env bash
set -euo pipefail

if ! command -v pidstat >/dev/null 2>&1; then
  echo "pidstat not found. Install sysstat first."
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <pid>"
  exit 1
fi

pidstat -h -u -r -p "$1" 1
