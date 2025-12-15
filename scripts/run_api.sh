#!/usr/bin/env bash
set -euo pipefail

# Run FastAPI backend
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

python api/main.py
