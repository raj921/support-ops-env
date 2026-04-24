#!/usr/bin/env bash
# Push the GRPO training notebook to Kaggle as a Kernel.
# Prereq: ~/.kaggle/kaggle.json from https://www.kaggle.com/settings → API → Create New Token
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
KERNEL_DIR="kernels/support-ops-grpo"

# Kaggle CLI accepts any of: kaggle.json, KAGGLE_USERNAME+KAGGLE_KEY, or KAGGLE_API_TOKEN (see kaggle-api authenticate()).
if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]] \
   && [[ -z "${KAGGLE_API_TOKEN:-}" ]] \
   && { [[ -z "${KAGGLE_USERNAME:-}" ]] || [[ -z "${KAGGLE_KEY:-}" ]]; }; then
  echo "No Kaggle credentials found. Use one of:"
  echo "  1) ~/.kaggle/kaggle.json (from Kaggle → Settings → API → Create New Token)"
  echo "  2) export KAGGLE_USERNAME=... KAGGLE_KEY=...   (values from that JSON)"
  echo "  3) export KAGGLE_API_TOKEN=...   (OAuth-style token if you use that flow)"
  exit 1
fi

if [[ ! -d .venv-kaggle ]]; then
  python3 -m venv .venv-kaggle
fi
.venv-kaggle/bin/pip install -q 'kaggle>=1.6.0'

cp -f support_ops_kaggle.ipynb "$KERNEL_DIR/support_ops_kaggle.ipynb"
echo "Synced notebook → $KERNEL_DIR/support_ops_kaggle.ipynb"
.venv-kaggle/bin/kaggle kernels push -p "$KERNEL_DIR"
echo "Done."
