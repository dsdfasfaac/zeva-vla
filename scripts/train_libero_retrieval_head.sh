#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
GPU="${GPU:-7}"

CHECKPOINT="${CHECKPOINT:-/path/to/behaviorvla/checkpoint}"
SOURCE_PATH="${SOURCE_PATH:-/path/to/libero/source_features.pt}"
MEMORY_BANK="${MEMORY_BANK:-/path/to/libero/memory_bank.pt}"
SAVE_DIR="${SAVE_DIR:-/path/to/retrieval/head/output}"

EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-32}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-300}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT/model.safetensors" ]]; then
  echo "Checkpoint not found: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi

if [[ ! -f "$MEMORY_BANK" ]]; then
  echo "Memory bank not found: $MEMORY_BANK" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

if [[ "$FORCE_EXTRACT" == "1" || ! -f "$SOURCE_PATH" ]]; then
  echo "[1/2] Extracting LIBERO VLM source features"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/extract_source.py \
    --config-name pi05_libero \
    --ckpt-path "$CHECKPOINT" \
    --batch-size "$EXTRACT_BATCH_SIZE" \
    --save-path "$SOURCE_PATH"
else
  echo "[1/2] Reusing existing source features: $SOURCE_PATH"
fi

echo "[2/2] Training LIBERO retrieval head"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/train_retrieval_head.py \
  --source-path "$SOURCE_PATH" \
  --memory-bank-path "$MEMORY_BANK" \
  --save-dir "$SAVE_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --eval-interval 5 \
  --save-interval 25 \
  --device cuda

echo "Retrieval head ready: $SAVE_DIR/best_model.pth"
