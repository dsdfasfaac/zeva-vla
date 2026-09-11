#!/usr/bin/env bash
# One side of a matched Base/ZeVA v2 pair; explicit artifacts, no legacy defaults.
set -euo pipefail

variant=${1:?Specify baseline or zeva}
case "$variant" in
  baseline|zeva) ;;
  *) echo "Expected baseline or zeva" >&2; exit 2 ;;
esac
zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
handoff=${ROBOTWIN_HANDOFF:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1}
: "${ZEVA_V2_CHECKPOINT:?Pin an immutable selected v2 checkpoint file}"
: "${ZEVA_V2_ARTIFACT_DIR:?Select complete bank/live/retrieval artifacts}"
: "${ZEVA_V2_PAIR_ROOT:?Set a fresh matched-pair output root}"
: "${CUDA_VISIBLE_DEVICES:?Select verified-free GPUs explicitly}"
: "${ZEVA_MAIN_PROCESS_PORT:?Set a distinct DDP port for each variant}"
processes=${ZEVA_PROCESSES:-4}
case "$processes" in
  4) accumulation=4 ;;
  8) accumulation=2 ;;
  *) echo "Use four or eight GPUs; global batch remains 256" >&2; exit 2 ;;
esac
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#devices[@]}" -ne "$processes" ]]; then
  echo "GPU list and process count disagree" >&2; exit 2
fi
output=$ZEVA_V2_PAIR_ROOT/$variant
if [[ -e "$output" ]]; then
  echo "Refusing to overwrite an existing run: $output" >&2; exit 2
fi
for artifact in "$ZEVA_V2_CHECKPOINT" \
  "$ZEVA_V2_ARTIFACT_DIR/train_causal_bank.pt" \
  "$ZEVA_V2_ARTIFACT_DIR/live_queries_h15.pt" \
  "$ZEVA_V2_ARTIFACT_DIR/task_retrieval.pth"; do
  test -s "$artifact"
done
# Stage2 itself checks v2 schema, complete artifacts, hashes and H15 lineage.
# No background launch here: the caller owns logging and the process handle.
ZEVA_PROCESSES=$processes bash "$zeva_root/scripts/train_robotwin_stage2_8gpu.sh" \
  --training-variant "$variant" \
  --foundation-checkpoint "$handoff/checkpoint/pretrained_model-best-v1" \
  --goal-embedding-checkpoint "$handoff/checkpoint/pretrained_model" \
  --zte-checkpoint "$ZEVA_V2_CHECKPOINT" \
  --causal-bank "$ZEVA_V2_ARTIFACT_DIR/train_causal_bank.pt" \
  --live-queries "$ZEVA_V2_ARTIFACT_DIR/live_queries_h15.pt" \
  --task-retrieval "$ZEVA_V2_ARTIFACT_DIR/task_retrieval.pth" \
  --dataset-root "${ROBOTWIN_DATASET_ROOT:-/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data}" \
  --task-subset "$zeva_root/configs/robotwin_zeva_advantage10.json" \
  --save-dir "$output" \
  --steps 5000 --warmup-steps 500 --save-freq 500 --eval-batches 128 \
  --batch-size 16 --gradient-accumulation-steps "$accumulation" \
  --num-workers 4 --video-backend torchcodec --decoder-threads 1 \
  --action-expert-learning-rate 5e-6 --learning-rate 5e-5 \
  --prior-loss-weight 0.01 --prior-residual-dropout-probability 0.4 \
  --baseline-preserve-interval 4 \
  --compile-model --compile-mode default --seed 1000
