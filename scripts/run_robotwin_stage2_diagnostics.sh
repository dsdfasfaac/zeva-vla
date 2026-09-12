#!/usr/bin/env bash
set -euo pipefail

# Read-only validation launcher for the verified Python 3.10 H100 runtime.
# Unlike the training launcher, this does not create/update runtime symlinks.
: "${CUDA_VISIBLE_DEVICES:?Select one idle GPU explicitly}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "Diagnostics are single-process; select exactly one GPU." >&2
  exit 2
fi
zeva_diag_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
zeva_handoff=${ROBOTWIN_HANDOFF:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1}
zeva_handoff_runtime="$zeva_handoff/runtime"
zeva_native_runtime=${NATIVE_TRANSFORMERS_RUNTIME:-/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912}
zeva_local_deps=${ZEVA_RUNTIME_DEPS:-/data1/dingxin/zeva-runtime-deps}
zeva_compiled_deps=${ZEVA_COMPILED_DEPS:-/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1}
zeva_python=${PI05_PYTHON:-python3}
zeva_system_site=$("$zeva_python" -c 'import site; print(site.getsitepackages()[0])')
export EGOSCALE_LEROBOT_SOURCE="$zeva_handoff_runtime/lerobot-main-py311-v1/src"
export PYTHONPATH="$zeva_native_runtime:$zeva_local_deps:$zeva_compiled_deps:$zeva_system_site:$zeva_handoff_runtime/h100-extra-deps:$zeva_handoff_runtime/lerobot-overlay-v2:$EGOSCALE_LEROBOT_SOURCE:$zeva_handoff_runtime/src:$zeva_diag_root/src:$zeva_diag_root:$zeva_handoff_runtime/lerobot-main-deps-py311-v1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=32
exec "$zeva_python" -u "$zeva_diag_root/scripts/eval_robotwin_stage2_diagnostics.py" "$@"
