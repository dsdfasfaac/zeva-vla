#!/usr/bin/env bash
set -euo pipefail

python_bin=${PI05_PYTHON:-python3}
zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
zeva_deps=${ZEVA_RUNTIME_DEPS:-/data1/dingxin/zeva-runtime-deps}

mkdir -p "$zeva_deps"
export PYTHONPATH="$zeva_deps${PYTHONPATH:+:$PYTHONPATH}"
export MAX_JOBS=${MAX_JOBS:-8}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export ZEVA_H100_ONLY=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE

if ! "$python_bin" -c 'import causal_conv1d' >/dev/null 2>&1; then
  "$python_bin" -m pip install \
    --target "$zeva_deps" \
    --no-deps \
    --no-build-isolation \
    "$zeva_root/causal-conv1d"
fi

if ! "$python_bin" -c 'import mamba_ssm' >/dev/null 2>&1; then
  "$python_bin" -m pip install \
    --target "$zeva_deps" \
    --no-deps \
    --no-build-isolation \
    "$zeva_root/mamba"
fi

"$python_bin" -c 'import causal_conv1d, mamba_ssm, torch; print(torch.__version__, causal_conv1d.__version__, mamba_ssm.__version__)'
