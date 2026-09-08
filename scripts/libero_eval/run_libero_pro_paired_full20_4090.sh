#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
stamp=${1:-20260903-v1}

bash "$script_dir/run_libero_pro_full20_4090.sh" \
  stage3 "zeva-stage3-libero-pro-full20-${stamp}"
bash "$script_dir/run_libero_pro_full20_4090.sh" \
  baseline "pi05-baseline-libero-pro-full20-${stamp}"
