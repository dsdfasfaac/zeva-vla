#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_libero_pro_full20_4090.sh baseline|stage3 OUTPUT_TAG}
output_tag=${2:?usage: run_libero_pro_full20_4090.sh baseline|stage3 OUTPUT_TAG}
case "$mode" in
  baseline|stage3) ;;
  *) echo "mode must be baseline or stage3" >&2; exit 2 ;;
esac

eval_root=/mnt/huangbingjia/eval/libero
bundle=$eval_root/handoffs/libero-pro-memory-eval-v1
model_root=$eval_root/zeva-vla-dingxin
deps=$eval_root/zeva-python-deps-py311
artifacts=$eval_root/zeva-artifacts/libero-v3-h5-lang
python=/mnt/huangbingjia/envs/libero-openpi-build/bin/python
provider_config=$model_root/scripts/libero_eval/libero_pro_${mode}_4090.json
overlay_patch=$model_root/scripts/libero_eval/libero_pro_h5_overlay.patch
patch_hash=$(sha256sum "$overlay_patch" | cut -c1-12)
eval_bundle=$eval_root/handoff-eval-work/libero-pro-memory-eval-h5-$patch_hash
run_root=$eval_root/handoff-eval-work/libero-pro-memory-client-server/runners/$output_tag
gpus=(0 1 2 3 5 7)
ports=(8500 8501 8502 8503 8504 8505)
pids=()

mkdir -p "$run_root/server-logs"
if [[ ! -f "$eval_bundle/.zeva-h5-overlay-$patch_hash" ]]; then
  overlay_tmp=$(mktemp -d "$eval_root/handoff-eval-work/libero-pro-h5-overlay.XXXXXX")
  cp -a "$bundle/." "$overlay_tmp/"
  patch --batch --forward -d "$overlay_tmp" -p1 < "$overlay_patch"
  touch "$overlay_tmp/.zeva-h5-overlay-$patch_hash"
  mv "$overlay_tmp" "$eval_bundle"
fi
for port in "${ports[@]}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "refusing to replace listener on port $port" >&2
    exit 1
  fi
done

cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
export PYTHONPATH=$bundle/src:$bundle:$model_root/src:$model_root:$deps

for index in "${!gpus[@]}"; do
  gpu=${gpus[$index]}
  port=${ports[$index]}
  CUDA_VISIBLE_DEVICES=$gpu "$python" "$eval_bundle/server/serve_policy.py" \
    --provider-module scripts.libero_eval.libero_pro_provider \
    --provider-config "$provider_config" \
    --checkpoint-path "$artifacts/baseline-handoff" \
    --device cuda:0 --host 127.0.0.1 --port "$port" \
    >"$run_root/server-logs/gpu${gpu}-port${port}.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" >"$run_root/server-pids.txt"

for index in "${!ports[@]}"; do
  port=${ports[$index]}
  pid=${pids[$index]}
  ready=0
  for _ in $(seq 1 180); do
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -100 "$run_root/server-logs/gpu${gpus[$index]}-port${port}.log" >&2
      exit 1
    fi
    if curl --silent --fail "http://127.0.0.1:${port}/v1/health" >/dev/null; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ $ready -ne 1 ]]; then
    echo "server on port $port did not become ready" >&2
    exit 1
  fi
done

source "$eval_bundle/environment/4090_shared_assets.sh"
export EVAL_BUNDLE=$eval_bundle
export OUTPUT_TAG=$output_tag
export GPUS="${gpus[*]}"
server_urls=()
for port in "${ports[@]}"; do
  server_urls+=("http://127.0.0.1:${port}")
done
export SERVER_URLS="${server_urls[*]}"

bash "$eval_bundle/scripts/run_full.sh"
