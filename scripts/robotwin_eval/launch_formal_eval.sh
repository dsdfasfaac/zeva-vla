#!/usr/bin/env bash
set -euo pipefail

# Eight ZeVA model servers on aigc29 and eight RoboTwin render clients on aigc24.
# ZeVA-only formal protocol: randomized/seen, 20 expert-valid seeds selected
# independently per task starting at 1000, H15 replanning, and one resumable
# progress file per task. No PI-baseline rollouts are launched by this script.

model_host=${MODEL_HOST:-aigc29}
render_host=${RENDER_HOST:-aigc24}
model_ip=${MODEL_IP:-172.16.80.163}
slots=${SLOTS:-8}
episodes=${EPISODES:-20}
base_port=${BASE_PORT:-19100}
zeva_root=/mnt/100T/users/dingxin/VLA/zeva-vla/ICML26-BehaviorVLA
shared_runtime=/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin
render_runtime=/data1/dingxin/robotwin-formal-eval/RoboTwin
output=${OUTPUT_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-zeva-only-stage2a-h15-seen-v1}
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
release_runtime=$handoff/runtime
model_config=$zeva_root/scripts/robotwin_eval/model_config.yml
adapter_sha256=$(sha256sum "$zeva_root/scripts/robotwin_eval/zeva_policy.py" | awk '{print $1}')

mkdir -p "$output"/{logs,progress,status,server-pids}
python3 - "$shared_runtime/task_config/_eval_step_limit.yml" "$output/tasks.txt" <<'PY'
import sys
import yaml

source, destination = sys.argv[1:]
tasks = list(yaml.safe_load(open(source, encoding="utf-8")))
if len(tasks) != 50:
    raise SystemExit(f"expected 50 RoboTwin tasks, got {len(tasks)}")
open(destination, "w", encoding="utf-8").write("\n".join(tasks) + "\n")
PY
mapfile -t tasks < "$output/tasks.txt"

cat > "$output/manifest.json" <<EOF
{
  "schema": "zeva-robotwin-formal-zeva-only-eval-v1",
  "stage1": "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/zte_best.pth",
  "stage2a": "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage2a-adapter/005000",
  "task_retrieval": "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1.5-task-retrieval/task_retrieval.pth",
  "causal_bank": "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/train_causal_bank.pt",
  "tasks": 50,
  "profile": "randomized_seen_large_d435",
  "task_config": "zeva_randomized",
  "episodes_per_task": $episodes,
  "absolute_start_seed": 1000,
  "seed_selection": "first-20-expert-valid-per-task",
  "instruction_type": "seen",
  "camera": "Large_D435_640x480",
  "action_contract": "chunk-start-relative-eef16-predict-h50-execute-h15",
  "image_contract": "chw-float32-zero-one",
  "evaluation_adapter_sha256": "$adapter_sha256",
  "model_host": "$model_host",
  "render_host": "$render_host",
  "slots": $slots
}
EOF

model_pythonpath=$(ssh "$model_host" "python3 - <<'PY'
import site
print(site.getsitepackages()[0])
PY
")
model_pythonpath=/data1/dingxin/zeva-runtime-deps:$model_pythonpath:$release_runtime/h100-extra-deps:$release_runtime/lerobot-main-deps-py311-v1:$release_runtime/lerobot-overlay-v2:$release_runtime/lerobot-main-py311-v1/src:$release_runtime/src:$zeva_root/src:$zeva_root

cleanup() {
  for slot in $(seq 0 $((slots - 1))); do
    pid_file="$output/server-pids/slot${slot}.pid"
    if [[ -f "$pid_file" ]]; then
      pid=$(<"$pid_file")
      ssh "$model_host" "kill '$pid' 2>/dev/null || true" || true
    fi
  done
}
trap cleanup EXIT INT TERM

for slot in $(seq 0 $((slots - 1))); do
  port=$((base_port + slot))
  server_log="$output/logs/server-slot${slot}.log"
  ssh "$model_host" "cd '$shared_runtime'; nohup env \
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES='$slot' PYTHONPATH='$model_pythonpath' \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    python3 script/policy_model_server.py --port '$port' --config '$model_config' \
      --overrides --host 0.0.0.0 > '$server_log' 2>&1 < /dev/null & echo \$!" \
      > "$output/server-pids/slot${slot}.pid"
done

for slot in $(seq 0 $((slots - 1))); do
  port=$((base_port + slot))
  for _ in $(seq 1 120); do
    if ssh "$model_host" "ss -ltn | grep -q ':$port '"; then
      break
    fi
    sleep 5
  done
  ssh "$model_host" "ss -ltn | grep -q ':$port '" || {
    echo "slot $slot model server did not become ready" >&2
    exit 1
  }
done

jobs=()
for task in "${tasks[@]}"; do
  jobs+=("$task")
done

workers=()
for slot in $(seq 0 $((slots - 1))); do
  (
    port=$((base_port + slot))
    for index in "${!jobs[@]}"; do
      (( index % slots == slot )) || continue
      task="${jobs[$index]}"
      job_id="$task"
      progress="$output/progress/$job_id.json"
      log="$output/logs/$job_id.log"
      status="$output/status/$job_id.json"
      result_dir="$output/results/$task"
      started=$(date -Iseconds)
      set +e
      ssh "$render_host" "cd '$render_runtime'; env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES='$slot' PYTHONPATH='$render_runtime/policy' \
        .venv_robotwin/bin/python script/eval_policy_client.py --port '$port' --config '$zeva_root/scripts/robotwin_eval/client_config.yml' \
        --overrides --task_name '$task' --task_config zeva_randomized --test_num '$episodes' \
        --instruction_type seen --seed 0 --absolute_start_seed 1000 --fixed_seed_sequence False \
        --policy_name zeva_policy --ckpt_setting stage2a-005000-h15-seen \
        --eval_video_log True --result_dir '$result_dir' \
        --execute_horizon 15 --chunk_length 50 --action_dim 16 \
        --server_host '$model_ip' --resume_progress_path '$progress'" > "$log" 2>&1
      rc=$?
      set -e
      finished=$(date -Iseconds)
      printf '{"job":"%s","slot":%d,"return_code":%d,"started":"%s","finished":"%s"}\n' \
        "$job_id" "$slot" "$rc" "$started" "$finished" > "$status"
      if (( rc != 0 )); then
        echo "$job_id failed with rc=$rc; progress is resumable" >&2
      fi
    done
  ) > "$output/logs/worker-slot${slot}.log" 2>&1 &
  workers+=("$!")
done

printf '%s\n' "${workers[@]}" > "$output/worker-pids.txt"
printf '{"state":"running","started":"%s","jobs":%d}\n' "$(date -Iseconds)" "${#jobs[@]}" > "$output/state.json"

worker_rc=0
for pid in "${workers[@]}"; do
  wait "$pid" || worker_rc=1
done

completed=$(find "$output/status" -type f -name '*.json' | wc -l)
failed=$(grep -l '"return_code":[1-9]' "$output"/status/*.json 2>/dev/null | wc -l || true)
if (( worker_rc == 0 )); then
  python3 - "$output" "$episodes" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
rows = []
for task in root.joinpath("tasks.txt").read_text().splitlines():
    progress = json.loads((root / "progress" / f"{task}.json").read_text())
    episodes = progress["episode_results"]
    seeds = [int(item["seed"]) for item in episodes]
    if not progress.get("complete") or len(episodes) != expected:
        raise RuntimeError(f"{task}: incomplete progress")
    if len(seeds) != expected or any(seed < 1000 for seed in seeds):
        raise RuntimeError(f"{task}: invalid expert-filtered seeds: {seeds}")
    if any(right <= left for left, right in zip(seeds, seeds[1:])):
        raise RuntimeError(f"{task}: seeds are not strictly increasing: {seeds}")
    videos = sorted((root / "results" / task).glob("episode*_randomized-true_success-*.mp4"))
    if len(videos) != expected:
        raise RuntimeError(f"{task}: expected {expected} videos, got {len(videos)}")
    successes = sum(bool(item["success"]) for item in episodes)
    video_successes = sum("success-true" in video.name for video in videos)
    if successes != video_successes:
        raise RuntimeError(f"{task}: progress/video success mismatch")
    rows.append({"task": task, "episodes": expected, "successes": successes,
                 "success_rate": successes / expected})
report = {
    "schema": "zeva-robotwin-zeva-only-eval-v1",
    "profile": "randomized_seen_large_d435",
    "absolute_start_seed": 1000,
    "seed_selection": "first-20-expert-valid-per-task",
    "task_count": len(rows),
    "episodes_per_task": expected,
    "total_episodes": len(rows) * expected,
    "total_successes": sum(row["successes"] for row in rows),
    "micro_success_rate": sum(row["successes"] for row in rows) / (len(rows) * expected),
    "macro_success_rate": sum(row["success_rate"] for row in rows) / len(rows),
    "tasks": rows,
}
temporary = root / "report.json.partial"
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "report.json")
PY
fi
printf '{"state":"complete","finished":"%s","jobs":%d,"completed":%d,"failed":%d}\n' \
  "$(date -Iseconds)" "${#jobs[@]}" "$completed" "$failed" > "$output/state.json"
exit "$worker_rc"
