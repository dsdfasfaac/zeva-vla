#!/usr/bin/env bash
# Canonical entrypoint for the two ZeVA PIM training settings.
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
setting=${1:---describe}

describe() {
  python3 - "$zeva_root/configs/robotwin_pim_training_settings.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
print("ZeVA PIM training settings")
print(f"default: {payload['default_setting']}")
for name, setting in payload["settings"].items():
    print(f"\n{name} [{setting['status']}]")
    print(f"  memory: {setting['memory_scope']}")
    print(f"  read:   {setting['read_contract']}")
    print(f"  reset:  {setting['reset_contract']}")
    print(f"  train:  {setting['training_pairs']}")
    print(f"  run:    {setting['entrypoint']}")
PY
}

case "$setting" in
  --describe|-h|--help)
    describe
    ;;
  cross-attempt)
    shift
    exec "$zeva_root/scripts/run_robotwin_cross_attempt_pim_stage2.sh" "$@"
    ;;
  within-episode)
    shift
    exec "$zeva_root/scripts/run_robotwin_episode_pim_stage2.sh" "$@"
    ;;
  *)
    echo "Unknown PIM setting: $setting" >&2
    echo "Use: $0 --describe|cross-attempt|within-episode" >&2
    exit 2
    ;;
esac
