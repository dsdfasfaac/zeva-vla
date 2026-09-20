#!/usr/bin/env bash
# Canonical name for the historical cross-attempt PIM training setting.
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$zeva_root/scripts/run_robotwin_pim_stage2.sh" "$@"
