#!/usr/bin/env python3
"""Release a deliberately stopped eval worker after resumed progress completes.

This is a small runtime safety guard for a recovery case where an eval launcher
must remain alive while one failed task resumes against its existing policy
server.  It never changes progress or restarts a process: the held launcher is
continued only after the progress file proves the requested episode count is
complete.  If the watched rescue process exits first, the guard fails closed
and leaves the launcher stopped for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


def process_cmdline(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError):
        return None


def completed(progress_path: Path, target_episodes: int) -> tuple[bool, int]:
    try:
        payload = json.loads(progress_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False, 0
    count = int(payload.get("completed_episodes", 0))
    return bool(payload.get("complete")) and count == target_episodes, count


def write_event(path: Path, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields["timestamp"] = datetime.now(timezone.utc).astimezone().isoformat()
    path.write_text(json.dumps(fields, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--target-episodes", type=int, required=True)
    parser.add_argument("--hold-pid", type=int, required=True)
    parser.add_argument("--hold-token", required=True)
    parser.add_argument("--watched-pid", type=int, required=True)
    parser.add_argument("--watched-token", required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()

    while True:
        is_complete, count = completed(args.progress, args.target_episodes)
        if is_complete:
            hold_cmd = process_cmdline(args.hold_pid)
            if hold_cmd is None or args.hold_token not in hold_cmd:
                write_event(
                    args.event_log,
                    state="hold_process_identity_mismatch",
                    completed_episodes=count,
                    hold_pid=args.hold_pid,
                    hold_cmdline=hold_cmd,
                )
                return 2
            os.kill(args.hold_pid, signal.SIGCONT)
            write_event(
                args.event_log,
                state="released",
                completed_episodes=count,
                hold_pid=args.hold_pid,
                watched_pid=args.watched_pid,
            )
            return 0

        watched_cmd = process_cmdline(args.watched_pid)
        if watched_cmd is None or args.watched_token not in watched_cmd:
            write_event(
                args.event_log,
                state="rescue_exited_before_completion",
                completed_episodes=count,
                hold_pid=args.hold_pid,
                watched_pid=args.watched_pid,
                watched_cmdline=watched_cmd,
            )
            return 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
