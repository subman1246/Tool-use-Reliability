"""Drive the BFCL sweep to its planned n across a multi-day token budget.

A Python twin of drive_bfcl.sh. The shell version works when launched from an interactive
shell, but launching it detached on Windows (Start-Process -> bash -lc) produced no output
and no child process, whereas Start-Process against python.exe works -- which is how the
repair arm's driver survives. Same logic, launchable the way that actually detaches.

Every completed call is in the content-addressed cache, so each attempt replays the
finished prefix for free and only pays for new ground. The daily token allowance is the
binding constraint, so the loop sleeps and retries rather than running once.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def completed(tag: str) -> int:
    try:
        meta = json.loads((ROOT / "data" / "results" / f"{tag}_meta.json")
                          .read_text(encoding="utf-8"))
        return int(meta.get("tasks_completed", 0))
    except Exception:                                            # noqa: BLE001
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bfclpilot")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=60)
    ap.add_argument("--interval", type=int, default=1200)
    ap.add_argument("--hours", type=float, default=60.0)
    args = ap.parse_args()

    deadline = time.time() + args.hours * 3600
    for i in range(1, args.attempts + 1):
        if time.time() > deadline:
            print("[bfcl-driver] wall-clock budget exhausted", flush=True)
            return
        print("[bfcl-driver] attempt %d, %.1fh of budget left"
              % (i, (deadline - time.time()) / 3600), flush=True)
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_bfcl_suite.py"),
             "--tag", args.tag, "--n", str(args.n)],
            cwd=str(ROOT), capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stdout.write(r.stderr[-2000:])
            print("[bfcl-driver] attempt %d exited %d" % (i, r.returncode), flush=True)
        done = completed(args.tag)
        print("[bfcl-driver] completed %d/%d" % (done, args.n), flush=True)
        if done >= args.n:
            print("[bfcl-driver] done", flush=True)
            return
        print("[bfcl-driver] waiting %ds for token refill" % args.interval, flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
