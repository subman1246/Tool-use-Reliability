"""Drive the real sweep to completion across a multi-day token budget.

Why this exists: the binding constraint is a per-model daily token allowance, and
those allowances behave as continuously-refilling buckets rather than
midnight-resetting counters -- the 429 that enforces a daily cap quotes a retry
delay of minutes, and the observed reset interval for a 1,000/day request cap is
86,400/1,000 seconds. So the way to spend an 8-day budget is not to run once a day;
it is to run, get refused, wait for the bucket to refill a little, and run again.
Every completed call is in the content-addressed response cache, so each attempt
replays finished work for free and extends the prefix.

This is a driver, not a new experiment: it shells out to run_real_suite.py with
the same arguments each time. All the science, pacing and stop-on-cap logic stays
there.

Usage:
  py scripts/resume_until_done.py --hours 12
  py scripts/resume_until_done.py --hours 12 --interval 1200 --tag real

Stops when every model has reached its planned allocation, when the wall-clock
budget runs out, or when an attempt fails for a reason that is not a rate limit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_progress(tag: str) -> tuple[dict, dict]:
    """Achieved vs planned per-depth counts, read from the run's own metadata."""
    meta_path = ROOT / "data" / "results" / f"{tag}_meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}, {}
    planned = {n: v.get("primary", {})
               for n, v in (meta.get("allocation") or {}).items()}
    short = meta.get("achieved_shortfall") or {}
    achieved = {n: v.get("achieved", {}) for n, v in short.items()}
    return planned, achieved


def summarise(tag: str) -> str:
    planned, achieved = load_progress(tag)
    if not planned:
        return "no metadata yet"
    parts = []
    for name, want in planned.items():
        got = achieved.get(name)
        if got is None:
            parts.append(f"{name.split('/')[-1]}=complete")
        else:
            parts.append(f"{name.split('/')[-1]}="
                         f"{sum(got.values())}/{sum(want.values())}")
    return "  ".join(parts)


def all_complete(tag: str) -> bool:
    planned, achieved = load_progress(tag)
    return bool(planned) and not achieved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=12.0,
                    help="wall-clock budget for this driver (default: %(default)s)")
    ap.add_argument("--interval", type=int, default=1200,
                    help="seconds to wait between attempts (default: %(default)s). "
                         "The buckets refill at TPD/86400 per second, so a longer "
                         "wait buys a proportionally larger burst; there is no "
                         "advantage to hammering.")
    ap.add_argument("--tag", default="real")
    ap.add_argument("--max-attempts", type=int, default=200)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="further arguments passed through to run_real_suite.py")
    args = ap.parse_args()

    deadline = time.monotonic() + args.hours * 3600
    cmd = [sys.executable, str(ROOT / "scripts" / "run_real_suite.py"),
           "--tag", args.tag] + list(args.extra)

    attempt = 0
    while attempt < args.max_attempts:
        attempt += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"\n[driver] wall-clock budget of {args.hours}h exhausted after "
                  f"{attempt - 1} attempt(s)")
            break

        print(f"\n[driver] attempt {attempt}, {remaining / 3600:.1f}h of budget "
              f"left", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        tail = [ln for ln in (proc.stdout or "").splitlines()
                if ln.strip() and "litellm" not in ln.lower()
                and "Provider List" not in ln and "Give Feedback" not in ln]
        for ln in tail[-25:]:
            print("  " + ln)
        if proc.returncode not in (0, 2):
            print(f"[driver] run exited {proc.returncode}, which is neither "
                  f"success nor a clean cap-stop. Stopping so the cause is not "
                  f"buried under retries.")
            print((proc.stderr or "")[-2000:])
            return proc.returncode

        print(f"[driver] progress: {summarise(args.tag)}", flush=True)
        if proc.returncode == 0 and all_complete(args.tag):
            print(f"\n[driver] every model reached its planned allocation after "
                  f"{attempt} attempt(s)")
            return 0

        wait = min(args.interval, max(0, deadline - time.monotonic()))
        if wait <= 0:
            break
        print(f"[driver] waiting {wait:.0f}s for token buckets to refill",
              flush=True)
        time.sleep(wait)

    print(f"\n[driver] stopping. final progress: {summarise(args.tag)}")
    print("[driver] re-run this script to continue; cached work replays free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
