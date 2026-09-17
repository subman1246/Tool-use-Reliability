"""Is each sweep actually alive right now? Re-derived, never trusted from a stored PID.

WHY THIS IS NOT A WATCHDOG

A watchdog process has exactly the same mortality as the thing it watches -- the failure
that lost ~44h was the parent dying and taking its children with it, and a watchdog
launched the same way would have died in the same instant. Running one would move the
problem rather than solve it, and would be worse than nothing because it would look like
coverage. So this is an on-demand status command. The honest complement is someone
checking it, which is literally what caught the failure.

WHY IT DOES NOT TRUST PIDs

A PID recorded at launch is meaningless after a restart: the process may be gone, or the
number may have been recycled onto something unrelated. Liveness is re-derived every run
from two independent signals that fail differently:

  process table   a driver process AND a working child, matched by command line rather
                  than by number. One alone is the signature of a failed detach -- the
                  driver up with nothing underneath it.
  log freshness   how long since the driver last wrote. A live sweep that is merely
                  waiting for its token bucket still ticks; a silent log past the retry
                  interval means stalled, not paced.

Either signal alone can mislead. A process can be alive and wedged; a log can look fresh
seconds after the writer died. Together they catch both.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

# name -> (driver fragment, WORKER fragment, log files, results jsonl, retry interval)
# The worker is matched separately on purpose: a driver with no worker under it, while the
# log is stale, is exactly the failed-detach signature. Matching only the driver would
# report that case as healthy.
SWEEPS = [
    ("repair arm (allam-2-7b)", "resume_until_done.py", "run_real_suite.py",
     ["repair_run2.log", "repair_run.log"], "repair_groq_allam-2-7b.jsonl", 1200),
    ("BFCL pilot (gpt-oss-120b)", "drive_bfcl", "run_bfcl_suite.py",
     ["bfclpilot_run3.log", "bfclpilot_run2.log", "bfclpilot_run.log"],
     "bfclpilot_groq_openai_gpt-oss-120b.jsonl", 1200),
]


def _command_lines() -> list[str]:
    """Every running process's command line. PowerShell because tasklist omits them."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=60)
        return [l for l in out.stdout.splitlines() if l.strip()]
    except Exception:                                            # noqa: BLE001
        try:                                                     # POSIX fallback
            out = subprocess.run(["ps", "-eo", "args"], capture_output=True,
                                 text=True, timeout=30)
            return [l for l in out.stdout.splitlines() if l.strip()]
        except Exception:                                        # noqa: BLE001
            return []


def _tail(p: Path, n: int = 3) -> str:
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        return chr(10).join(txt.splitlines()[-n:])
    except Exception:                                            # noqa: BLE001
        return ""


def _newest_log(names: list[str]) -> Path | None:
    live = [RESULTS / n for n in names if (RESULTS / n).exists()]
    return max(live, key=lambda p: p.stat().st_mtime) if live else None


def _achieved(jsonl: str) -> str:
    p = RESULTS / jsonl
    if not p.exists():
        return "no results file"
    import collections
    per = collections.defaultdict(set)
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            per[r.get("depth")].add(r.get("pair_id") or r.get("task_id"))
    except Exception as e:                                       # noqa: BLE001
        return "unreadable (%s)" % type(e).__name__
    return str({k: len(v) for k, v in sorted(per.items()) if k is not None})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-running", action="store_true",
                    help="exit nonzero if any sweep is not alive (for use in a check)")
    args = ap.parse_args()

    cmds = _command_lines()
    if not cmds:
        print("WARNING: could not read the process table; liveness is unknown, not clean.")

    # Only real interpreter processes count. Without this the check matches the shell
    # wrappers and greps of whoever is INSPECTING the sweep -- a command line mentioning
    # the script name is not the script running -- and a dead sweep reads as healthy.
    # This false positive appeared on the first run of this very script.
    def _is_worker(c: str) -> bool:
        low = c.lower()
        return ("python" in low) and ("sweep_status" not in low)

    cmds = [c for c in cmds if _is_worker(c)]

    bad = []
    for name, dfrag, wfrag, logs, jsonl, interval in SWEEPS:
        drivers = sum(1 for c in cmds if dfrag in c)
        workers = sum(1 for c in cmds if wfrag in c)
        log = _newest_log(logs)
        age = (time.time() - log.stat().st_mtime) if log else None
        stale = age is not None and age > interval * 2

        finished = log is not None and "done" in _tail(log)
        if drivers == 0 and finished:
            # A sweep that ran to completion also has no driver. Reporting that as DEAD
            # would cry wolf on every success and train the reader to ignore the check.
            state = "FINISHED (ran to completion)"
        elif drivers == 0:
            state = "DEAD (no driver)"
        elif workers > 0:
            state = "alive (driver+worker)"
        elif stale:
            # driver up, nothing underneath it, and no recent write: the failed-detach
            # signature, and the case a driver-only check would call healthy
            state = "SUSPECT (driver, no worker, stale log)"
        else:
            state = "alive (idle between attempts)"

        print("%-28s %-34s drv=%d wrk=%d  log=%s  achieved=%s"
              % (name, state, drivers, workers,
                 ("%.0fs ago%s" % (age, " STALE" if stale else "")) if age is not None
                 else "none", _achieved(jsonl)))
        if state.startswith(("DEAD", "SUSPECT")):
            bad.append(name)

    if bad:
        print("\nNOT RUNNING: %s" % ", ".join(bad))
        print("Relaunch detached (see docs/RUNBOOK_expansion_phases.md) -- nohup from an "
              "agent tool shell does NOT survive the session.")
    else:
        print("\nall sweeps alive")
    return 1 if (bad and args.expect_running) else 0


if __name__ == "__main__":
    sys.exit(main())
