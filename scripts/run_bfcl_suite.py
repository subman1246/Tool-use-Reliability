"""Run the BFCL L_d arm against a real provider.

SINGLE-MODEL CHECK, NOT COMPARATIVE. Of the paper's five models, three are retired from
Groq (llama-3.1-8b-instant, llama-3.3-70b-versatile, qwen/qwen3.6-27b) and gpt-oss-20b has
no synthetic baseline to compare a real-benchmark result against. gpt-oss-120b alone.

Reuses the cache, rate limiter and backend from the synthetic sweep, so an interrupted run
resumes for free: every completed call is content-addressed, and re-running replays the
finished prefix without spending. That matters here more than on the synthetic arms --
BFCL prompts run ~2,600 tokens per call, and gpt-oss-120b's daily allowance is 200,000,
so any full arm necessarily spans days.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv                                         # noqa: E402

from tur.harness.bfcl_runner import (curated_subset, run_bfcl_free,     # noqa: E402
                                     run_bfcl_teacher_forced)
from tur.harness.cache import Cache                                     # noqa: E402
from tur.harness.runner import DailyCapReached, LiteLLMBackend, RateLimiter  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="groq/openai/gpt-oss-120b")
    ap.add_argument("--tag", default="bfclpilot")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--depth", type=int, default=6,
                    help="exact gold-call depth to draw tasks at")
    ap.add_argument("--tpm", type=int, default=6000)
    ap.add_argument("--rpd", type=int, default=1000)
    ap.add_argument("--tpd", type=int, default=200000)
    ap.add_argument("--headroom", type=float, default=0.8)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not set; refusing to start a paid run blind")

    results = ROOT / "data" / "results"
    results.mkdir(parents=True, exist_ok=True)
    cache = Cache(ROOT / "data" / "cache")
    limiter = RateLimiter(tpm=args.tpm, rpd=args.rpd, tpd=args.tpd,
                          headroom=args.headroom)
    backend = LiteLLMBackend(args.model, cache=cache, limiter=limiter)

    tasks = [t for t in curated_subset(args.depth, args.depth)][:args.n]
    if len(tasks) < args.n:
        raise SystemExit("only %d tasks at depth %d, asked for %d"
                         % (len(tasks), args.depth, args.n))

    safe = args.model.replace("/", "_")
    out_path = results / ("%s_%s.jsonl" % (args.tag, safe))
    print("BFCL L_d arm -- SINGLE-MODEL CHECK (%s), NOT comparative" % args.model)
    print("tasks: %d at depth %d  ->  %s" % (len(tasks), args.depth, out_path))

    started = time.time()
    records = []
    stopped_early = False
    for i, task in enumerate(tasks):
        if args.max_minutes and (time.time() - started) / 60 > args.max_minutes:
            print("  wall-clock budget reached at task %d/%d; stopping cleanly with a "
                  "prefix" % (i, len(tasks)))
            stopped_early = True
            break
        try:
            for fn in (run_bfcl_teacher_forced, run_bfcl_free):
                records.extend(fn(task, backend))
        except DailyCapReached as e:
            # A truncated prefix is usable data at the achieved n; a crash is not.
            print("  daily cap reached at task %d/%d (%s); writing the prefix"
                  % (i, len(tasks), e))
            stopped_early = True
            break
        done = sum(1 for r in records if r.run_mode == "bfcl_free")
        print("  [%2d/%2d] %-22s depth=%d  free-steps so far=%d"
              % (i + 1, len(tasks), task.task_id, task.depth, done), flush=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r)) + "\n")

    meta = {
        "model": args.model, "tag": args.tag, "single_model_check": True,
        "comparative": False, "depth": args.depth,
        "tasks_planned": args.n,
        "tasks_completed": len({r.task_id for r in records}),
        "stopped_early": stopped_early,
        "scorer": "fixed-gold per-call (OURS, not BFCL's state-based checker)",
        "backend_stats": backend.stats(),
    }
    (results / ("%s_meta.json" % args.tag)).write_text(json.dumps(meta, indent=2),
                                                       encoding="utf-8")
    print("\nwrote %d records over %d tasks -> %s"
          % (len(records), meta["tasks_completed"], out_path))


if __name__ == "__main__":
    main()
