"""The REAL experiment: same pipeline as run_full_analysis.py, but backed by
actual models via LiteLLM instead of simulated policies. This is the script
to run once you're in an environment with API keys and network access.

Usage:
  cp .env.example .env   # fill in keys; loaded automatically, no `source` needed
  python scripts/estimate_cost.py            # do this FIRST
  python scripts/run_real_suite.py --pilot   # small pilot run
  python scripts/run_real_suite.py           # full sweep

On Windows, invoke via the py launcher (`py scripts/run_real_suite.py`) -- the
`python` on PATH is often the Microsoft Store stub. There is no `source` on
Windows either, which is why keys are loaded from .env in-process below rather
than being expected in the environment.

Reads the model suite from config/default.yaml so the model list is a config
change, not a code change.

Results are written under a run tag (--tag, default "real") so that ablations
(e.g. the native calling-mode subset) don't overwrite the main run. Pass the
same tag to `python -m tur.analysis.plots --tag <tag>` to make its figures.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from tur.tasks.dag import generate_suite
from tur.harness.cache import Cache
from tur.harness.executor import FeedbackMode
from tur.harness.runner import run_free, run_teacher_forced, LiteLLMBackend
from tur.analysis.aggregate import (load_records, aggregate_by_depth,
                                    stats_to_arrays, bootstrap_L_ci)
from tur.model.hierarchical import build_and_sample


def delta_1(records: list[dict]) -> dict:
    """Model-free propagation check (see run_full_analysis.py for the same
    function; duplicated here to keep this script runnable standalone)."""
    by_task = {}
    for r in records:
        if r["run_mode"] != "free" or r.get("backend_error", False):
            continue
        by_task.setdefault(r["task_id"], {})[r["step"]] = r["args_correct_strict"]

    given_correct, given_wrong = [], []
    for steps in by_task.values():
        for t in sorted(steps):
            if (t + 1) in steps:
                (given_correct if steps[t] else given_wrong).append(steps[t + 1])
    gc = float(np.mean(given_correct)) if given_correct else float("nan")
    gw = float(np.mean(given_wrong)) if given_wrong else float("nan")
    return {"delta_1": gc - gw, "p_next_given_correct": gc,
           "p_next_given_wrong": gw, "n_given_correct": len(given_correct),
           "n_given_wrong": len(given_wrong)}

OUT_DIR = "data/results"
DEFAULT_TAG = "real"


def run_model(model_cfg: dict, depths: list[int], per_depth: int, seeds: int,
             max_retries: int, distractor_level: int, feedback: FeedbackMode,
             call_mode: str, cache_dir: str) -> tuple[list[dict], dict]:
    name = model_cfg["name"]
    cache = Cache(f"{cache_dir}/{name.replace('/', '_')}")
    backend = LiteLLMBackend(name, temperature=0.0, cache=cache)

    records = []
    for seed in range(seeds):
        suite = generate_suite(depths, per_depth, distractor_level,
                               base_seed=seed * 31 + 1000)
        for i, task in enumerate(suite):
            f = run_free(task, backend, call_mode, feedback, max_retries)
            t = run_teacher_forced(task, backend, call_mode, feedback, max_retries)
            records += [r.__dict__ for r in f] + [r.__dict__ for r in t]
            if (i + 1) % 50 == 0:
                print(f"  {name}: seed {seed}, {i + 1}/{len(suite)} tasks "
                     f"(calls={backend.n_calls} cache_hits={backend.n_cache_hits} "
                     f"failures={backend.n_failures})")
    return records, backend.stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--pilot", action="store_true",
                    help="tiny run (2 depths, 5 tasks/depth, 1 seed) to sanity "
                         "check everything before spending real budget")
    ap.add_argument("--models", nargs="+", default=None,
                    help="override the model list from config, by name")
    ap.add_argument("--call-mode", choices=["uniform", "native"], default="uniform")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="run tag for output filenames (default: %(default)s). "
                         "Use a distinct tag for ablations so they don't "
                         "overwrite the main run, then pass the same tag to "
                         "tur.analysis.plots --tag.")
    args = ap.parse_args()
    tag = args.tag

    # Load API keys from the repo-root .env before any backend is constructed.
    # Anchored to this file's location, not the CWD, so the script works when
    # invoked from anywhere.
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)
    if not env_path.exists():
        print(f"WARNING: no .env at {env_path}; relying on ambient environment")

    cfg = yaml.safe_load(open(args.config))
    depths = cfg.get("depths", [1, 2, 4, 6, 8])
    per_depth = cfg.get("per_depth", 200)
    # per_depth may be a per-depth mapping (see config comments) or a single
    # count. Normalise YAML keys to int and fail loudly on a depth with no
    # allocation, rather than silently generating zero tasks for it and
    # producing an empty depth bin the fit would then have to cope with.
    if isinstance(per_depth, dict):
        per_depth = {int(k): int(v) for k, v in per_depth.items()}
        missing = [d for d in depths if not per_depth.get(d)]
        if missing:
            raise SystemExit(f"per_depth has no positive count for depths "
                             f"{missing}; either add them or drop them from "
                             f"`depths`.")
    seeds = cfg.get("seeds", 2)
    distractor_level = cfg.get("distractor_level", 1)
    max_retries = cfg.get("max_retries", 1)
    feedback = FeedbackMode(cfg.get("feedback", "structured"))
    models = cfg.get("models", [])

    if args.pilot:
        print("=== PILOT MODE: 2 depths, 5 tasks/depth, 1 seed ===\n")
        depths, per_depth, seeds = [1, 4], 5, 1

    if args.models:
        models = [m for m in models if m["name"] in args.models]

    if not models:
        raise SystemExit("No models configured. Check config/default.yaml "
                         "or pass --models.")

    os.makedirs(OUT_DIR, exist_ok=True)
    cache_dir = cfg.get("paths", {}).get("cache", "data/cache")

    # Map the config's family strings to integer group indices for the
    # hierarchical model's partial pooling. Without this, every model lands in
    # its own group and the pooling does nothing.
    fam_names = []
    for m in models:
        f = m.get("family", m["name"])
        if f not in fam_names:
            fam_names.append(f)
    fam_to_idx = {f: i for i, f in enumerate(fam_names)}
    print(f"Model families for pooling: {fam_to_idx}")

    p_rows, f_rows, succ_rows, tri_rows, group, names, backend_stats = \
        [], [], [], [], [], [], {}
    delta_results, L_ci, extra = {}, {}, {}

    for gi, m in enumerate(models):
        print(f"\n=== running {m['name']} ===")
        records, stats = run_model(m, depths, per_depth, seeds, max_retries,
                                   distractor_level, feedback, args.call_mode,
                                   cache_dir)
        backend_stats[m["name"]] = stats
        print(f"  done. {stats}")

        path = f"{OUT_DIR}/{tag}_{m['name'].replace('/', '_')}.jsonl"
        with open(path, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

        loaded = load_records(path)
        stats_by_depth = aggregate_by_depth(loaded, depths)
        p, f_syn, filled = stats_to_arrays(stats_by_depth)
        succ = np.array([round(s.g_t * s.n_g) if s.n_g else 0 for s in stats_by_depth])
        tri = np.array([s.n_g for s in stats_by_depth])

        p_rows.append(p); f_rows.append(f_syn)
        succ_rows.append(succ); tri_rows.append(tri)
        group.append(fam_to_idx[m.get("family", m["name"])])
        names.append(m["name"])

        n_be = sum(s.n_backend_err for s in stats_by_depth)
        if n_be:
            print(f"  NOTE: {n_be} backend-call failures excluded from stats "
                 f"for {m['name']}")

        delta_results[m["name"]] = delta_1(loaded)
        L_ci[m["name"]] = {}
        for d in depths:
            Lp, lo, hi = bootstrap_L_ci(loaded, d, n_boot=1000, seed=13)
            L_ci[m["name"]][str(d)] = {"L": Lp, "lo": lo, "hi": hi}
        extra[m["name"]] = [{"depth": s.depth, "f_syn": s.f_syn,
                             "n_fresh_errors": s.n_fresh_errors,
                             "selection_acc": s.selection_acc,
                             "selection_gold_acc": s.selection_gold_acc,
                             "parse_fail_rate": s.parse_fail_rate,
                             "stalled_rate": s.stalled_rate,
                             "n_backend_err": s.n_backend_err}
                            for s in stats_by_depth]

    if args.pilot:
        print("\n=== PILOT COMPLETE ===")
        print("Check the printed backend stats above for failures/retries.")
        print("If everything looks sane, rerun without --pilot for the full sweep.")
        return

    p = np.array(p_rows); f_syn = np.array(f_rows)
    successes = np.array(succ_rows); trials = np.array(tri_rows)
    group = np.array(group)

    print("\nfitting hierarchical model ...")
    idata = build_and_sample(p, f_syn, successes, trials, group,
                             draws=1500, tune=1500, chains=4, seed=7,
                             target_accept=0.97)

    with open(f"{OUT_DIR}/{tag}_idata.pkl", "wb") as fh:
        pickle.dump(idata, fh)
    with open(f"{OUT_DIR}/{tag}_meta.json", "w") as fh:
        json.dump({"names": names, "group": group.tolist(), "depths": depths,
                  "p": p.tolist(), "f_syn": f_syn.tolist(),
                  "successes": successes.tolist(), "trials": trials.tolist(),
                  "backend_stats": backend_stats, "models_config": models,
                  "delta_1": delta_results, "L_ci": L_ci,
                  "per_depth_extra": extra},
                 fh, indent=2)

    print(f"\nsaved -> {OUT_DIR}/{tag}_idata.pkl, {tag}_meta.json")
    print(f"Next: python -m tur.analysis.plots --tag {tag}   "
         "(use `py` on Windows) to generate the figures.")


if __name__ == "__main__":
    main()
