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
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from tur.tasks.dag import generate_suite, generate_routing_suite
from tur.harness.cache import Cache
from tur.harness.executor import FeedbackMode
from tur.harness.runner import (run_free, run_teacher_forced, LiteLLMBackend,
                                RateLimiter, DailyCapReached)
from tur.analysis.aggregate import (load_records, aggregate_by_depth,
                                    stats_to_arrays, bootstrap_L_ci,
                                    aggregate_by_step, measure_recovery)
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
TPD_PATH = "data/results/discovered_tpd.json"

_print_lock = threading.Lock()


def log(lines: list[str], msg: str) -> None:
    """Buffer a model's output, and echo progress under a lock.

    Models run concurrently, so unsynchronised prints interleave into something
    unreadable. Each worker accumulates its own transcript for the end-of-model
    report and only the short progress lines go to the console live.
    """
    lines.append(msg)
    with _print_lock:
        print(msg, flush=True)


def _suite_for(variant: str, depths, per_depth, distractor_level: int, seed: int):
    gen = generate_routing_suite if variant == "routing" else generate_suite
    return gen(depths, per_depth, distractor_level, base_seed=seed * 31 + 1000)


def load_discovered_tpd(path: str = TPD_PATH) -> dict[str, int]:
    """Per-model daily TOKEN limits learned from previous runs.

    No provider exposes a daily token limit in any response header; the only
    place the number appears is the body of the 429 that enforces it. So the
    limiter seeds a conservative lower bound and learns the real value the first
    time it is refused, and the value is persisted here so that later days plan
    against a measured limit instead of rediscovering it by getting blocked.
    """
    try:
        with open(path) as fh:
            return {k: int(v) for k, v in json.load(fh).items()}
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return {}


def save_discovered_tpd(discovered: dict[str, int], path: str = TPD_PATH) -> None:
    if not discovered:
        return
    merged = load_discovered_tpd(path)
    for name, value in discovered.items():
        # keep the largest value ever observed: a limit read off a 429 is a real
        # ceiling, and a smaller later reading usually means the window had not
        # reset rather than that the ceiling moved down
        if value and value > merged.get(name, 0):
            merged[name] = int(value)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(merged, fh, indent=2, sort_keys=True)


def structural_anomalies(stats_by_depth, loaded: list[dict],
                         depths: list[int]) -> list[str]:
    """Flag results that look structurally wrong rather than merely bad.

    A weak model scoring badly is a result. A model whose p_t and g_t are
    identical at every depth, or whose errors all land in one bucket, is a signal
    that something in the harness or the scorer is not doing its job -- exactly
    the failure mode that made the first four simulated runs meaningless. These
    are reported so the run can be stopped and understood, not silently averaged
    into the suite.
    """
    out = []
    p = [s.p_t for s in stats_by_depth]
    g = [s.g_t for s in stats_by_depth]

    same = [d for d, a, b in zip(depths, p, g)
            if a == a and b == b and abs(a - b) < 1e-9]
    if len(same) == len([a for a in p if a == a]) and len(same) > 1:
        out.append(f"p_t and g_t are IDENTICAL at every depth {same}: the free "
                   f"and teacher-forced arms are not diverging at all, which is "
                   f"what a degenerate task suite looks like (the linear suite "
                   f"did exactly this)")

    deep = [(d, a) for d, a in zip(depths, p) if d > 1 and a == a]
    if deep and all(a >= 0.999 for _, a in deep):
        out.append(f"p_t is 1.000 at every depth beyond 1: no errors at all, so "
                   f"there is nothing for propagation to act on")
    if deep and all(a <= 0.02 for _, a in deep):
        out.append(f"p_t <= 0.02 at every depth beyond 1: near-total failure, "
                   f"which usually means a prompt/parse problem rather than model "
                   f"incapacity -- check raw rows before believing it")

    buckets: dict[str, int] = {}
    for r in loaded:
        if r["run_mode"] == "free" and not r.get("backend_error", False) \
                and not r["args_correct_strict"]:
            buckets[r.get("error_type") or "none"] = \
                buckets.get(r.get("error_type") or "none", 0) + 1
    total = sum(buckets.values())
    if total >= 20 and len(buckets) == 1:
        only = next(iter(buckets))
        out.append(f"all {total} errors fall in ONE bucket ({only!r}): the "
                   f"error-type decomposition has nothing to decompose, which is "
                   f"how the mis-bucketed selection errors presented")

    pf = [s.parse_fail_rate for s in stats_by_depth if s.parse_fail_rate == s.parse_fail_rate]
    if pf and min(pf) > 0.5:
        out.append(f"parse failures exceed 50% at every depth (min "
                   f"{min(pf):.2f}): scores measure formatting, not tool use")
    return out


def run_model(model_cfg: dict, depths: list[int], per_depth, seeds: int,
             max_retries: int, distractor_level: int, feedback: FeedbackMode,
             call_mode: str, cache_dir: str, headroom: float = 0.80,
             variant: str = "routing", tpd: int | None = None,
             lines: list[str] | None = None
             ) -> tuple[list[dict], dict, bool]:
    """Run one model's full sweep on one task variant.

    Returns (records, backend stats, hit_daily_cap). The cap flag is returned
    rather than swallowed: a sweep cut short by an exhausted allowance is not a
    finished sweep, and the caller must not persist it as one.
    """
    lines = [] if lines is None else lines
    name = model_cfg["name"]
    cache = Cache(f"{cache_dir}/{name.replace('/', '_')}")
    limiter = RateLimiter(tpm=model_cfg.get("tpm"), rpd=model_cfg.get("rpd"),
                          tpd=tpd, headroom=headroom)
    backend = LiteLLMBackend(name, temperature=0.0, cache=cache, limiter=limiter)
    if limiter.tpm or limiter.rpd:
        log(lines, f"  {name} [{variant}] pacing: tpm={limiter.tpm} "
                   f"rpd={limiter.rpd} tpd={limiter.tpd} "
                   f"headroom={headroom:.0%} "
                   f"(effective {limiter.budget:,.0f} tok/min)")
    else:
        log(lines, f"  WARNING: no rpd/tpm in config for {name}; running unpaced")

    records = []
    for seed in range(seeds):
        suite = _suite_for(variant, depths, per_depth, distractor_level, seed)
        for i, task in enumerate(suite):
            try:
                f = run_free(task, backend, call_mode, feedback, max_retries)
                t = run_teacher_forced(task, backend, call_mode, feedback,
                                       max_retries)
            except DailyCapReached as e:
                log(lines, f"\n  !! DAILY CAP REACHED for {name} [{variant}] at "
                           f"seed {seed}, task {i + 1}/{len(suite)}: {e}")
                return records, backend.stats(), True
            records += [r.__dict__ for r in f] + [r.__dict__ for r in t]
            if (i + 1) % 25 == 0:
                paced = backend.stats().get("limiter", {}).get("paced_sleep_s", 0)
                log(lines, f"  {name} [{variant}]: seed {seed}, "
                           f"{i + 1}/{len(suite)} tasks "
                           f"(calls={backend.n_calls} "
                           f"cache_hits={backend.n_cache_hits} "
                           f"failures={backend.n_failures} "
                           f"429s={backend.n_rate_limited} paced_sleep={paced}s)")
    return records, backend.stats(), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--pilot", action="store_true",
                    help="tiny run (2 depths, 5 tasks/depth, 1 seed) to sanity "
                         "check everything before spending real budget")
    ap.add_argument("--models", nargs="+", default=None,
                    help="override the model list from config, by name")
    ap.add_argument("--call-mode", choices=["uniform", "native"], default="uniform")
    ap.add_argument("--headroom", type=float, default=0.80,
                    help="fraction of each model's TPM ceiling to pace at "
                         "(default: %(default)s). Below 1.0 on purpose: the "
                         "token estimate is approximate and overshooting just "
                         "converts into provider 429s and wasted retries.")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="run tag for output filenames (default: %(default)s). "
                         "Use a distinct tag for ablations so they don't "
                         "overwrite the main run, then pass the same tag to "
                         "tur.analysis.plots --tag.")
    ap.add_argument("--jobs", type=int, default=0,
                    help="models to sweep concurrently (default: all of them). "
                         "Rate limits are enforced PER MODEL, so concurrent "
                         "models do not compete for the same allowance and "
                         "calendar time becomes the slowest model rather than "
                         "the sum. 1 forces the old sequential behaviour.")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the control arm (see config: control_arm)")
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
    variant = cfg.get("task_variant", "routing")

    # The control arm is a second, much smaller sweep on the OTHER task variant.
    # Its expected result is L_t ~ 0, and it is run per model alongside the
    # primary arm rather than as a separate invocation so the two always come
    # from the same models, the same day, and the same code.
    ctrl = cfg.get("control_arm") or {}
    if ctrl and not args.no_control:
        ctrl_depths = ctrl.get("depths", [1, 4, 8])
        ctrl_pd = ctrl.get("per_depth", 20)
        if isinstance(ctrl_pd, dict):
            ctrl_pd = {int(k): int(v) for k, v in ctrl_pd.items()}
            missing = [d for d in ctrl_depths if not ctrl_pd.get(d)]
            if missing:
                raise SystemExit(f"control_arm.per_depth has no positive count "
                                 f"for depths {missing}")
        ctrl_variant = ctrl.get("task_variant",
                                "linear" if variant == "routing" else "routing")
    else:
        ctrl_depths = ctrl_pd = ctrl_variant = None

    if args.pilot:
        print("=== PILOT MODE: 2 depths, 5 tasks/depth, 1 seed ===\n")
        depths, per_depth, seeds = [1, 4], 5, 1
        ctrl_depths = ctrl_pd = ctrl_variant = None

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
    extra_filled: dict[str, list[int]] = {}   # model -> depths with substituted inputs
    recov, per_step = {}, {}

    control, anomalies = {}, {}
    known_tpd = load_discovered_tpd()
    if known_tpd:
        print(f"Seeding limiters with previously discovered TPD: {known_tpd}")

    def sweep_one(m: dict) -> dict:
        """One model: primary arm, then control arm. Runs in its own thread.

        Safe to run concurrently because everything a model touches is its own:
        its rate limiter, its response cache directory, and its output file.
        Rate limits are enforced per model, so two models pacing at once are not
        drawing down a shared allowance -- which is the whole reason this is
        worth parallelising. Calendar time becomes the slowest model instead of
        the sum over models.
        """
        name = m["name"]
        lines: list[str] = []
        log(lines, f"\n=== running {name} [{variant}] ===")
        recs, stats, capped = run_model(
            m, depths, per_depth, seeds, max_retries, distractor_level,
            feedback, args.call_mode, cache_dir, args.headroom,
            variant=variant, tpd=known_tpd.get(name), lines=lines)
        out = {"model": m, "records": recs, "stats": stats, "capped": capped,
               "lines": lines, "control": None}
        if capped:
            return out
        if ctrl_variant:
            log(lines, f"  {name}: primary arm done, running "
                       f"{ctrl_variant} control arm")
            c_recs, c_stats, c_capped = run_model(
                m, ctrl_depths, ctrl_pd, seeds, max_retries, distractor_level,
                feedback, args.call_mode, cache_dir, args.headroom,
                variant=ctrl_variant, tpd=known_tpd.get(name), lines=lines)
            out["control"] = {"records": c_recs, "stats": c_stats,
                              "capped": c_capped}
        return out

    n_jobs = args.jobs if args.jobs > 0 else len(models)
    n_jobs = max(1, min(n_jobs, len(models)))
    print(f"\nsweeping {len(models)} model(s) with {n_jobs} concurrent job(s)")
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        # map preserves input order, so the results list is deterministic even
        # though the sweeps finish in whatever order the providers allow
        results = list(pool.map(sweep_one, models))

    # Persist any daily token limits learned from 429 bodies before doing
    # anything that can fail: these are expensive to discover (each one costs a
    # blocked run) and are worth keeping even from a run that then halts.
    save_discovered_tpd({r["model"]["name"]: r["stats"].get("limiter", {})
                        .get("tpd_discovered")
                         for r in results
                         if r["stats"].get("limiter", {}).get("tpd_discovered")})

    capped_runs = [r for r in results if r["capped"]
                   or (r["control"] or {}).get("capped")]
    if capped_runs:
        for r in capped_runs:
            nm = r["model"]["name"].replace("/", "_")
            part = f"{OUT_DIR}/{tag}_{nm}.partial.jsonl"
            with open(part, "w") as fh:
                for rec in r["records"]:
                    fh.write(json.dumps(rec) + "\n")
        print("\n" + "=" * 70)
        print("RUN HALTED: daily allowance exhausted")
        print("=" * 70)
        for r in capped_runs:
            print(f"  capped       : {r['model']['name']}")
            print(f"  records kept : {len(r['records'])} -> "
                  f"{tag}_{r['model']['name'].replace('/', '_')}.partial.jsonl")
            print(f"  backend      : {r['stats']}")
        done = [r["model"]["name"] for r in results
                if not (r["capped"] or (r["control"] or {}).get("capped"))]
        print(f"  completed    : {done}")
        print()
        print("  Nothing was written to the normal output path for a capped")
        print("  model, so a partial sweep cannot be analysed as if it were")
        print("  complete. Every successful call is in the response cache, so")
        print("  re-running the same command once the allowance resets replays")
        print("  finished work for free and continues from there.")
        if known_tpd:
            print(f"  discovered TPD so far -> {TPD_PATH}")
        print("=" * 70)
        raise SystemExit(2)

    for r in results:
        m, records, stats = r["model"], r["records"], r["stats"]
        backend_stats[m["name"]] = stats
        print(f"  {m['name']}: done. {stats}")

        path = f"{OUT_DIR}/{tag}_{m['name'].replace('/', '_')}.jsonl"
        with open(path, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

        loaded = load_records(path)
        stats_by_depth = aggregate_by_depth(loaded, depths)
        p, f_syn, filled = stats_to_arrays(stats_by_depth)
        # stats_to_arrays substitutes a neighbouring depth's value wherever a
        # bin is degenerate (p_t NaN or 0, or no fresh errors to estimate f_syn
        # from). The fit then treats that substitute as a measurement. On weak
        # models at depth this is likely, not hypothetical, so say so loudly --
        # run_full_analysis printed this note and run_real_suite silently
        # dropped it.
        if any(filled):
            bad = [depths[i] for i, fl in enumerate(filled) if fl]
            print(f"  WARNING: {m['name']} needed substituted inputs at "
                 f"depth(s) {bad} -- p_t and/or f_syn were degenerate there "
                 f"and were filled from a neighbouring depth. The fit for this "
                 f"model is partly driven by placeholder values; L_t at those "
                 f"depths is the trustworthy quantity.")
            extra_filled[m["name"]] = bad
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
        recov[m["name"]] = measure_recovery(loaded)
        per_step[m["name"]] = [st.__dict__ for st in aggregate_by_step(loaded)]
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

        # per-model report: correct-invocation rates by depth, then anomalies
        print(f"  {m['name']} correct-invocation rate by depth "
              f"(p_t teacher-forced / g_t free / L_t):")
        for s in stats_by_depth:
            print(f"    d{s.depth}: p_t={s.p_t:.3f} g_t={s.g_t:.3f} "
                  f"L_t={s.L_t:+.3f}  (n_p={s.n_p} n_g={s.n_g})")
        flags = structural_anomalies(stats_by_depth, loaded, depths)
        if flags:
            anomalies[m["name"]] = flags
            print(f"  !! STRUCTURAL ANOMALIES for {m['name']}:")
            for fl in flags:
                print(f"     - {fl}")

        # control arm: expected to be a null (L_t ~ 0)
        if r["control"]:
            c_path = (f"{OUT_DIR}/{tag}_control_"
                      f"{m['name'].replace('/', '_')}.jsonl")
            with open(c_path, "w") as fh:
                for rec in r["control"]["records"]:
                    fh.write(json.dumps(rec) + "\n")
            c_loaded = load_records(c_path)
            c_stats = aggregate_by_depth(c_loaded, ctrl_depths)
            c_L = {str(d): dict(zip(("L", "lo", "hi"),
                                    bootstrap_L_ci(c_loaded, d, 1000, seed=13)))
                   for d in ctrl_depths}
            control[m["name"]] = {
                "variant": ctrl_variant, "depths": ctrl_depths,
                "per_depth": ctrl_pd, "L_ci": c_L,
                "backend_stats": r["control"]["stats"],
                "by_depth": [{"depth": s.depth, "p_t": s.p_t, "g_t": s.g_t,
                              "L_t": s.L_t, "n_p": s.n_p, "n_g": s.n_g,
                              "f_syn": s.f_syn,
                              "n_fresh_errors": s.n_fresh_errors}
                             for s in c_stats]}
            print(f"  {m['name']} CONTROL arm ({ctrl_variant}), "
                  f"expected L_t ~ 0:")
            for s in c_stats:
                ci = c_L[str(s.depth)]
                print(f"    d{s.depth}: p_t={s.p_t:.3f} g_t={s.g_t:.3f} "
                      f"L_t={s.L_t:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]")

    if anomalies:
        print("\n" + "=" * 70)
        print("STRUCTURAL ANOMALIES DETECTED -- read these before the analysis")
        print("=" * 70)
        for nm, flags in anomalies.items():
            print(f"  {nm}:")
            for fl in flags:
                print(f"    - {fl}")
        print("  Results were still written; the fit below may be meaningless.")
        print("=" * 70)

    if args.pilot:
        print("\n=== PILOT COMPLETE ===")
        print("Check the printed backend stats above for failures/retries.")
        print("If everything looks sane, rerun without --pilot for the full sweep.")
        return

    p = np.array(p_rows); f_syn = np.array(f_rows)
    successes = np.array(succ_rows); trials = np.array(tri_rows)
    group = np.array(group)

    # measured recovery rates -> informative prior centres, per family
    G = int(group.max()) + 1

    def fam_mean(key):
        out = []
        for g in range(G):
            vals = [recov[n][key] for i, n in enumerate(names)
                    if group[i] == g and not np.isnan(recov[n][key])]
            out.append(float(np.mean(vals)) if vals else float("nan"))
        return out

    prior_rs, prior_rm = fam_mean("r_syn_chain"), fam_mean("r_sem_chain")
    print(f"measured recovery -> prior centres: r_syn={prior_rs} r_sem={prior_rm}")

    print("\nfitting hierarchical model ...")
    idata = build_and_sample(p, f_syn, successes, trials, group,
                             draws=1500, tune=1500, chains=4, seed=7,
                             target_accept=0.97,
                             prior_r_syn=prior_rs, prior_r_sem=prior_rm)

    with open(f"{OUT_DIR}/{tag}_idata.pkl", "wb") as fh:
        pickle.dump(idata, fh)
    with open(f"{OUT_DIR}/{tag}_meta.json", "w") as fh:
        json.dump({"names": names, "group": group.tolist(), "depths": depths,
                  "p": p.tolist(), "f_syn": f_syn.tolist(),
                  "successes": successes.tolist(), "trials": trials.tolist(),
                  "backend_stats": backend_stats, "models_config": models,
                  "delta_1": delta_results, "L_ci": L_ci,
                  "per_depth_extra": extra,
                  "substituted_input_depths": extra_filled,
                  "measured_recovery": recov, "per_step": per_step,
                  "priors_used": {"r_syn": prior_rs, "r_sem": prior_rm},
                  "task_variant": variant, "control_arm": control,
                  "structural_anomalies": anomalies,
                  "discovered_tpd": load_discovered_tpd()},
                 fh, indent=2)

    print(f"\nsaved -> {OUT_DIR}/{tag}_idata.pkl, {tag}_meta.json")
    print(f"Next: python -m tur.analysis.plots --tag {tag}   "
         "(use `py` on Windows) to generate the figures.")


if __name__ == "__main__":
    main()
