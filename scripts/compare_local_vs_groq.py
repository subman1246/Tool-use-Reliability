"""Is local-quantized inference the SAME CONDITION as the frozen Groq-hosted run?

Run:  py scripts/compare_local_vs_groq.py --local-tag localval \
          --local-model local/meta-llama/Llama-3.1-8B-Instruct \
          --frozen-model groq/llama-3.1-8b-instant

This answers one question and refuses to answer it vaguely: may results from the local
backend be pooled with, or compared directly against, the frozen provider-hosted numbers,
or must they be reported as a separate condition?

WHY THE QUESTION IS NOT RHETORICAL

4-bit weights, a different chat template, a different sampling stack and different
hardware are four independent reasons a local run can differ from a hosted endpoint. The
paper's central quantities are p_d and g_d, and a shift in either changes L_d = 1 - g_d/p_d
directly. If the local backend reproduces the frozen numbers we gain the retired models
back; if it does not, every local result is a new condition and saying otherwise would
silently merge two populations, which is the same error the cache-namespacing guard in
local_backend.py exists to prevent at the storage layer.

WHAT IS COMPARED

The SAME TASKS. Task identity depends only on (depth, k, base_seed), so a local sweep run
with the same seeds produces the same task set, and the comparison is paired at the task
level rather than being two independent samples that happen to share a depth. Tasks
present in only one arm are excluded and counted, never silently pooled.

  p_d   teacher-forced invocation accuracy at depth d
  g_d   free-running invocation accuracy at depth d
  L_d   1 - g_d/p_d

THE VERDICT RULE, FIXED IN ADVANCE

Stated here rather than chosen after seeing the numbers. Per depth, on the paired task
set, the arms are treated as the same condition only if BOTH:

  - |p_local - p_frozen| <= TOL and |g_local - g_frozen| <= TOL, and
  - the 89% paired bootstrap interval for each difference contains zero

TOL is 0.05, deliberately loose: the claim being tested is "close enough to pool", and a
tight threshold would fail on sampling noise at these sample sizes while a loose one that
still excludes zero is genuinely disqualifying. A single depth failing is enough to make
the whole arm a separate condition -- pooling at some depths and not others would produce
a dataset whose meaning changes along its main axis.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

RES = "data/results"
TOL = 0.05
N_BOOT = 10000
SEED = 20260829


def load(tag: str, model: str) -> list[dict]:
    p = os.path.join(RES, "%s_%s.jsonl" % (tag, model.replace("/", "_")))
    if not os.path.exists(p):
        raise SystemExit(
            "no records at %s\n"
            "Run the local sweep on Colab first and copy its jsonl here." % p)
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def by_task_depth(rows: list[dict], mode: str) -> dict:
    """{(task_id, depth): [correct, ...]} for one run mode."""
    out: dict = {}
    for r in rows:
        if r["run_mode"] != mode or r.get("backend_error"):
            continue
        out.setdefault((r["task_id"], r["depth"]), []).append(
            bool(r["args_correct_strict"]))
    return out


def paired_diff_ci(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float]:
    if not len(a):
        return float("nan"), float("nan")
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(np.percentile(d, 5.5)), float(np.percentile(d, 94.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-tag", default="localval")
    ap.add_argument("--local-model", required=True)
    ap.add_argument("--frozen-tag", default="real")
    ap.add_argument("--frozen-model", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    loc = load(args.local_tag, args.local_model)
    fro = load(args.frozen_tag, args.frozen_model)
    rng = np.random.default_rng(SEED)

    print("=" * 80)
    print("LOCAL-QUANTIZED vs FROZEN PROVIDER-HOSTED")
    print("=" * 80)
    print("local : %s  (tag %s, %d rows)" % (args.local_model, args.local_tag, len(loc)))
    print("frozen: %s  (tag %s, %d rows)" % (args.frozen_model, args.frozen_tag, len(fro)))
    print("tolerance %.2f on p_d and g_d, plus an 89%% paired interval containing zero"
          % TOL)

    report: dict = {"local_model": args.local_model, "frozen_model": args.frozen_model,
                    "tol": TOL, "depths": {}, "verdict": None}
    verdict_same = True

    for mode, sym in (("teacher_forced", "p"), ("free", "g")):
        L = by_task_depth(loc, mode)
        F = by_task_depth(fro, mode)
        shared_depths = sorted({d for _, d in L} & {d for _, d in F})
        print("")
        print("-" * 80)
        print("%s_d  (%s)" % (sym, mode))
        print("-" * 80)
        print("  %5s %7s %9s %9s %9s %22s  %s"
              % ("depth", "n_task", "local", "frozen", "diff", "89% interval", "status"))
        for d in shared_depths:
            ids = sorted({t for t, dd in L if dd == d} & {t for t, dd in F if dd == d})
            only_l = len({t for t, dd in L if dd == d}) - len(ids)
            only_f = len({t for t, dd in F if dd == d}) - len(ids)
            if not ids:
                continue
            a = np.array([np.mean(L[(t, d)]) for t in ids])
            b = np.array([np.mean(F[(t, d)]) for t in ids])
            lo, hi = paired_diff_ci(a, b, rng)
            diff = a.mean() - b.mean()
            ok = abs(diff) <= TOL and lo <= 0 <= hi
            verdict_same &= ok
            print("  %5d %7d %9.3f %9.3f %+9.3f   [%+.3f, %+.3f]  %s%s"
                  % (d, len(ids), a.mean(), b.mean(), diff, lo, hi,
                     "same" if ok else "DIFFERS",
                     "" if not (only_l or only_f)
                     else "  (excluded: %d local-only, %d frozen-only)" % (only_l, only_f)))
            report["depths"]["%s|%d" % (sym, d)] = {
                "n_tasks": len(ids), "local": float(a.mean()), "frozen": float(b.mean()),
                "diff": float(diff), "lo": lo, "hi": hi, "same": bool(ok),
                "excluded_local_only": only_l, "excluded_frozen_only": only_f}

    print("")
    print("=" * 80)
    if verdict_same:
        report["verdict"] = "same_condition"
        print("VERDICT: SAME CONDITION.")
        print("  Local-quantized output matches the frozen provider-hosted numbers")
        print("  within tolerance at every shared depth, in both run modes. Local")
        print("  results may be reported as a continuation of the frozen arm.")
    else:
        report["verdict"] = "separate_condition"
        print("VERDICT: SEPARATE CONDITION.")
        print("  At least one depth diverges. Local results are NOT a continuation of")
        print("  the frozen arm and must be reported as their own condition, tagged as")
        print("  local-quantized wherever they appear. Do not pool, and do not compare")
        print("  a local number against a frozen one as though they measured the same")
        print("  thing.")
    print("=" * 80)

    dest = args.out or os.path.join(RES, "%s_equivalence.json" % args.local_tag)
    with open(dest, "w") as fh:
        json.dump(report, fh, indent=2)
    print("saved -> " + dest)


if __name__ == "__main__":
    main()
