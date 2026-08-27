"""Analyse the controlled error-injection arm.

Run:  py scripts/report_injection.py --tag inject

Why this is a separate script from report_real.py. That one is built around the
free/teacher_forced contrast and filters on run_mode == "free"; injection records
carry run_mode inj_clean / inj_corrupt and are PAIRED, which changes the estimator.
Feeding them through the unpaired machinery would throw away the pairing that is the
entire reason the arm exists.

THE ESTIMAND

Each pair contributes two calls built from a byte-identical prefix, differing in one
substituted number. Severity at position j under corruption mode m is

    pi_hat(j, m) = P(correct | clean prefix) - P(correct | corrupted prefix)

with "correct" read conditional-on-state: correct GIVEN the value actually held. Because
the two members share a prefix by construction, this difference is a causal effect of
context corruption rather than an association with it. That is the distinction from every
other severity number in the project, all of which are observational -- in the free arm
the model chooses its own corruption, so whatever made it err upstream is still present
downstream.

WHY THE PAIRED TEST, NOT A TWO-PROPORTION TEST

Treating the clean and corrupted arms as independent samples discards the matching and
inflates the interval: the two members share task, prefix, branch structure and held
position, so their outcomes are strongly correlated. Both are reported here -- the paired
bootstrap resamples PAIRS, and McNemar's discordant counts (b, c) show directly how many
pairs actually flipped, which is the quantity a reader should be shown before believing
any difference of proportions.

WHAT WOULD FALSIFY WHAT

  pi_hat ~ 0 everywhere        corrupting the context does not degrade the next call, and
                               the propagation account loses its mechanism -- the observed
                               free-arm decline would have to come from something else.
  pi_hat >> observational      the observational estimate (+0.316 for allam on the routing
                               arm) is biased DOWN, not up.
  pi_hat rising in j           the step model's assumption that pi is constant along the
                               chain is wrong, and the recurrence is misspecified in a way
                               no aggregate fit could reveal.
  flip >> preserving           severity lives in the selection channel: corruption hurts
                               because it changes which tool is right.
  preserving comparable        a wrong value degrades the call even when the correct tool
                               is unchanged, i.e. there is a pure argument-channel effect.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np

RES = "data/results"
N_BOOT = 10000
SEED = 20260827

# Observational conditional-severity estimates from the routing arm, for contrast only.
# These are what the causal numbers below are meant to be compared against.
OBSERVATIONAL = {"groq/allam-2-7b": 0.316, "groq/llama-3.1-8b-instant": 0.149}


def load_pairs(path: str):
    """Group records into (clean, corrupt) pairs keyed by pair_id."""
    with open(path) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    rows = [r for r in rows if r.get("pair_id")]
    by_pair: dict[str, dict] = collections.defaultdict(dict)
    for r in rows:
        if r.get("backend_error"):
            continue
        by_pair[r["pair_id"]][r["run_mode"]] = r
    pairs = []
    dropped = 0
    for pid, d in by_pair.items():
        if "inj_clean" in d and "inj_corrupt" in d:
            pairs.append((d["inj_clean"], d["inj_corrupt"]))
        else:
            # A half-pair is not usable: the whole design is the matched contrast.
            # Silently keeping the surviving member would turn a paired estimate into
            # a mongrel of paired and unpaired data.
            dropped += 1
    return pairs, dropped, len(rows)


def paired_boot(clean: np.ndarray, corr: np.ndarray, rng) -> tuple[float, float]:
    """Percentile CI for the mean difference, resampling PAIRS."""
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(N_BOOT, n))
    diffs = clean[idx].mean(axis=1) - corr[idx].mean(axis=1)
    return float(np.percentile(diffs, 5.5)), float(np.percentile(diffs, 94.5))


def cell_stats(pairs, field: str, rng):
    clean = np.array([c[field] for c, _ in pairs], dtype=float)
    corr = np.array([k[field] for _, k in pairs], dtype=float)
    lo, hi = paired_boot(clean, corr, rng)
    # McNemar discordant counts: b = clean right & corrupt wrong, c = the reverse
    b = int(sum(1 for c, k in pairs if c[field] and not k[field]))
    cc = int(sum(1 for c, k in pairs if k[field] and not c[field]))
    return {"n": len(pairs), "clean": float(clean.mean()) if len(clean) else float("nan"),
            "corrupt": float(corr.mean()) if len(corr) else float("nan"),
            "delta": float(clean.mean() - corr.mean()) if len(clean) else float("nan"),
            "lo": lo, "hi": hi, "b": b, "c": cc}


def fmt(s: dict) -> str:
    return ("n=%3d  clean=%.3f  corrupt=%.3f  delta=%+.3f  [%+.3f, %+.3f]  "
            "discordant b=%d c=%d" % (s["n"], s["clean"], s["corrupt"], s["delta"],
                                      s["lo"], s["hi"], s["b"], s["c"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="inject")
    ap.add_argument("--model", default="groq/allam-2-7b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = os.path.join(RES, args.tag + "_" + args.model.replace("/", "_") + ".jsonl")
    if not os.path.exists(path):
        raise SystemExit("no records at " + path + " -- has the sweep run?")

    pairs, dropped, n_rows = load_pairs(path)
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("CONTROLLED ERROR INJECTION -- " + args.model)
    print("=" * 78)
    print("records %d, complete pairs %d, incomplete pairs dropped %d"
          % (n_rows, len(pairs), dropped))
    if not pairs:
        raise SystemExit("no complete pairs")
    print("scoring: correct_given_state (correct GIVEN the value actually held)")
    print("interval: 89%% percentile bootstrap over PAIRS, %d resamples" % N_BOOT)

    positions = sorted({c["inject_at"] for c, _ in pairs})
    modes = sorted({c["injection"] for c, _ in pairs})

    out: dict = {"tag": args.tag, "model": args.model, "n_pairs": len(pairs),
                 "dropped": dropped, "cells": {}, "by_mode": {}, "channels": {}}

    for mode in modes:
        sub = [p for p in pairs if p[0]["injection"] == mode]
        print("")
        print("-" * 78)
        print("MODE: " + mode + ("   (correct TOOL changes -- selection channel)"
                                 if mode == "parity_flip"
                                 else "   (correct tool UNCHANGED -- argument channel)"))
        print("-" * 78)
        s = cell_stats(sub, "correct_given_state", rng)
        print("  ALL POSITIONS   " + fmt(s))
        out["by_mode"][mode] = s
        print("")
        for j in positions:
            cj = [p for p in sub if p[0]["inject_at"] == j]
            if not cj:
                continue
            sj = cell_stats(cj, "correct_given_state", rng)
            print("  j=%d             %s" % (j, fmt(sj)))
            out["cells"]["%s|j%d" % (mode, j)] = sj

    # Channel decomposition: which component of the call actually degrades.
    print("")
    print("-" * 78)
    print("CHANNEL DECOMPOSITION (all positions pooled)")
    print("-" * 78)
    for mode in modes:
        sub = [p for p in pairs if p[0]["injection"] == mode]
        print("  " + mode + ":")
        for field, label in (("selection_correct", "tool selection "),
                             ("args_correct_given_state", "argument       "),
                             ("correct_given_state", "both (overall) ")):
            s = cell_stats(sub, field, rng)
            print("    " + label + " " + fmt(s))
            out["channels"]["%s|%s" % (mode, field)] = s

    print("")
    print("-" * 78)
    print("CONTRAST WITH THE OBSERVATIONAL ESTIMATE")
    print("-" * 78)
    obs = OBSERVATIONAL.get(args.model)
    flip = out["by_mode"].get("parity_flip")
    if obs is not None and flip is not None:
        print("  observational (routing free arm, conditional) : %+.3f" % obs)
        print("  causal (parity_flip, all positions)           : %+.3f  [%+.3f, %+.3f]"
              % (flip["delta"], flip["lo"], flip["hi"]))
        inside = flip["lo"] <= obs <= flip["hi"]
        print("  observational value inside the causal 89% interval: "
              + ("YES" if inside else "NO"))
        if not inside:
            direction = "UNDER" if obs < flip["lo"] else "OVER"
            print("  -> the observational estimate is biased %s relative to the "
                  "causal one." % direction)
            print("     This is a discrepancy between two numbers the paper would")
            print("     otherwise treat as measuring the same quantity. Report it;")
            print("     do not reconcile it by preferring whichever is convenient.")

    # Position trend: is pi constant along the chain, as the step model assumes?
    print("")
    print("-" * 78)
    print("POSITION PROFILE (is pi constant along the chain?)")
    print("-" * 78)
    for mode in modes:
        ds = [(j, out["cells"]["%s|j%d" % (mode, j)]["delta"])
              for j in positions if "%s|j%d" % (mode, j) in out["cells"]]
        if len(ds) < 2:
            continue
        span = max(d for _, d in ds) - min(d for _, d in ds)
        trend = " -> ".join("j=%d:%+.3f" % (j, d) for j, d in ds)
        print("  " + mode + ": " + trend)
        print("      spread across positions: %.3f" % span)
        out["channels"]["%s|position_spread" % mode] = span

    dest = args.out or os.path.join(RES, args.tag + "_injection_summary.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print("")
    print("saved -> " + dest)


if __name__ == "__main__":
    main()
