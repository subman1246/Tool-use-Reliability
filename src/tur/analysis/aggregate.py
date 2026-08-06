"""Aggregate raw StepRecord logs into the quantities the model and figures need.

Guards against the edge cases that show up once models get weak or flaky:
  - p_t = 0 (every teacher-forced attempt at that depth failed): L_t is
    undefined (0/0 or division by zero), reported as NaN rather than crashing
    or silently returning inf.
  - f_syn(t) with zero errors at a depth: reported as NaN, not 0/0.
  - empty bins (no records at a depth for a model): reported as NaN, not a
    KeyError.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class DepthStats:
    depth: int
    p_t: float          # teacher-forced correct-invocation rate (strict)
    g_t: float           # free-run correct-invocation rate (strict)
    n_p: int              # teacher-forced sample count at this depth
    n_g: int              # free-run sample count at this depth
    f_syn: float         # syntactic share of FRESH errors (clean-context only)
    n_fresh_errors: int   # fresh-error count feeding f_syn
    L_t: float            # 1 - g_t / p_t, NaN if p_t == 0
    selection_acc: float  # conditional selection accuracy (given ref held)
    selection_gold_acc: float  # agreement with the gold trajectory
    parse_fail_rate: float
    stalled_rate: float   # share of steps entered on a stalled chain


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _safe_mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def aggregate_by_depth(records: list[dict], depths: list[int]) -> list[DepthStats]:
    tf = defaultdict(list)   # depth -> list of records, run_mode == teacher_forced
    fr = defaultdict(list)   # depth -> list of records, run_mode == free
    for r in records:
        (tf if r["run_mode"] == "teacher_forced" else fr)[r["depth"]].append(r)

    out = []
    for d in depths:
        tf_d = tf.get(d, [])
        fr_d = fr.get(d, [])

        p_t = _safe_mean([r["args_correct_strict"] for r in tf_d])
        g_t = _safe_mean([r["args_correct_strict"] for r in fr_d])
        sel_acc = _safe_mean([r.get("selection_correct", False) for r in fr_d])
        sel_gold = _safe_mean([r.get("selection_matches_gold",
                                     r.get("selection_correct", False))
                               for r in fr_d])
        parse_fail = _safe_mean([r["error_type"] == "syntactic" for r in fr_d])
        stalled_rate = _safe_mean([bool(r.get("stalled_in", False)) for r in fr_d])

        # f_syn must be estimated from FRESH errors only. A step that is wrong
        # solely because its input was already poisoned upstream is not new
        # evidence about how often this model produces syntactic vs semantic
        # failures, and counting it inflates the semantic share with what is
        # really propagation. Restrict to steps entered on a clean context.
        fresh = [r for r in fr_d
                 if r.get("context_clean_in", True)
                 and not r.get("stalled_in", False)
                 and not r["args_correct_strict"]]
        n_syn = sum(1 for r in fresh if r["error_type"] == "syntactic")
        f_syn = (n_syn / len(fresh)) if fresh else float("nan")

        if not tf_d or p_t == 0 or np.isnan(p_t):
            L_t = float("nan")
        else:
            L_t = 1.0 - g_t / p_t if not np.isnan(g_t) else float("nan")

        out.append(DepthStats(depth=d, p_t=p_t, g_t=g_t, n_p=len(tf_d),
                              n_g=len(fr_d), f_syn=f_syn,
                              n_fresh_errors=len(fresh),
                              L_t=L_t, selection_acc=sel_acc,
                              selection_gold_acc=sel_gold,
                              parse_fail_rate=parse_fail,
                              stalled_rate=stalled_rate))
    return out


def bootstrap_L_ci(records: list[dict], depth: int, n_boot: int = 2000,
                   alpha: float = 0.11, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap a confidence interval for L_t = 1 - g_t/p_t at one depth.

    L_t is a ratio of two independently estimated rates, so its uncertainty is
    not simply the uncertainty of either one. Resampling tasks (not individual
    steps) preserves within-task correlation, which matters because steps in
    the same chain are not independent once propagation is in play.

    Returns (L_point, lo, hi) at the (1-alpha) level; defaults to an 89% CI.
    """
    rng = np.random.default_rng(seed)
    tf = [r for r in records if r["run_mode"] == "teacher_forced" and r["depth"] == depth]
    fr = [r for r in records if r["run_mode"] == "free" and r["depth"] == depth]
    if not tf or not fr:
        return float("nan"), float("nan"), float("nan")

    def by_task(rows):
        d = {}
        for r in rows:
            d.setdefault(r["task_id"], []).append(r["args_correct_strict"])
        return list(d.values())

    tf_tasks, fr_tasks = by_task(tf), by_task(fr)
    if not tf_tasks or not fr_tasks:
        return float("nan"), float("nan"), float("nan")

    def rate(task_groups, idx):
        vals = [v for i in idx for v in task_groups[i]]
        return float(np.mean(vals)) if vals else float("nan")

    p_point = rate(tf_tasks, range(len(tf_tasks)))
    g_point = rate(fr_tasks, range(len(fr_tasks)))
    L_point = 1 - g_point / p_point if p_point else float("nan")

    draws = []
    for _ in range(n_boot):
        ti = rng.integers(0, len(tf_tasks), len(tf_tasks))
        fi = rng.integers(0, len(fr_tasks), len(fr_tasks))
        p_b, g_b = rate(tf_tasks, ti), rate(fr_tasks, fi)
        if p_b and not np.isnan(p_b) and not np.isnan(g_b):
            draws.append(1 - g_b / p_b)
    if not draws:
        return L_point, float("nan"), float("nan")
    lo = float(np.percentile(draws, 100 * alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return L_point, lo, hi


def stats_to_arrays(stats: list[DepthStats]):
    """Return p, f_syn arrays with NaNs filled by nearest valid neighbour.

    The hierarchical model needs a value at every depth to drive the state
    recurrence. A NaN from an empty or degenerate bin is filled from the
    nearest depth that has data (forward, then backward), which is a
    conservative placeholder that should be flagged in diagnostics, not a
    substitute for having enough samples.
    """
    p = np.array([s.p_t for s in stats], dtype=float)
    f = np.array([s.f_syn for s in stats], dtype=float)
    filled_flags = np.isnan(p) | np.isnan(f)
    p = _fill_nan(p)
    f = _fill_nan(f, default=0.5)
    return p, f, filled_flags


def _fill_nan(arr: np.ndarray, default: float = None) -> np.ndarray:
    arr = arr.copy()
    n = len(arr)
    # forward fill
    for i in range(1, n):
        if np.isnan(arr[i]) and not np.isnan(arr[i - 1]):
            arr[i] = arr[i - 1]
    # backward fill
    for i in range(n - 2, -1, -1):
        if np.isnan(arr[i]) and not np.isnan(arr[i + 1]):
            arr[i] = arr[i + 1]
    if np.isnan(arr).any():
        arr[np.isnan(arr)] = default if default is not None else 0.5
    return arr
