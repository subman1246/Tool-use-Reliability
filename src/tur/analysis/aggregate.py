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
    n_backend_err: int = 0  # provider-call failures excluded from all other stats


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
        tf_d = [r for r in tf.get(d, []) if not r.get("backend_error", False)]
        fr_d = [r for r in fr.get(d, []) if not r.get("backend_error", False)]
        n_backend_err = (sum(1 for r in tf.get(d, []) if r.get("backend_error", False))
                        + sum(1 for r in fr.get(d, []) if r.get("backend_error", False)))

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
                              stalled_rate=stalled_rate,
                              n_backend_err=n_backend_err))
    return out


@dataclass
class StepStats:
    """Error composition resolved by STEP INDEX rather than by task depth.

    H4 is a claim about position in the chain: the error mix shifts from
    selection-dominated at the first step to argument-dominated deeper in. That
    trend cannot be read off a per-task-depth aggregation, because a depth-8
    task contributes steps 0..7 to a single bin and any within-chain trend is
    averaged away before it can be seen. Measured on a policy configured with a
    syntactic share rising 0.20 -> 0.80 across steps, the per-depth estimator
    recovered 0.42 -> 0.54: the direction survived, the magnitude did not.
    """
    step: int
    n: int                    # fresh-error trials at this step index
    f_syn: float              # syntactic share of fresh errors
    n_fresh_errors: int
    sel_err_share: float      # share of fresh errors that are selection errors
    arg_err_share: float      # share that are argument errors
    sel_to_arg: float         # ratio; the H4 quantity. NaN if no argument errors
    p_step: float             # teacher-forced correctness at this step index
    g_step: float             # free-run correctness at this step index


def aggregate_by_step(records: list[dict], max_step: int | None = None
                      ) -> list[StepStats]:
    """Per-step-index error composition, pooled across task depths.

    Restricted to steps entered on a clean, non-stalled context, for the same
    reason f_syn is: a step that is wrong only because its input was already
    corrupted upstream is not fresh evidence about which error type this model
    produces, and counting it mixes propagation into the composition.
    """
    fr = [r for r in records
          if r["run_mode"] == "free" and not r.get("backend_error", False)]
    tf = [r for r in records
          if r["run_mode"] == "teacher_forced" and not r.get("backend_error", False)]
    if not fr:
        return []
    if max_step is None:
        max_step = max(r["step"] for r in fr)

    out = []
    for s in range(max_step + 1):
        fr_s = [r for r in fr if r["step"] == s]
        tf_s = [r for r in tf if r["step"] == s]
        fresh = [r for r in fr_s
                 if r.get("context_clean_in", True)
                 and not r.get("stalled_in", False)]
        fresh_err = [r for r in fresh if not r["args_correct_strict"]]

        n_syn = sum(1 for r in fresh_err if r["error_type"] == "syntactic")
        # A selection error names the wrong tool; an argument error names the
        # right tool with a wrong value. Both are semantic (they execute).
        sel_err = sum(1 for r in fresh_err
                      if r["error_type"] != "syntactic"
                      and not r.get("selection_matches_gold", True))
        arg_err = sum(1 for r in fresh_err
                      if r["error_type"] != "syntactic"
                      and r.get("selection_matches_gold", True))
        ne = len(fresh_err)
        out.append(StepStats(
            step=s, n=len(fresh), n_fresh_errors=ne,
            f_syn=(n_syn / ne) if ne else float("nan"),
            sel_err_share=(sel_err / ne) if ne else float("nan"),
            arg_err_share=(arg_err / ne) if ne else float("nan"),
            sel_to_arg=(sel_err / arg_err) if arg_err else float("nan"),
            p_step=_safe_mean([r["args_correct_strict"] for r in tf_s]),
            g_step=_safe_mean([r["args_correct_strict"] for r in fr_s])))
    return out


def measure_recovery(records: list[dict]) -> dict:
    """Measure r_syn and r_sem directly from labelled transitions.

    The methodology states that both recovery rates are measured from labelled
    transitions rather than inferred from the shape of a curve, and that the
    measured values enter the fit as informative priors. Nothing computed them
    until now, so that claim was unsupported in print -- this closes it.

    Two distinct notions are reported, because the methodology uses both:

      chain-level : the rate at which a poisoned context returns to a correct,
                    on-track chain on the following step, split by the ORIGIN of
                    the corruption. This matches the state model's r_syn/r_sem,
                    which govern how poisoned mass returns to the clean state,
                    and is what feeds the priors.
      retry-level : among steps where a syntactic failure triggered an in-step
                    retry, the share that ended correct. This is a different
                    quantity (within-step, syntactic only) and is reported
                    separately rather than conflated.

    A step is "on track" when the value it carried in equals the value the gold
    trajectory expects, which is exactly context_clean_in.
    """
    by_task: dict[str, list[dict]] = {}
    for r in records:
        if r["run_mode"] != "free" or r.get("backend_error", False):
            continue
        by_task.setdefault(r["task_id"], []).append(r)

    trials = {"syntax": 0, "semantic": 0}
    successes = {"syntax": 0, "semantic": 0}
    for rows in by_task.values():
        rows.sort(key=lambda r: r["step"])
        origin = None
        for i, r in enumerate(rows):
            clean_in = r.get("context_clean_in", True)
            if not clean_in and origin is not None and i + 1 < len(rows):
                # currently poisoned with a known origin: does the NEXT step
                # enter on a clean, on-track context?
                trials[origin] += 1
                if rows[i + 1].get("context_clean_in", True):
                    successes[origin] += 1
            if clean_in and not r["args_correct_strict"]:
                # a fresh error here sets the origin carried forward
                origin = ("syntax" if r["error_type"] == "syntactic"
                          else "semantic")
            elif clean_in:
                origin = None

    retried = [r for rows in by_task.values() for r in rows
               if r.get("n_attempts", 1) > 1]
    return {
        "r_syn_chain": (successes["syntax"] / trials["syntax"]
                        if trials["syntax"] else float("nan")),
        "r_sem_chain": (successes["semantic"] / trials["semantic"]
                        if trials["semantic"] else float("nan")),
        "n_syn_trials": trials["syntax"], "n_sem_trials": trials["semantic"],
        "r_syn_retry": (sum(1 for r in retried if r.get("recovered", False))
                        / len(retried)) if retried else float("nan"),
        "n_retry_trials": len(retried),
    }


def bootstrap_L_ci(records: list[dict], depth: int, n_boot: int = 2000,
                   alpha: float = 0.11, seed: int = 0,
                   paired: bool = True) -> tuple[float, float, float]:
    """Bootstrap a confidence interval for L_t = 1 - g_t/p_t at one depth.

    L_t is a ratio of two estimated rates, so its uncertainty is not simply the
    uncertainty of either one. Resampling tasks (not individual steps) preserves
    within-task correlation, which matters because steps in the same chain are
    not independent once propagation is in play.

    `paired` resamples ONE set of task ids and reads both arms at those same
    ids. Every task is run under both protocols, so the arms are paired by
    construction; resampling them independently throws that pairing away and
    widens the interval for no reason. This matters more at the reduced task
    counts a rate-limited real run can afford. Falls back to independent
    resampling when the arms do not share ids.

    Returns (L_point, lo, hi) at the (1-alpha) level; defaults to an 89% CI.
    """
    rng = np.random.default_rng(seed)
    tf = [r for r in records if r["run_mode"] == "teacher_forced"
         and r["depth"] == depth and not r.get("backend_error", False)]
    fr = [r for r in records if r["run_mode"] == "free"
         and r["depth"] == depth and not r.get("backend_error", False)]
    if not tf or not fr:
        return float("nan"), float("nan"), float("nan")

    def by_task_map(rows):
        d: dict[str, list[bool]] = {}
        for r in rows:
            d.setdefault(r["task_id"], []).append(r["args_correct_strict"])
        return d

    tf_map, fr_map = by_task_map(tf), by_task_map(fr)
    if not tf_map or not fr_map:
        return float("nan"), float("nan"), float("nan")

    def rate_from(groups):
        vals = [v for g in groups for v in g]
        return float(np.mean(vals)) if vals else float("nan")

    p_point = rate_from(list(tf_map.values()))
    g_point = rate_from(list(fr_map.values()))
    L_point = 1 - g_point / p_point if p_point else float("nan")

    shared = sorted(set(tf_map) & set(fr_map))
    draws = []
    if paired and len(shared) >= 2:
        for _ in range(n_boot):
            idx = rng.integers(0, len(shared), len(shared))
            p_b = rate_from([tf_map[shared[i]] for i in idx])
            g_b = rate_from([fr_map[shared[i]] for i in idx])
            if p_b and not np.isnan(p_b) and not np.isnan(g_b):
                draws.append(1 - g_b / p_b)
    else:
        tf_tasks, fr_tasks = list(tf_map.values()), list(fr_map.values())
        for _ in range(n_boot):
            ti = rng.integers(0, len(tf_tasks), len(tf_tasks))
            fi = rng.integers(0, len(fr_tasks), len(fr_tasks))
            p_b = rate_from([tf_tasks[i] for i in ti])
            g_b = rate_from([fr_tasks[i] for i in fi])
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
