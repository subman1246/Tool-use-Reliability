"""Validate the estimator on ROUTING tasks against known ground truth.

Why this exists: the official validation run (run_full_analysis.py) used the
LINEAR task variant only, but the real experiment's primary suite is routing --
linear tasks turned out to carry no propagation signal on real models. So the
estimator has never been checked against known truth on the task family that
carries every headline claim. This closes that gap.

Two questions:

  1. Does L_t recover the configured severity ordering on routing, as it did on
     linear? Routing differs structurally: a wrong value sends the agent down a
     wrong BRANCH, so selection errors can propagate, which linear cannot
     exercise.

  2. Does the error-type decomposition detect a depth-varying error mix (H4)?
     The linear suite could not test H4 because every policy was configured with
     a FLAT syntax share, so there was nothing for the decomposition to find.
     Two policies here vary syntax_share with depth in opposite directions, so a
     detector that works must recover both the rising and the falling trend --
     and must not invent one for the flat controls.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np

from tur.tasks.dag import generate_routing_suite
from tur.harness.executor import FeedbackMode
from tur.harness.runner import run_free, run_teacher_forced, MockBackend
from tur.harness.sim_policy import SimPolicy, SimPolicyConfig
from tur.analysis.aggregate import (load_records, aggregate_by_depth,
                                    stats_to_arrays, bootstrap_L_ci)
from tur.model.hierarchical import build_and_sample, identifiability

DEPTHS = [1, 2, 4, 6, 8]
PER_DEPTH = 200
SEEDS = 2
OUT_DIR = "data/results"
RUN_TAG = "routingval"


class DepthVaryingSyntaxPolicy(SimPolicy):
    """SimPolicy whose syntactic share is a function of depth.

    H4 predicts the error mix shifts with depth (selection-dominated early,
    argument-dominated later). Every policy in the default suite holds
    syntax_share constant, so the H4 test in the linear validation run was
    vacuous by construction -- it could only ever confirm "flat in, flat out".
    Here syntax_share moves linearly from `share_at_1` at step 0 to `share_at_8`
    at step 7, giving the decomposition a known non-flat trend to recover.
    """

    def __init__(self, cfg: SimPolicyConfig, share_at_1: float, share_at_8: float):
        super().__init__(cfg)
        self.share_at_1 = share_at_1
        self.share_at_8 = share_at_8

    def syntax_share_at(self, step: int) -> float:
        frac = min(1.0, step / 7.0)
        return self.share_at_1 + (self.share_at_8 - self.share_at_1) * frac

    def __call__(self, task, step: int, ref: int, attempt: int):
        # Mutate the configured share for this step, then defer to the parent so
        # every other mechanism (recovery, poisoning, retry) stays identical.
        self.cfg.syntax_share = self.syntax_share_at(step)
        return super().__call__(task, step, ref, attempt)


# Two flat controls spanning weak/strong, plus two depth-varying policies with
# opposite trends. Flat controls must show no trend; the others must show theirs.
SUITE = [
    ("flat-weak", SimPolicyConfig("flat-weak", p0=0.78, p_slope=0.020, pi=0.70,
                                  syntax_share=0.50, r_syn=0.35, r_sem=0.05,
                                  seed=11), None),
    ("flat-strong", SimPolicyConfig("flat-strong", p0=0.95, p_slope=0.006, pi=0.22,
                                    syntax_share=0.50, r_syn=0.75, r_sem=0.30,
                                    seed=12), None),
    ("rising-syn", SimPolicyConfig("rising-syn", p0=0.88, p_slope=0.012, pi=0.45,
                                   syntax_share=0.50, r_syn=0.55, r_sem=0.15,
                                   seed=13), (0.20, 0.80)),
    ("falling-syn", SimPolicyConfig("falling-syn", p0=0.86, p_slope=0.013, pi=0.40,
                                    syntax_share=0.50, r_syn=0.50, r_sem=0.12,
                                    seed=14), (0.80, 0.20)),
]
FAMILY = {"flat-weak": 0, "flat-strong": 0, "rising-syn": 1, "falling-syn": 1}


def run_policy(cfg: SimPolicyConfig, trend) -> list[dict]:
    records = []
    for seed in range(SEEDS):
        c = SimPolicyConfig(**{**cfg.__dict__, "seed": cfg.seed * 97 + seed})
        policy = (DepthVaryingSyntaxPolicy(c, *trend) if trend else SimPolicy(c))
        backend = MockBackend(policy, model=cfg.name)
        suite = generate_routing_suite(DEPTHS, PER_DEPTH, distractor_level=1,
                                       base_seed=seed * 31 + 1000)
        for task in suite:
            # see run_full_analysis for why the run mode must be signalled
            policy.stateless = False
            f = run_free(task, backend, "uniform", FeedbackMode.STRUCTURED, 1)
            policy.stateless = True
            t = run_teacher_forced(task, backend, "uniform",
                                   FeedbackMode.STRUCTURED, 1)
            policy.stateless = False
            records += [r.__dict__ for r in f] + [r.__dict__ for r in t]
    return records


def delta_1(records):
    by_task = {}
    for r in records:
        if r["run_mode"] != "free" or r.get("backend_error", False):
            continue
        by_task.setdefault(r["task_id"], {})[r["step"]] = r["args_correct_strict"]
    gc, gw = [], []
    for steps in by_task.values():
        for t in sorted(steps):
            if (t + 1) in steps:
                (gc if steps[t] else gw).append(steps[t + 1])
    a = float(np.mean(gc)) if gc else float("nan")
    b = float(np.mean(gw)) if gw else float("nan")
    return {"delta_1": a - b, "p_next_given_correct": a,
            "p_next_given_wrong": b, "n_given_correct": len(gc),
            "n_given_wrong": len(gw)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    p_rows, f_rows, s_rows, t_rows, group, names = [], [], [], [], [], []
    L_ci, d1, per_depth, truth, filled_note = {}, {}, {}, {}, {}

    for name, cfg, trend in SUITE:
        print(f"running {name} (routing{', depth-varying syntax' if trend else ''}) ...")
        records = run_policy(cfg, trend)
        path = f"{OUT_DIR}/{RUN_TAG}_{name}.jsonl"
        with open(path, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

        loaded = load_records(path)
        stats = aggregate_by_depth(loaded, DEPTHS)
        p, f_syn, fl = stats_to_arrays(stats)
        if any(fl):
            filled_note[name] = [DEPTHS[i] for i, x in enumerate(fl) if x]
            print(f"  note: substituted inputs at depth(s) {filled_note[name]}")

        p_rows.append(p); f_rows.append(f_syn)
        s_rows.append(np.array([round(s.g_t * s.n_g) if s.n_g else 0 for s in stats]))
        t_rows.append(np.array([s.n_g for s in stats]))
        group.append(FAMILY[name]); names.append(name)

        d1[name] = delta_1(loaded)
        L_ci[name] = {str(d): dict(zip(("L", "lo", "hi"),
                                       bootstrap_L_ci(loaded, d, 1000, seed=13)))
                      for d in DEPTHS}
        per_depth[name] = [{"depth": s.depth, "p_t": s.p_t, "g_t": s.g_t,
                            "L_t": s.L_t, "f_syn": s.f_syn,
                            "n_fresh_errors": s.n_fresh_errors,
                            "selection_acc": s.selection_acc,
                            "selection_gold_acc": s.selection_gold_acc,
                            "parse_fail_rate": s.parse_fail_rate,
                            "stalled_rate": s.stalled_rate} for s in stats]
        truth[name] = {**cfg.__dict__,
                       "syntax_trend": ({"at_depth1": trend[0], "at_depth8": trend[1]}
                                        if trend else "flat")}

    print("\nfitting hierarchical model ...")
    idata = build_and_sample(np.array(p_rows), np.array(f_rows),
                             np.array(s_rows), np.array(t_rows),
                             np.array(group), draws=1500, tune=1500, chains=4,
                             seed=7, target_accept=0.97)

    ident = {n: identifiability(idata, i) for i, n in enumerate(names)}
    with open(f"{OUT_DIR}/{RUN_TAG}_idata.pkl", "wb") as fh:
        pickle.dump(idata, fh)
    with open(f"{OUT_DIR}/{RUN_TAG}_meta.json", "w") as fh:
        json.dump({"names": names, "group": [int(g) for g in group],
                   "depths": DEPTHS, "task_variant": "routing",
                   "p": np.array(p_rows).tolist(),
                   "f_syn": np.array(f_rows).tolist(),
                   "successes": np.array(s_rows).tolist(),
                   "trials": np.array(t_rows).tolist(),
                   "delta_1": d1, "L_ci": L_ci, "per_depth_extra": per_depth,
                   "truth": truth, "identifiability": ident,
                   "substituted_input_depths": filled_note,
                   "config": [c.__dict__ for _, c, _ in SUITE]}, fh, indent=2)
    print(f"\nsaved -> {OUT_DIR}/{RUN_TAG}_idata.pkl, {RUN_TAG}_meta.json")


if __name__ == "__main__":
    main()
