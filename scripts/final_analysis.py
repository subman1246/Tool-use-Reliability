"""Final analysis over FROZEN data. Makes no API calls, ever.

`run_real_suite.py` interleaves collection with analysis, so re-running it to refresh
the fit would issue new requests for any task not yet collected. After the data freeze
that is exactly what must not happen: the dataset is fixed, and the analysis has to be a
pure function of the files on disk. This script reads `<tag>_groq_*.jsonl` and nothing
else -- no backend is constructed, so no call can be made even accidentally.

Writes the same `<tag>_meta.json` and `<tag>_idata.pkl` artifacts the runner would, so
`report_real.py` and `tur.analysis.plots` work unchanged.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np

from tur.analysis.aggregate import (aggregate_by_depth, aggregate_by_step,
                                    bootstrap_L_ci, load_records,
                                    measure_recovery, stats_to_arrays)
from tur.model.hierarchical import build_and_sample

OUT_DIR = "data/results"


def delta_1(records: list[dict]) -> dict:
    by_task: dict[str, dict[int, bool]] = {}
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
    return {"delta_1": a - b, "p_next_given_correct": a, "p_next_given_wrong": b,
            "n_given_correct": len(gc), "n_given_wrong": len(gw)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="real")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    ap.add_argument("--min-tasks", type=int, default=10,
                    help="models with fewer completed tasks than this are excluded "
                         "from the fit and recorded as data-insufficient")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    depths = args.depths

    import yaml
    cfg = yaml.safe_load(open(args.config))
    cfg_by_name = {m["name"]: m for m in cfg.get("models", [])}

    files = sorted(glob.glob(f"{OUT_DIR}/{args.tag}_groq_*.jsonl"))
    print(f"frozen dataset: {len(files)} model file(s) under tag '{args.tag}'")

    names, group, fam_names = [], [], []
    p_rows, f_rows, succ_rows, tri_rows = [], [], [], []
    L_ci, d1, recov, per_step, extra, filled_rec = {}, {}, {}, {}, {}, {}
    achieved, insufficient = {}, {}

    for f in files:
        stem = os.path.basename(f).replace(f"{args.tag}_", "").replace(".jsonl", "")
        # recover the provider-qualified name from the flattened filename
        name = next((n for n in cfg_by_name if n.replace("/", "_") == stem), stem)
        recs = load_records(f)
        seen: dict[int, set] = {}
        for r in recs:
            seen.setdefault(r["depth"], set()).add(r["task_id"])
        got = {d: len(v) for d, v in sorted(seen.items())}
        achieved[name] = got
        n_tasks = sum(got.values())
        if n_tasks < args.min_tasks:
            insufficient[name] = got
            print(f"  EXCLUDED (data-insufficient): {name} -- {n_tasks} tasks {got}")
            continue

        stats = aggregate_by_depth(recs, depths)
        p, f_syn, fl = stats_to_arrays(stats)
        if fl["p"].any():
            filled_rec[name] = {"p": [depths[i] for i, x in enumerate(fl["p"]) if x],
                                "f_syn": [depths[i] for i, x in enumerate(fl["f_syn"]) if x]}
        p_rows.append(p); f_rows.append(f_syn)
        succ_rows.append(np.array([round(s.g_t * s.n_g) if s.n_g else 0 for s in stats]))
        tri_rows.append(np.array([s.n_g for s in stats]))
        fam = cfg_by_name.get(name, {}).get("family", name)
        if fam not in fam_names:
            fam_names.append(fam)
        group.append(fam_names.index(fam))
        names.append(name)

        d1[name] = delta_1(recs)
        recov[name] = measure_recovery(recs)
        per_step[name] = [st.__dict__ for st in aggregate_by_step(recs)]
        L_ci[name] = {str(d): dict(zip(("L", "lo", "hi"),
                                       bootstrap_L_ci(recs, d, 2000, seed=13)))
                      for d in depths}
        extra[name] = [{"depth": s.depth, "p_t": s.p_t, "g_t": s.g_t, "L_t": s.L_t,
                        "n_p": s.n_p, "n_g": s.n_g, "f_syn": s.f_syn,
                        "n_fresh_errors": s.n_fresh_errors,
                        "selection_acc": s.selection_acc,
                        "selection_gold_acc": s.selection_gold_acc,
                        "parse_fail_rate": s.parse_fail_rate,
                        "stalled_rate": s.stalled_rate,
                        "n_backend_err": s.n_backend_err} for s in stats]
        print(f"  included: {name} -- {n_tasks} tasks {got}")

    if not names:
        raise SystemExit("no model has enough data to analyse")

    grp = np.array(group)
    G = int(grp.max()) + 1

    def fam_mean(key):
        out = []
        for g in range(G):
            vals = [recov[n][key] for i, n in enumerate(names)
                    if grp[i] == g and not np.isnan(recov[n][key])]
            out.append(float(np.mean(vals)) if vals else float("nan"))
        return out

    prior_rs, prior_rm = fam_mean("r_syn_chain"), fam_mean("r_sem_chain")
    print(f"\nmeasured recovery -> prior centres: r_syn={prior_rs} r_sem={prior_rm}")
    print("fitting hierarchical model on frozen data ...")
    idata = build_and_sample(np.array(p_rows), np.array(f_rows),
                             np.array(succ_rows), np.array(tri_rows), grp,
                             draws=1500, tune=1500, chains=4, seed=7,
                             target_accept=0.97,
                             prior_r_syn=prior_rs, prior_r_sem=prior_rm)

    with open(f"{OUT_DIR}/{args.tag}_idata.pkl", "wb") as fh:
        pickle.dump(idata, fh)
    with open(f"{OUT_DIR}/{args.tag}_meta.json", "w") as fh:
        json.dump({"names": names, "group": grp.tolist(), "depths": depths,
                   "p": np.array(p_rows).tolist(),
                   "f_syn": np.array(f_rows).tolist(),
                   "successes": np.array(succ_rows).tolist(),
                   "trials": np.array(tri_rows).tolist(),
                   "delta_1": d1, "L_ci": L_ci, "per_depth_extra": extra,
                   "measured_recovery": recov, "per_step": per_step,
                   "priors_used": {"r_syn": prior_rs, "r_sem": prior_rm},
                   "substituted_input_depths": filled_rec,
                   "models_config": [cfg_by_name.get(n, {"name": n}) for n in names],
                   "task_variant": cfg.get("task_variant", "routing"),
                   "arg_shift": int(cfg.get("arg_shift", 0) or 0),
                   "achieved_n": achieved,
                   "data_insufficient": insufficient,
                   "frozen": True}, fh, indent=2)
    print(f"\nsaved -> {OUT_DIR}/{args.tag}_meta.json, {args.tag}_idata.pkl")


if __name__ == "__main__":
    main()
