"""Posterior predictive check for the hierarchical propagation model.

Makes no API calls: it reuses an already-fit trace and the frozen task records.

Why this exists. The paper asserts that the fitted severity/recovery model is
*misspecified* on real data -- that clean MCMC diagnostics show only that a
misspecified posterior was explored successfully -- but until now it showed no check
that could actually fail. This is that check.

The quantity checked is the one on which the model and the data disagree most sharply.
Under the model, a call made on a poisoned context succeeds with probability
(1 - pi) * p_t, which is strictly positive for any pi < 1. Summed over every poisoned
call in the dataset, the model therefore predicts some number of gold matches among
corrupted-context calls. The observed number is zero. If the posterior predictive
distribution of that count rarely or never reaches zero, the model is making a
prediction the data falsifies, and "misspecified" stops being an assertion.

Two nodes are checked:

  obs               the standard check -- replicated free-arm success counts against
                    the observed ones. A model can pass this while failing the targeted
                    check, because the marginal rate g_t averages over clean and
                    poisoned calls and the clean ones dominate.
  poisoned_matches  the targeted check -- replicated gold matches among poisoned calls
                    only.

Run:  py scripts/posterior_predictive_check.py --tag real
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

RES = "data/results"


def observed_poisoned_cells(tag, names, depths):
    """n_poisoned and gold-match counts per (model, depth), from frozen records."""
    n_pois = np.zeros((len(names), len(depths)), dtype=int)
    matches = np.zeros((len(names), len(depths)), dtype=int)
    for i, name in enumerate(names):
        path = os.path.join(RES, tag + "_" + name.replace("/", "_") + ".jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        free = [r for r in rows
                if r["run_mode"] == "free" and not r.get("backend_error", False)]
        for j, d in enumerate(depths):
            po = [r for r in free
                  if r["depth"] == d and not r.get("context_clean_in", True)]
            n_pois[i, j] = len(po)
            matches[i, j] = sum(1 for r in po if r["args_correct_strict"])
    return n_pois, matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="real")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = args.tag

    with open(os.path.join(RES, tag + "_meta.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(RES, tag + "_idata.pkl"), "rb") as fh:
        idata = pickle.load(fh)

    names, depths = meta["names"], meta["depths"]
    p = np.array(meta["p"], float)
    f_syn = np.array(meta["f_syn"], float)
    successes = np.array(meta["successes"], int)
    trials = np.array(meta["trials"], int)
    group = np.array(meta["group"], int)
    M, T = p.shape
    G = int(group.max()) + 1

    n_pois, obs_matches = observed_poisoned_cells(tag, names, depths)
    print("tag=" + tag + "  models=" + str(M) + "  depths=" + str(depths))
    print("observed poisoned calls: " + str(int(n_pois.sum()))
          + ", gold matches among them: " + str(int(obs_matches.sum())))

    import pymc as pm
    import pytensor.tensor as pt

    pT, fT = p.T, f_syn.T
    priors = meta.get("priors_used") or {}

    def logit_centre(vals, fallback):
        if vals is None:
            return fallback
        arr = np.asarray(vals, float)
        if arr.shape != (G,) or not np.all(np.isfinite(arr)):
            return fallback
        arr = np.clip(arr, 0.02, 0.98)
        return np.log(arr / (1.0 - arr))

    mu_rs_c = logit_centre(priors.get("r_syn"), 0.0)
    mu_rm_c = logit_centre(priors.get("r_sem"), -1.0)
    prior_sd = 1.0 if priors else 1.5

    # Rebuilt to match hierarchical.build_and_sample exactly, so the posterior draws
    # already in the trace are valid for these variables. The only addition is the
    # second observed node used for the targeted check.
    with pm.Model():
        mu_pi = pm.Normal("mu_pi", 0.0, 1.5, shape=G)
        mu_rs = pm.Normal("mu_rs", mu_rs_c, prior_sd, shape=G)
        mu_rm = pm.Normal("mu_rm", mu_rm_c, prior_sd, shape=G)
        sd_pi = pm.HalfNormal("sd_pi", 1.0)
        sd_rs = pm.HalfNormal("sd_rs", 1.0)
        sd_rm = pm.HalfNormal("sd_rm", 1.0)
        z_pi = pm.Normal("z_pi", 0.0, 1.0, shape=M)
        z_rs = pm.Normal("z_rs", 0.0, 1.0, shape=M)
        z_rm = pm.Normal("z_rm", 0.0, 1.0, shape=M)

        pi = pm.Deterministic("pi", pm.math.sigmoid(mu_pi[group] + sd_pi * z_pi))
        r_syn = pm.Deterministic("r_syn", pm.math.sigmoid(mu_rs[group] + sd_rs * z_rs))
        r_sem = pm.Deterministic("r_sem", pm.math.sigmoid(mu_rm[group] + sd_rm * z_rm))

        c = pt.ones((M,))
        s = pt.zeros((M,))
        m = pt.zeros((M,))
        x_rows = [s + m]
        for t in range(T - 1):
            p_t = pt.as_tensor_variable(pT[t])
            f_t = pt.as_tensor_variable(fT[t])
            err = 1.0 - p_t
            c, s, m = (c * p_t + s * r_syn + m * r_sem,
                       c * err * f_t + s * (1.0 - r_syn),
                       c * err * (1.0 - f_t) + m * (1.0 - r_sem))
            x_rows.append(s + m)
        x = pt.stack(x_rows, axis=0)

        g = pt.as_tensor_variable(pT) * (1.0 - pi[None, :] * x)
        g = pt.clip(g, 1e-6, 1 - 1e-6)
        pm.Binomial("obs", n=trials.T, p=g, observed=successes.T)

        # A call on a POISONED context succeeds at (1 - pi) * p_t under this model.
        g_pois = pt.clip((1.0 - pi[None, :]) * pt.as_tensor_variable(pT),
                         1e-12, 1 - 1e-6)
        pm.Binomial("poisoned_matches", n=n_pois.T, p=g_pois, observed=obs_matches.T)

        ppc = pm.sample_posterior_predictive(
            idata, var_names=["obs", "poisoned_matches"],
            random_seed=11, progressbar=False)

    rep_pois = ppc.posterior_predictive["poisoned_matches"].values
    rep_tot = rep_pois.reshape(-1, T * M).sum(axis=1)
    n_rep = int(rep_tot.size)
    n_zero = int((rep_tot == 0).sum())
    frac_zero = float(n_zero) / n_rep

    rep_obs = ppc.posterior_predictive["obs"].values.reshape(-1, T * M)
    obs_tot = int(successes.T.sum())
    obs_rep_tot = rep_obs.sum(axis=1)
    p_obs = float((obs_rep_tot >= obs_tot).mean())

    print("")
    print("=== TARGETED CHECK: gold matches among poisoned-context calls ===")
    print("  observed                       : " + str(int(obs_matches.sum())))
    print("  posterior predictive replicates: " + str(n_rep))
    print("  replicate mean                 : %.1f" % rep_tot.mean())
    print("  replicate 5.5-94.5%% interval   : [%.0f, %.0f]"
          % (np.percentile(rep_tot, 5.5), np.percentile(rep_tot, 94.5)))
    print("  replicate minimum              : " + str(int(rep_tot.min())))
    print("  FRACTION OF REPLICATES == 0    : %.6f  (%d of %d)"
          % (frac_zero, n_zero, n_rep))
    print("")
    print("=== STANDARD CHECK: total free-arm successes ===")
    print("  observed %d, replicate mean %.0f, Bayesian p = %.3f"
          % (obs_tot, obs_rep_tot.mean(), p_obs))

    out = args.out or os.path.join(RES, tag + "_ppc.json")
    payload = {
        "tag": tag,
        "n_replicates": n_rep,
        "observed_poisoned_matches": int(obs_matches.sum()),
        "observed_poisoned_calls": int(n_pois.sum()),
        "replicate_mean": float(rep_tot.mean()),
        "replicate_min": int(rep_tot.min()),
        "replicate_max": int(rep_tot.max()),
        "replicate_q055": float(np.percentile(rep_tot, 5.5)),
        "replicate_q945": float(np.percentile(rep_tot, 94.5)),
        "fraction_zero": frac_zero,
        "n_zero": n_zero,
        "obs_total_successes": obs_tot,
        "obs_ppc_bayes_p": p_obs,
        "replicates": rep_tot.astype(int).tolist(),
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("")
    print("saved -> " + out)


if __name__ == "__main__":
    main()
