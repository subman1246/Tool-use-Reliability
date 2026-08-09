"""Bayesian hierarchical estimation of pi, r_syn, r_sem per model.

Point fits go singular when a weak model flatlines to its floor early, so the
depth grid cannot separate high severity from low recovery. Partial pooling
across families and scales plus weakly informative priors regularises those
cases, and the fit-free L_t remains the fallback when a parameter is not
identified for a given model.

Data (all arrays shaped [n_models, n_depths]):
  p         measured clean baseline p_t (context-length degradation lives here)
  f_syn     measured syntactic share of errors at each depth
  successes global-correct counts
  trials    task counts
  group     [n_models] integer family id for pooling
"""

from __future__ import annotations

import numpy as np


def build_and_sample(p, f_syn, successes, trials, group,
                     draws=800, tune=800, chains=2, seed=0, target_accept=0.9,
                     prior_r_syn=None, prior_r_sem=None):
    """Fit the hierarchical propagation model.

    prior_r_syn / prior_r_sem: optional per-FAMILY recovery rates measured from
    labelled transitions, on the probability scale, used to centre the recovery
    hyperpriors. Omit for the generic weakly-informative priors.
    """
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt

    p = np.asarray(p, float); f_syn = np.asarray(f_syn, float)
    successes = np.asarray(successes, int); trials = np.asarray(trials, int)
    group = np.asarray(group, int)
    M, T = p.shape
    G = int(group.max()) + 1

    pT = p.T            # [T, M]
    fT = f_syn.T        # [T, M]

    def recur(p_t, f_t, c, s, m, r_syn, r_sem):
        err = 1.0 - p_t
        c1 = c * p_t + s * r_syn + m * r_sem
        s1 = c * err * f_t + s * (1.0 - r_syn)
        m1 = c * err * (1.0 - f_t) + m * (1.0 - r_sem)
        return c1, s1, m1

    # Optional informative centres for the recovery hyperpriors, taken from
    # recovery rates MEASURED off labelled transitions (aggregate.measure_
    # recovery). The methodology says the measured rates "enter as informative
    # priors, tightening the real-data fits"; until this existed they did not,
    # and the priors were generic. Passed on the probability scale, per family,
    # and converted to the logit scale here. A measured rate at 0 or 1 is
    # clipped rather than sent to +-inf.
    def _logit_centre(vals, fallback):
        if vals is None:
            return fallback
        arr = np.asarray(vals, float)
        if arr.shape != (G,) or not np.all(np.isfinite(arr)):
            return fallback
        arr = np.clip(arr, 0.02, 0.98)
        return np.log(arr / (1.0 - arr))

    mu_rs_centre = _logit_centre(prior_r_syn, 0.0)
    mu_rm_centre = _logit_centre(prior_r_sem, -1.0)
    informed = (prior_r_syn is not None) or (prior_r_sem is not None)
    # Tighter when informed by measurement, still wide enough for the data to
    # dominate where the data are plentiful.
    prior_sd = 1.0 if informed else 1.5

    with pm.Model() as model:
        # hyperpriors per family, on the logit scale
        mu_pi = pm.Normal("mu_pi", 0.0, 1.5, shape=G)
        mu_rs = pm.Normal("mu_rs", mu_rs_centre, prior_sd, shape=G)
        mu_rm = pm.Normal("mu_rm", mu_rm_centre, prior_sd, shape=G)
        sd_pi = pm.HalfNormal("sd_pi", 1.0)
        sd_rs = pm.HalfNormal("sd_rs", 1.0)
        sd_rm = pm.HalfNormal("sd_rm", 1.0)

        z_pi = pm.Normal("z_pi", 0.0, 1.0, shape=M)
        z_rs = pm.Normal("z_rs", 0.0, 1.0, shape=M)
        z_rm = pm.Normal("z_rm", 0.0, 1.0, shape=M)

        pi = pm.Deterministic("pi", pm.math.sigmoid(mu_pi[group] + sd_pi * z_pi))
        r_syn = pm.Deterministic("r_syn", pm.math.sigmoid(mu_rs[group] + sd_rs * z_rs))
        r_sem = pm.Deterministic("r_sem", pm.math.sigmoid(mu_rm[group] + sd_rm * z_rm))

        # unroll the recurrence over the small, fixed depth grid (avoids scan)
        c = pt.ones((M,)); s = pt.zeros((M,)); m = pt.zeros((M,))
        x_rows = [s + m]
        for t in range(T - 1):
            p_t = pt.as_tensor_variable(pT[t])   # [M]
            f_t = pt.as_tensor_variable(fT[t])
            c, s, m = recur(p_t, f_t, c, s, m, r_syn, r_sem)
            x_rows.append(s + m)
        x = pt.stack(x_rows, axis=0)             # [T, M]
        g = pt.as_tensor_variable(pT) * (1.0 - pi[None, :] * x)
        g = pt.clip(g, 1e-6, 1 - 1e-6)

        pm.Binomial("obs", n=trials.T, p=g, observed=successes.T)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                          random_seed=seed, target_accept=target_accept,
                          progressbar=False)
    return idata


def identifiability(idata, model_index: int) -> dict:
    """Posterior correlation of pi with each recovery rate for one model.

    Values near +/-1 mean the parameters are not separately identified and the
    fit-free L_t should be reported instead of a point estimate.
    """
    import numpy as np
    pi = idata.posterior["pi"].values[..., model_index].reshape(-1)
    rs = idata.posterior["r_syn"].values[..., model_index].reshape(-1)
    rm = idata.posterior["r_sem"].values[..., model_index].reshape(-1)
    return {
        "corr_pi_rsyn": float(np.corrcoef(pi, rs)[0, 1]),
        "corr_pi_rsem": float(np.corrcoef(pi, rm)[0, 1]),
    }
