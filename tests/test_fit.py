"""Simulate data with known parameters and check the fit recovers them.

Two families, three scales each. Poisoning severity pi falls with scale (H2),
syntactic recovery is moderate, semantic recovery is low (H5b). We simulate the
global-correctness counts from the state model, fit the hierarchical model, and
confirm the posterior means land near the truth with healthy diagnostics.
"""

import numpy as np
import arviz as az

from tur.model.state import simulate
from tur.model.hierarchical import build_and_sample, identifiability


def main():
    rng = np.random.default_rng(3)
    T = 5
    depths = np.array([1, 2, 4, 6, 8])
    # baseline degrades with depth (context length); slightly lower for smaller models
    scales = [0.0, 1.0, 2.0]  # small, mid, large within a family
    families = [0, 1]

    p_rows, f_rows, succ_rows, tri_rows, group = [], [], [], [], []
    truth = []
    for fam in families:
        for sc in scales:
            base = 0.90 + 0.03 * sc - 0.01 * fam
            p = np.clip(base - 0.03 * np.arange(T), 0.4, 0.99)
            f_syn = np.full(T, 0.5)
            pi = float(np.clip(0.75 - 0.22 * sc, 0.05, 0.95))   # falls with scale
            r_syn = 0.65
            r_sem = 0.10
            succ, trials, _ = simulate(p, f_syn, pi, r_syn, r_sem, 200, rng)
            p_rows.append(p); f_rows.append(f_syn)
            succ_rows.append(succ); tri_rows.append(trials); group.append(fam)
            truth.append((pi, r_syn, r_sem))

    p = np.array(p_rows); f_syn = np.array(f_rows)
    successes = np.array(succ_rows); trials = np.array(tri_rows)
    group = np.array(group)

    idata = build_and_sample(p, f_syn, successes, trials, group,
                             draws=800, tune=800, chains=2, seed=1)

    summ = az.summary(idata, var_names=["pi", "r_syn", "r_sem"])
    print(summ.to_string())

    pi_hat = idata.posterior["pi"].mean(("chain", "draw")).values
    rhat_ds = az.rhat(idata, var_names=["pi", "r_syn", "r_sem"])
    max_rhat = float(max(float(rhat_ds[v].max()) for v in rhat_ds.data_vars))
    print("\nmax R-hat:", round(max_rhat, 3))

    print("\npi truth vs posterior mean:")
    ok = True
    for i, (pi_true, _, _) in enumerate(truth):
        err = abs(pi_hat[i] - pi_true)
        flag = "" if err < 0.15 else "  <-- off"
        if err >= 0.15:
            ok = False
        print(f"  model {i} (fam {group[i]}): truth {pi_true:.2f}  "
              f"post {pi_hat[i]:.2f}  |err| {err:.2f}{flag}")

    print("\nidentifiability (model 0):", identifiability(idata, 0))
    print("\nrecovers pi within 0.15:", ok, " | R-hat ok:", max_rhat < 1.1)


if __name__ == "__main__":
    main()
