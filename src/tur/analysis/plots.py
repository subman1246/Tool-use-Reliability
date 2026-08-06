"""Generate every figure from the saved run artifacts (JSONL logs, meta.json,
and the pickled PyMC idata). Run after run_full_analysis.py. Nothing here is
hand-drawn; every number comes from the aggregation layer or the posterior.
"""

from __future__ import annotations

import json
import pickle

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "data/results"
FIG_DIR = f"{OUT_DIR}/figures"
RUN_TAG = "official"

COLORS = {"fam0-weak": "#c0392b", "fam0-mid": "#e67e22", "fam0-strong": "#27ae60",
         "fam1-weak": "#8e44ad", "fam1-mid": "#2980b9", "fam1-strong": "#16a085"}


def load_all():
    meta = json.load(open(f"{OUT_DIR}/{RUN_TAG}_meta.json"))
    idata = pickle.load(open(f"{OUT_DIR}/{RUN_TAG}_idata.pkl", "rb"))
    return meta, idata


def fig_p_vs_g(meta):
    """p_t (clean baseline) vs g_t (global rate) per model: the propagation gap."""
    depths = meta["depths"]
    names = meta["names"]
    p = np.array(meta["p"]); succ = np.array(meta["successes"]); tri = np.array(meta["trials"])
    g = np.divide(succ, tri, out=np.full_like(succ, np.nan, dtype=float), where=tri > 0)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, name, p_row, g_row in zip(axes.flat, names, p, g):
        ax.plot(depths, p_row, "o-", color="#333333", label="$p_t$ (clean baseline)")
        ax.plot(depths, g_row, "s-", color=COLORS.get(name, "#1155CC"), label="$g_t$ (free run)")
        ax.fill_between(depths, g_row, p_row, alpha=0.15, color=COLORS.get(name, "#1155CC"))
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("depth"); ax.set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("correct-invocation rate")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Clean baseline vs global correctness — the propagation gap")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig1_p_vs_g.png", dpi=150)
    plt.close(fig)


def fig_L_bars(meta):
    """Net propagation loss L_t = 1 - g_t/p_t, by depth, with bootstrap CIs."""
    depths = meta["depths"]; names = meta["names"]
    p = np.array(meta["p"]); succ = np.array(meta["successes"]); tri = np.array(meta["trials"])
    g = np.divide(succ, tri, out=np.full_like(succ, np.nan, dtype=float), where=tri > 0)
    L = 1 - g / p
    ci = meta.get("L_ci", {})

    x = np.arange(len(depths)); w = 0.13
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(names):
        yerr = None
        if name in ci:
            lo = np.array([ci[name][str(d)]["lo"] for d in depths])
            hi = np.array([ci[name][str(d)]["hi"] for d in depths])
            yerr = np.vstack([np.clip(L[i] - lo, 0, None),
                              np.clip(hi - L[i], 0, None)])
        ax.bar(x + (i - len(names) / 2) * w, L[i], width=w,
              label=name, color=COLORS.get(name), yerr=yerr,
              error_kw={"linewidth": 0.8, "ecolor": "#444444"}, capsize=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(depths)
    ax.set_xlabel("depth"); ax.set_ylabel("$L_t = 1 - g_t / p_t$")
    ax.set_title("Net propagation loss by depth (89% bootstrap CI)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig2_Lt_bars.png", dpi=150)
    plt.close(fig)


def fig_posterior_forest(idata, meta):
    """Forest plot of pi, r_syn, r_sem posteriors across models, built directly
    from posterior draws (arviz's plot_forest API differs across versions, so
    this avoids depending on it)."""
    names = meta["names"]
    for var, fname, title in [("pi", "fig3a_pi_forest", "Poisoning severity (pi)"),
                              ("r_syn", "fig3b_rsyn_forest", "Syntactic recovery (r_syn)"),
                              ("r_sem", "fig3c_rsem_forest", "Semantic recovery (r_sem)")]:
        draws = idata.posterior[var].values.reshape(-1, len(names))  # [samples, M]
        lo = np.percentile(draws, 5.5, axis=0)
        hi = np.percentile(draws, 94.5, axis=0)
        mean = draws.mean(axis=0)

        fig, ax = plt.subplots(figsize=(7, 4))
        y = np.arange(len(names))
        ax.errorbar(mean, y, xerr=[mean - lo, hi - mean], fmt="o",
                   color="#2c3e50", ecolor="#7f8c8d", capsize=3)
        ax.set_yticks(y); ax.set_yticklabels(names)
        ax.set_xlabel(var); ax.set_title(f"{title} — posterior mean, 89% CI")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/{fname}.png", dpi=150)
        plt.close(fig)


def fig_pi_vs_scale(idata, meta):
    """pi vs configured scale tier, one line per family: visual check of H2."""
    names = meta["names"]
    tier_order = {"weak": 0, "mid": 1, "strong": 2}
    pi_mean = idata.posterior["pi"].mean(("chain", "draw")).values
    pi_sd = idata.posterior["pi"].std(("chain", "draw")).values

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for fam, fam_color in [("fam0", "#c0392b"), ("fam1", "#2980b9")]:
        idx = [i for i, n in enumerate(names) if n.startswith(fam)]
        idx.sort(key=lambda i: tier_order[names[i].split("-")[1]])
        xs = [tier_order[names[i].split("-")[1]] for i in idx]
        ys = [pi_mean[i] for i in idx]
        es = [pi_sd[i] for i in idx]
        ax.errorbar(xs, ys, yerr=es, marker="o", label=fam, color=fam_color, capsize=3)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["weak", "mid", "strong"])
    ax.set_ylabel("posterior mean $\\pi$"); ax.set_title("Poisoning severity vs. scale tier (H2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig4_pi_vs_scale.png", dpi=150)
    plt.close(fig)


def fig_error_type_by_depth(meta):
    """Stacked share of syntactic vs semantic errors by depth, pooled and per model."""
    depths = meta["depths"]; names = meta["names"]
    f_syn = np.array(meta["f_syn"])  # [M, T], NaN-filled already in the saved meta? (raw here)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)
    for ax, name, row in zip(axes.flat, names, f_syn):
        row = np.array(row, dtype=float)
        syn = np.nan_to_num(row, nan=0.5)
        sem = 1 - syn
        ax.bar(range(len(depths)), syn, label="syntactic", color="#7f8c8d")
        ax.bar(range(len(depths)), sem, bottom=syn, label="semantic", color="#e74c3c")
        ax.set_xticks(range(len(depths))); ax.set_xticklabels(depths)
        ax.set_title(name, fontsize=10); ax.set_xlabel("depth")
    axes[0, 0].set_ylabel("share of errors")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Error-type composition by depth (H4)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig5_error_type_by_depth.png", dpi=150)
    plt.close(fig)


def fig_trace_diag(idata):
    """MCMC health: rank plots for the family-level hyperparameters, falling
    back to a manual trace plot if arviz's rank-plot API is incompatible with
    the installed version."""
    try:
        az.plot_rank(idata, var_names=["mu_pi", "mu_rs", "mu_rm"])
        plt.gcf().suptitle("MCMC rank plots (mixing diagnostic)")
        plt.gcf().tight_layout()
        plt.gcf().savefig(f"{FIG_DIR}/fig6_mcmc_rank.png", dpi=150)
        plt.close("all")
    except Exception as e:
        print(f"  (plot_rank unavailable: {e}; falling back to trace plot)")
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        for ax, var in zip(axes, ["mu_pi", "mu_rs", "mu_rm"]):
            arr = idata.posterior[var].values  # [chain, draw, group]
            for c in range(arr.shape[0]):
                for g in range(arr.shape[2]):
                    ax.plot(arr[c, :, g], alpha=0.6, linewidth=0.7)
            ax.set_ylabel(var)
        axes[-1].set_xlabel("draw")
        fig.suptitle("MCMC chain traces (mixing diagnostic)")
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/fig6_mcmc_rank.png", dpi=150)
        plt.close(fig)


def fig_delta1(meta):
    """Model-free propagation check: P(t+1 correct | t correct/wrong), per model."""
    names = meta["names"]
    d1 = meta["delta_1"]
    gc = [d1[n]["p_next_given_correct"] for n in names]
    gw = [d1[n]["p_next_given_wrong"] for n in names]

    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, gc, width=w, label="next step | this step correct", color="#27ae60")
    ax.bar(x + w / 2, gw, width=w, label="next step | this step wrong", color="#c0392b")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("P(next step correct)")
    ax.set_title("Model-free propagation check ($\\Delta_1$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig7_delta1.png", dpi=150)
    plt.close(fig)


def main():
    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    meta, idata = load_all()
    fig_p_vs_g(meta)
    fig_L_bars(meta)
    fig_posterior_forest(idata, meta)
    fig_pi_vs_scale(idata, meta)
    fig_error_type_by_depth(meta)
    fig_trace_diag(idata)
    fig_delta1(meta)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
