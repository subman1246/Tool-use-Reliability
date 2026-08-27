"""Reproduce figures that use only aggregate values reported in main.tex.

No task-level records are available in this Prism workspace.  Accordingly,
this script contains no resampling or newly inferred confidence intervals.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(".")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def severity_comparison() -> None:
    models = ["llama-3.1-8b-instant", "allam-2-7b"]
    conditional = np.array([0.149, 0.316])
    lower = np.array([0.021, 0.066])
    upper = np.array([0.268, 0.503])
    y = np.arange(len(models))[::-1]

    fig, ax = plt.subplots(figsize=(5.2, 1.75))
    ax.errorbar(
        conditional,
        y,
        xerr=np.vstack([conditional - lower, upper - conditional]),
        fmt="o",
        color="#1a5276",
        capsize=3,
        linewidth=1.4,
        label="conditional-on-state (89% interval)",
        zorder=3,
    )
    ax.scatter(
        np.ones_like(y),
        y,
        marker="D",
        s=30,
        color="#c0392b",
        label="fixed-gold agreement (boundary)",
        zorder=3,
    )
    for yi, value in zip(y, conditional):
        ax.text(value + 0.025, yi + 0.10, f"{value:.3f}", color="#1a5276")

    ax.axvline(0, color="0.55", linewidth=0.8)
    ax.axvline(1, color="0.75", linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.03, 1.08)
    ax.set_yticks(y, models)
    ax.set_xlabel(r"severity  $1-P(\mathrm{correct}\mid\mathrm{corrupted})/P(\mathrm{correct}\mid\mathrm{clean})$")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_severity_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def propagation_gap() -> None:
    data = {
        "llama-3.1-8b-instant": {
            "depth": np.array([1, 2, 4, 6, 8]),
            "p": np.array([42 / 60, 83 / 120, 156 / 260, 223 / 390, 100 / 184]),
            "g": np.array([42 / 60, 71 / 120, 95 / 260, 70 / 390, 32 / 184]),
        },
        "allam-2-7b": {
            "depth": np.array([1, 2, 4, 6]),
            "p": np.array([32 / 60, 46 / 120, 111 / 260, 114 / 240]),
            "g": np.array([32 / 60, 42 / 120, 54 / 260, 36 / 240]),
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharey=True)
    for ax, (name, values) in zip(axes, data.items()):
        d, p, g = values["depth"], values["p"], values["g"]
        ax.plot(d, p, "o-", color="#1a5276", label=r"$\bar p_d$ (baseline)")
        ax.plot(d, g, "s--", color="#c0392b", label=r"$\bar g_d$ (free-running)")
        ax.fill_between(d, g, p, color="#c0392b", alpha=0.09)
        ax.set_title(name)
        ax.set_xlabel("dependency depth")
        ax.set_xticks([1, 2, 4, 6, 8])
        ax.set_xlim(0.7, 8.3)
        ax.set_ylim(0, 0.78)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("pooled invocation correctness")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_propagation_gap.pdf", bbox_inches="tight")
    plt.close(fig)


def metric_sensitivity() -> None:
    # Unique integer counts consistent with Table 3's n and three-decimal rates.
    data = {
        "llama-3.1-8b-instant": {
            "depth": np.array([1, 2, 4, 6, 8]),
            "p": np.array([42 / 60, 83 / 120, 156 / 260, 223 / 390, 100 / 184]),
            "g": np.array([42 / 60, 71 / 120, 95 / 260, 70 / 390, 32 / 184]),
            "color": "#1a5276",
        },
        "allam-2-7b": {
            "depth": np.array([1, 2, 4, 6]),
            "p": np.array([32 / 60, 46 / 120, 111 / 260, 114 / 240]),
            "g": np.array([32 / 60, 42 / 120, 54 / 260, 36 / 240]),
            "color": "#c0392b",
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35))
    for name, values in data.items():
        depth = values["depth"]
        p = values["p"]
        g = values["g"]
        axes[0].plot(depth, p - g, "o-", color=values["color"], label=name)
        axes[1].plot(depth, (1 - g) / (1 - p), "o-", color=values["color"], label=name)

    axes[0].axhline(0, color="0.65", linewidth=0.8)
    axes[1].axhline(1, color="0.65", linewidth=0.8)
    axes[0].set_ylabel(r"absolute excess error  $\bar p_d-\bar g_d$")
    axes[1].set_ylabel(r"error amplification  $(1-\bar g_d)/(1-\bar p_d)$")
    for ax in axes:
        ax.set_xlabel("dependency depth")
        ax.set_xticks([1, 2, 4, 6, 8])
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_metric_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def discrimination() -> None:
    models = [
        "qwen3.6-27b",
        "gpt-oss-120b",
        "llama-3.3-70b\n-versatile",
        "llama-3.1-8b\n-instant",
        "allam-2-7b",
    ]
    values = np.array([1.000, 0.978, 0.828, 0.146, -0.176])
    colors = ["#6c3483", "#7d6608", "#117864", "#1a5276", "#c0392b"]

    fig, ax = plt.subplots(figsize=(5.75, 3.0))
    y = np.arange(len(models))
    ax.barh(y, values, color=colors, height=0.56)
    ax.set_yticks(y, models)
    ax.set_xlim(-0.48, 1.15)
    ax.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel(
        r"discrimination  $P(\mathrm{first\!\!-listed}\mid\mathrm{even})"
        r"-P(\mathrm{first\!\!-listed}\mid\mathrm{odd})$"
    )
    for yi, value in zip(y, values):
        if value >= 0:
            ax.text(value + 0.03, yi, f"+{value:.3f}", va="center", ha="left")
        else:
            ax.text(value - 0.03, yi, f"{value:.3f}", va="center", ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", width=0.8)
    fig.savefig(OUT / "fig_discrimination.pdf", bbox_inches="tight")
    plt.close(fig)




def lt_all_models() -> None:
    # All five usable models' L_d with reported 89% intervals (Table 2 / Figure 3 in main.tex).
    # Colors matched to discrimination() for a consistent per-model palette across figures.
    series = {
        "llama-3.1-8b-instant": {
            "depth": np.array([1, 2, 4, 6, 8]),
            "L": np.array([0.000, 0.145, 0.391, 0.686, 0.680]),
            "lo": np.array([0.000, 0.082, 0.311, 0.607, 0.574]),
            "hi": np.array([0.000, 0.222, 0.474, 0.766, 0.772]),
            "color": "#1a5276", "marker": "o",
        },
        "allam-2-7b": {
            "depth": np.array([1, 2, 4, 6]),
            "L": np.array([0.000, 0.087, 0.514, 0.684]),
            "lo": np.array([0.000, 0.022, 0.429, 0.624]),
            "hi": np.array([0.000, 0.163, 0.600, 0.745]),
            "color": "#c0392b", "marker": "s",
        },
        "llama-3.3-70b-versatile": {
            "depth": np.array([4, 6]),
            "L": np.array([0.000, 0.190]),
            "lo": np.array([0.000, 0.062]),
            "hi": np.array([0.000, 0.345]),
            "color": "#117864", "marker": "^",
        },
        "gpt-oss-120b": {
            "depth": np.array([4, 6]),
            "L": np.array([0.027, 0.044]),
            "lo": np.array([0.000, 0.000]),
            "hi": np.array([0.083, 0.135]),
            "color": "#7d6608", "marker": "D",
        },
        "qwen3.6-27b": {
            "depth": np.array([4, 6]),
            "L": np.array([0.000, 0.000]),
            "lo": np.array([0.000, 0.000]),
            "hi": np.array([0.000, 0.000]),
            "color": "#6c3483", "marker": "v",
        },
    }

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for name, d in series.items():
        yerr = np.vstack([d["L"] - d["lo"], d["hi"] - d["L"]])
        ax.errorbar(d["depth"], d["L"], yerr=yerr, fmt=d["marker"] + "-", color=d["color"],
                    label=name, capsize=2.5, linewidth=1.4, markersize=4.5, alpha=0.92)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("dependency depth")
    ax.set_ylabel(r"$L_d = 1 - \bar g_d/\bar p_d$")
    ax.set_xticks([1, 2, 4, 6, 8])
    ax.set_ylim(-0.05, 0.82)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_Lt_all_models.pdf", bbox_inches="tight")
    plt.close(fig)


def posterior_predictive() -> None:
    """Posterior predictive check for the fitted severity/recovery model.

    Numbers are read from data/results/real_ppc.json, produced by
    scripts/posterior_predictive_check.py, which reuses the already-fit trace and
    makes no model queries.  The observed value is a structural zero: no call made
    on a corrupted context matched the canonical target anywhere in the dataset.
    """
    import json

    src = Path(__file__).resolve().parents[1] / "data" / "results" / "real_ppc.json"
    d = json.loads(src.read_text())
    rep = np.array(d["replicates"], dtype=float)
    observed = d["observed_poisoned_matches"]
    n_pois = d["observed_poisoned_calls"]

    fig, ax = plt.subplots(figsize=(5.75, 2.6))
    ax.hist(rep, bins=np.arange(rep.min() - 0.5, rep.max() + 1.5, 2.0),
            color="#1a5276", edgecolor="none", alpha=0.85,
            label=f"posterior predictive replicates (n={rep.size:,})")
    ax.axvline(observed, color="#c0392b", linewidth=1.6,
               label=f"observed = {observed}")
    ax.set_xlabel(
        f"replicated canonical matches among the {n_pois} corrupted-context calls"
    )
    ax.set_ylabel("replicates")
    ax.set_xlim(-4, rep.max() * 1.04)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", width=0.8)
    ax.annotate(
        "no replicate reaches the observed value"
        + "\n(minimum %d; mean %.0f)" % (int(rep.min()), rep.mean()),
        xy=(observed, ax.get_ylim()[1] * 0.55),
        xytext=(rep.mean() * 0.42, ax.get_ylim()[1] * 0.72),
        arrowprops=dict(arrowstyle="->", color="#c0392b", linewidth=0.9),
        color="#c0392b", ha="left",
    )
    fig.savefig(OUT / "fig_ppc.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    severity_comparison()
    propagation_gap()
    metric_sensitivity()
    discrimination()
    lt_all_models()
    posterior_predictive()
