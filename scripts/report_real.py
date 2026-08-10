"""Headline numbers for the real run: everything Phase 5 has to report.

Reads only the artifacts the run wrote (`<tag>_meta.json`, `<tag>_idata.pkl`, and
the per-model JSONL) so nothing here can invent a figure that the run did not
produce. Every quantity printed is traceable to one of those files.

Sections, in the order they are reported:
  1. p_t, g_t, L_t by depth per model, with paired-bootstrap CIs
  2. the LINEAR control arm -- expected L_t ~ 0; if it is not, that outranks
     every other result here
  3. delta_1 per model (model-free propagation check)
  4. fitted pi / r_syn / r_sem, identifiability correlations, MCMC diagnostics
  5. H2 on BOTH scale axes, dense and MoE contrasts stated separately
  6. H4 by step index, pooled across models, with the per-model directional view
  7. simulated-vs-real agreement, where real is primary by pre-registration
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

RES = Path("data/results")


def load(tag: str):
    meta = json.loads((RES / f"{tag}_meta.json").read_text())
    idata = None
    p = RES / f"{tag}_idata.pkl"
    if p.exists():
        with open(p, "rb") as fh:
            idata = pickle.load(fh)
    return meta, idata


def short(name: str) -> str:
    return name.split("/")[-1]


def deepest_with_data(meta: dict, name: str) -> int | None:
    """Deepest depth where this model actually has a measured L_t.

    Needed because the planned grid runs to depth 8 but a cap-stopped or
    time-bounded sweep leaves a nested prefix that may not reach it. Reporting
    "L_t at depth 8 = nan" states nothing; reporting the deepest depth that has
    data, labelled with which depth that is, states what was measured.
    """
    for d in reversed(meta["depths"]):
        ci = meta["L_ci"].get(name, {}).get(str(d), {})
        L = ci.get("L")
        if L is not None and L == L:
            return d
    return None


def sec(title: str) -> None:
    print()
    print("=" * 84)
    print(title)
    print("=" * 84)


def report_rates(meta: dict) -> None:
    sec("1. CLEAN BASELINE p_t, FREE-RUNNING g_t, NET PROPAGATION LOSS L_t")
    depths = meta["depths"]
    alloc = meta.get("allocation") or {}
    for i, name in enumerate(meta["names"]):
        n_i = alloc.get(name, {}).get("primary")
        extra = {x["depth"]: x for x in meta["per_depth_extra"][name]}
        print(f"\n  {short(name)}"
              + (f"   planned n {n_i}" if n_i else ""))
        print(f"    {'depth':>6}{'p_t':>8}{'g_t':>8}{'L_t':>9}"
              f"{'89% CI on L_t':>22}{'n_g':>7}{'fresh':>7}")
        for j, d in enumerate(depths):
            ci = meta["L_ci"][name].get(str(d), {})
            p = meta["p"][i][j]
            trials = meta["trials"][i][j]
            succ = meta["successes"][i][j]
            g = succ / trials if trials else float("nan")
            L = ci.get("L", float("nan"))
            band = (f"[{ci['lo']:+.3f}, {ci['hi']:+.3f}]"
                    if "lo" in ci else "n/a")
            print(f"    {d:>6}{p:>8.3f}{g:>8.3f}{L:>9.3f}{band:>22}"
                  f"{trials:>7}{extra.get(d, {}).get('n_fresh_errors', 0):>7}")
    filled = meta.get("substituted_input_depths") or {}
    if filled:
        print(f"\n  WARNING: substituted (placeholder) inputs at "
              f"{ {short(k): v for k, v in filled.items()} }. p_t and/or f_syn "
              f"were degenerate there and were filled from a neighbouring depth, "
              f"so the FIT is partly driven by placeholders at those depths. "
              f"L_t remains the trustworthy quantity.")


def report_control(meta: dict) -> None:
    sec("2. LINEAR CONTROL ARM -- expected L_t ~ 0")
    ctrl = meta.get("control_arm") or {}
    if not ctrl:
        print("  NOT RUN. The control arm is the null that shows the primary "
              "arm's propagation signal comes from task structure rather than "
              "from the harness or the scorer, so its absence is a real gap.")
        return
    violations = []
    for name, info in ctrl.items():
        print(f"\n  {short(name)}  ({info['variant']}, n={info.get('per_depth')})")
        print(f"    {'depth':>6}{'p_t':>8}{'g_t':>8}{'L_t':>9}{'89% CI':>22}")
        for row in info["by_depth"]:
            d = row["depth"]
            ci = info["L_ci"].get(str(d), {})
            band = (f"[{ci['lo']:+.3f}, {ci['hi']:+.3f}]" if "lo" in ci else "n/a")
            print(f"    {d:>6}{row['p_t']:>8.3f}{row['g_t']:>8.3f}"
                  f"{row['L_t']:>9.3f}{band:>22}")
            # a null is violated when the interval excludes 0 on the high side
            if "lo" in ci and ci["lo"] > 0.05 and d > 1:
                violations.append((short(name), d, ci))
    print()
    if violations:
        print("  !! CONTROL ARM IS NOT NULL. The linear suite shows propagation")
        print("     loss whose interval excludes 0 at:")
        for nm, d, ci in violations:
            print(f"       {nm} depth {d}: [{ci['lo']:+.3f}, {ci['hi']:+.3f}]")
        print("     This matters more than any other result in this report. The")
        print("     linear arm announces the tool order and copies the previous")
        print("     result verbatim, so a non-zero L_t there points at the")
        print("     harness or the scorer, not at the models. STOP and diagnose")
        print("     before interpreting the primary arm.")
    else:
        print("  Control arm is null as expected: no depth beyond 1 has a lower")
        print("  CI bound above +0.05. The propagation signal in the primary arm")
        print("  is attributable to task structure, not to the measurement path.")


def report_delta1(meta: dict) -> None:
    sec("3. delta_1 -- MODEL-FREE PROPAGATION CHECK")
    print(f"  {'model':<28}{'delta_1':>9}{'P(ok|ok)':>10}{'P(ok|bad)':>11}"
          f"{'n_ok':>8}{'n_bad':>8}")
    for name in meta["names"]:
        d = meta["delta_1"][name]
        print(f"  {short(name):<28}{d['delta_1']:>+9.3f}"
              f"{d['p_next_given_correct']:>10.3f}{d['p_next_given_wrong']:>11.3f}"
              f"{d['n_given_correct']:>8}{d['n_given_wrong']:>8}")
    print("\n  delta_1 > 0 means a wrong step makes the NEXT step more likely to")
    print("  be wrong -- propagation, with no parametric assumption at all.")
    zeros = [short(n) for n in meta["names"]
             if meta["delta_1"][n]["n_given_wrong"] >= 20
             and meta["delta_1"][n]["p_next_given_wrong"] == 0.0]
    if zeros:
        print()
        print(f"  P(next correct | this one wrong) is EXACTLY 0.000 for "
              f"{len(zeros)} model(s):")
        print(f"    {', '.join(zeros)}")
        print("  Not 'low' -- zero, over hundreds of transitions. Once the chain")
        print("  leaves the gold trajectory it never returns. So delta_1 reduces")
        print("  to P(next correct | this one correct), and the semantic recovery")
        print("  rate r_sem is 0 as a measurement rather than as an estimate.")
        print("  This is the sharpest divergence from the simulated policies,")
        print("  which were configured with r_sem between 0.05 and 0.30, and it")
        print("  matters for the fit: with no recovery to trade against, pi stops")
        print("  being confounded with r_sem (see the correlations in section 4).")


def report_fit(meta: dict, idata) -> None:
    sec("4. FITTED SEVERITY AND RECOVERY, WITH IDENTIFIABILITY DIAGNOSTICS")
    if idata is None:
        print("  no idata artifact; fit not run")
        return
    import arviz as az
    from tur.model.hierarchical import identifiability
    names = meta["names"]
    print(f"  priors centred on measured transitions: {meta.get('priors_used')}")
    print()
    print(f"  {'model':<26}{'pi':>19}{'r_syn':>19}{'r_sem':>19}"
          f"{'c(pi,rsyn)':>12}{'c(pi,rsem)':>12}")
    for i, name in enumerate(names):
        row = []
        for v in ("pi", "r_syn", "r_sem"):
            x = idata.posterior[v].values[:, :, i].ravel()
            lo, hi = np.percentile(x, [5.5, 94.5])
            row.append(f"{x.mean():.2f} [{lo:.2f},{hi:.2f}]")
        ident = identifiability(idata, i)
        print(f"  {short(name):<26}{row[0]:>19}{row[1]:>19}{row[2]:>19}"
              f"{ident['corr_pi_rsyn']:>12.3f}{ident['corr_pi_rsem']:>12.3f}")
    print()
    worst = max(max(abs(identifiability(idata, i)["corr_pi_rsyn"]),
                    abs(identifiability(idata, i)["corr_pi_rsem"]))
                for i in range(len(names)))
    print(f"  Largest |posterior correlation| between pi and a recovery rate: "
          f"{worst:.2f}.")
    print("  These are reported WITH the fit, not separately: a high correlation")
    print("  means the data cannot separate 'how bad is poisoning' from 'how fast")
    print("  do you recover', so pi is diagnostic rather than a headline. L_t is")
    print("  the fit-free quantity and carries the primary claims.")
    print()
    print("  MCMC diagnostics (targets: R-hat < 1.01, ESS > 400, 0 divergences)")
    ok = True
    for v in ("pi", "r_syn", "r_sem"):
        rhat = float(np.nanmax(az.rhat(idata)[v].values))
        ess = float(np.nanmin(az.ess(idata)[v].values))
        flag = "" if (rhat < 1.01 and ess > 400) else "   <-- OUT OF TARGET"
        ok &= rhat < 1.01 and ess > 400
        print(f"    {v:<8} max R-hat {rhat:.4f}   min ESS {ess:>8.0f}{flag}")
    div = int(idata.sample_stats["diverging"].values.sum())
    print(f"    divergences: {div}" + ("" if div == 0 else "   <-- RAISE target_accept"))
    print(f"    verdict: {'healthy' if ok and div == 0 else 'NOT healthy'}")


def report_h2(meta: dict, idata) -> None:
    sec("5. H2 -- SEVERITY vs SCALE, ON BOTH AXES")
    if idata is None:
        print("  no idata artifact; fit not run")
        return
    cfg = {m["name"]: m for m in meta.get("models_config", [])}
    names = meta["names"]
    pi = {n: float(idata.posterior["pi"].values[:, :, i].mean())
          for i, n in enumerate(names)}
    print("  Reported against BOTH total and active parameters, and the two")
    print("  contrasts are stated separately rather than averaged: they are not")
    print("  comparable in kind. A dense 8B->70B step is a ~8.75x change in both")
    print("  total and active parameters; a sparse 20B->120B step is 6x total but")
    print("  only 1.4x active. Pooling them into one 'scale' claim would hide that.")
    print()
    fams: dict[str, list] = {}
    for n in names:
        fams.setdefault(cfg.get(n, {}).get("family", n), []).append(n)
    for fam, members in fams.items():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda n: cfg.get(n, {}).get("scale") or 0)
        lo, hi = members[0], members[-1]
        s_lo = cfg.get(lo, {}).get("scale")
        s_hi = cfg.get(hi, {}).get("scale")
        a_lo = cfg.get(lo, {}).get("active_scale") or s_lo
        a_hi = cfg.get(hi, {}).get("active_scale") or s_hi
        kind = "sparse MoE" if cfg.get(lo, {}).get("active_scale") else "dense"
        print(f"  {fam} ({kind}): {short(lo)} -> {short(hi)}")
        print(f"    total params  {s_lo} -> {s_hi}  ({s_hi / s_lo:.2f}x)")
        print(f"    active params {a_lo} -> {a_hi}  ({a_hi / a_lo:.2f}x)")
        print(f"    pi            {pi[lo]:.3f} -> {pi[hi]:.3f}  "
              f"(delta {pi[hi] - pi[lo]:+.3f})")
        d_lo = [x for x in meta["per_depth_extra"][lo]]
        d_lo, d_hi = deepest_with_data(meta, lo), deepest_with_data(meta, hi)
        if d_lo and d_hi:
            print(f"    L_t: {meta['L_ci'][lo][str(d_lo)]['L']:+.3f} at depth "
                  f"{d_lo} -> {meta['L_ci'][hi][str(d_hi)]['L']:+.3f} at depth "
                  f"{d_hi}" + ("   (DIFFERENT depths -- not a like-for-like "
                               "contrast)" if d_lo != d_hi else ""))
        else:
            print("    L_t: not measurable for at least one of the pair yet")
        print()
    print("  NOTE: two points per family is not a scale CURVE. No free host offers")
    print("  3+ sizes of one family any more, so H2 is tested as a signed contrast")
    print("  at two points, not as a fitted trend.")


def report_h4(meta: dict) -> None:
    sec("6. H4 -- ERROR-TYPE COMPOSITION BY STEP INDEX")
    from scipy import stats as sps
    per_step = meta.get("per_step") or {}
    if not per_step:
        print("  no per-step aggregation in metadata")
        return
    max_step = max(len(v) for v in per_step.values())
    pooled_sel = np.zeros(max_step)
    pooled_arg = np.zeros(max_step)
    pooled_n = np.zeros(max_step)
    pooled_syn = np.zeros(max_step)

    print("  per-model (directional only -- per-model counts do not support a")
    print("  per-model trend claim at this n; see the power note in the paper):")
    print(f"    {'model':<24}{'rho(sel/arg, step)':>20}{'p':>8}{'fresh errors':>14}")
    for name, rows in per_step.items():
        ratios = [(r["step"], r["sel_to_arg"]) for r in rows
                  if r["sel_to_arg"] == r["sel_to_arg"]]
        tot = sum(r["n_fresh_errors"] for r in rows)
        if len(ratios) >= 3:
            xs, ys = zip(*ratios)
            rho, p = sps.spearmanr(xs, ys)
            print(f"    {short(name):<24}{rho:>+20.3f}{p:>8.3f}{tot:>14}")
        else:
            print(f"    {short(name):<24}{'too few bins':>20}{'':>8}{tot:>14}")
        for r in rows:
            i = r["step"]
            n = r["n_fresh_errors"]
            if not n:
                # A step with no fresh errors carries NaN shares by construction
                # (0/0). It must be skipped, not weighted by zero: nan * 0 is nan,
                # not 0, so a single model contributing no errors at a step index
                # turned the whole pooled share for that step into nan. That is
                # exactly what a ceiling-level model does at every step, so the
                # pooled H4 table came out entirely nan while showing healthy
                # error counts beside it.
                continue
            pooled_n[i] += n
            pooled_sel[i] += (r["sel_err_share"] or 0.0) * n
            pooled_arg[i] += (r["arg_err_share"] or 0.0) * n
            pooled_syn[i] += (r["f_syn"] or 0.0) * n

    print()
    print("  POOLED across models (the suite-level claim H4 is reported as):")
    print(f"    {'step':>5}{'fresh errors':>14}{'f_syn':>8}{'sel share':>11}"
          f"{'arg share':>11}{'sel/arg':>9}{'usable':>8}")
    sel_ratio, steps_ok = [], []
    for i in range(max_step):
        n = pooled_n[i]
        if n <= 0:
            continue
        fs, sel, arg = pooled_syn[i] / n, pooled_sel[i] / n, pooled_arg[i] / n
        ratio = sel / arg if arg > 0 else float("nan")
        usable = n >= 30
        if usable and ratio == ratio:
            sel_ratio.append((i, ratio))
            steps_ok.append(i)
        print(f"    {i:>5}{n:>14.0f}{fs:>8.3f}{sel:>11.3f}{arg:>11.3f}"
              f"{ratio:>9.3f}{'yes' if usable else 'THIN':>8}")
    print()
    # Distinguish the two very different reasons H4 can fail to be testable.
    n_usable = sum(1 for i in range(max_step) if pooled_n[i] >= 30)
    arg_total = sum(pooled_arg[i] for i in range(max_step))
    sel_total = sum(pooled_sel[i] for i in range(max_step))
    if len(sel_ratio) >= 4:
        xs, ys = zip(*sel_ratio)
        rho, p = sps.spearmanr(xs, ys)
        verdict = ("selection-dominated early, argument-dominated later "
                   "(H4 as stated)" if rho < 0 else
                   "argument share FALLS with depth (opposite of H4)")
        print(f"  pooled sel/arg vs step index over {len(xs)} usable bins "
              f"{list(xs)}:")
        print(f"    Spearman rho = {rho:+.3f}, p = {p:.4f}  ->  "
              f"{'SIGNIFICANT' if p < 0.05 else 'not significant'}")
        print(f"    direction: {verdict}")
    elif arg_total < 1 and sel_total > 20:
        print("  H4 IS NOT APPLICABLE ON THIS TASK VARIANT, and the reason is")
        print("  more informative than a power limitation.")
        print()
        print(f"  Of {sel_total + arg_total:.0f} fresh (clean-context) errors, "
              f"{sel_total:.0f} are SELECTION errors and {arg_total:.0f} are")
        print("  argument errors. The argument channel is empty, so the")
        print("  selection-to-argument ratio H4 is about is undefined at every")
        print("  step index -- there is no mix to shift.")
        print()
        print("  Mechanism: on a routing task a fresh error means the parity rule")
        print("  was mis-applied, i.e. the wrong tool was named. Transcribing the")
        print("  argument is trivial by comparison -- it is copied verbatim from")
        print("  the stated previous result -- and these models essentially never")
        print("  get it wrong while holding a correct value. Argument errors DO")
        print("  appear once the context is already poisoned, because the rule")
        print("  applied to a corrupted value can coincidentally name the gold")
        print("  tool while carrying the wrong number; but those are propagated")
        print("  errors, which the composition deliberately excludes.")
        print()
        print("  So H4 is withdrawn for this run, not because the counts are thin")
        print("  but because one of its two categories does not occur where the")
        print("  composition is defined. Testing it would need a task variant")
        print("  whose arguments can be got wrong independently of tool choice.")
    else:
        print(f"  ONLY {n_usable} step bins clear 30 fresh errors and only "
              f"{len(sel_ratio)} yield a defined")
        print("  sel/arg ratio, fewer than the 4 a rank-correlation trend needs to")
        print("  reach p<0.05 even at rho = -1. H4 IS WITHDRAWN for this run")
        print("  rather than reported on counts that cannot support it.")


def report_bias(meta: dict, tag: str) -> None:
    """Rule-following vs position bias, and the parity structure of held values.

    Reads `held_ref` and `first_listed_even` straight off the records, so this runs
    on any condition without a cache replay.

    The confound this section exists to break: with a FIXED presentation order the
    correct tool for an even ref is also the first-listed tool, so "always picks the
    first-listed option" and "follows the rule but only succeeds on even refs"
    predict the same data. Two things separate them. The discrimination statistic
    P(pick first-listed | even) - P(pick first-listed | odd) is 0 for a pure position
    bias and large for genuine rule-following, whatever the accuracy. And under
    `shuffle_branch_order` the order varies per step, which decouples the two
    directly.
    """
    sec("8. RULE-FOLLOWING vs POSITION BIAS, AND PARITY STRUCTURE")
    depths = meta["depths"]
    any_data = False
    for name in meta["names"]:
        path = RES / f"{tag}_{name.replace('/', '_')}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.open() if l.strip()]
        fr = [r for r in rows if r["run_mode"] == "free"
              and not r.get("backend_error", False)
              and r.get("held_ref") is not None]
        if len(fr) < 30:
            continue
        any_data = True
        even = [r for r in fr if r["held_ref"] % 2 == 0]
        odd = [r for r in fr if r["held_ref"] % 2 == 1]
        acc = lambda rs: (sum(r["selection_correct"] for r in rs) / len(rs)
                          if rs else float("nan"))
        print("\n  " + short(name) +
              f"   (n={len(fr)} free steps with a recorded held ref)")
        print(f"    rule-application accuracy: held even {acc(even):.3f} "
              f"(n={len(even)})   held odd {acc(odd):.3f} (n={len(odd)})")
        # did it pick the FIRST-LISTED tool? recoverable from selection_correct plus
        # the parity and the order: correct tool is the even-branch tool iff held is
        # even, and the even branch is listed first iff first_listed_even
        def picked_first(r):
            correct_is_even_branch = r["held_ref"] % 2 == 0
            first_is_even = r.get("first_listed_even", True)
            picked_even_branch = (correct_is_even_branch
                                  if r["selection_correct"] else
                                  not correct_is_even_branch)
            return picked_even_branch == first_is_even
        pe = [picked_first(r) for r in even]
        po = [picked_first(r) for r in odd]
        if pe and po:
            fe, fo = sum(pe) / len(pe), sum(po) / len(po)
            disc = fe - fo
            print(f"    picks the FIRST-LISTED tool: ref even {fe:.3f}   "
                  f"ref odd {fo:.3f}   discrimination {disc:+.3f}")
            verdict = ("POSITION BIAS -- barely conditions on the ref at all"
                       if abs(disc) < 0.10 else
                       "conditions on the ref, i.e. genuinely applies the rule")
            print(f"      -> {verdict}")
        orders = {r.get("first_listed_even", True) for r in fr}
        if len(orders) > 1:
            print("    presentation order was RANDOMISED, so parity and position are")
            print("    decoupled; the four cells below settle it without inference:")
            for par, pname in ((0, "even"), (1, "odd ")):
                for fl in (True, False):
                    cell = [r for r in fr if r["held_ref"] % 2 == par
                            and r.get("first_listed_even", True) is fl]
                    if len(cell) >= 10:
                        print(f"      held {pname}, even-branch listed "
                              f"{'first ' if fl else 'second'}: "
                              f"acc={acc(cell):.3f} (n={len(cell)})")
        else:
            print("    presentation order was FIXED (even branch always first), so")
            print("    position and parity are confounded here; the discrimination")
            print("    statistic above is what separates them. See the `shuffle`")
            print("    condition for the direct control.")
        # parity balance of held values, which differs between conditions
        print(f"    share of held refs that are even: {len(even) / len(fr):.3f}"
              f"   (a skew here confounds any clean-vs-poisoned comparison)")
    if not any_data:
        print("  No records carry `held_ref` yet -- it is recorded from this run")
        print("  onward. Re-run to populate (cached completions replay for free).")


def report_agreement(meta: dict) -> None:
    sec("7. SIMULATED vs REAL -- real is primary by pre-registration")
    val = RES / "routingval_meta.json"
    if not val.exists():
        print("  no routing validation artifact to compare against")
        return
    v = json.loads(val.read_text())
    print("  Validation (simulated routing policies) vs real models, on the")
    print("  quantities the validation could actually establish:")
    print()
    real_pairs = [(deepest_with_data(meta, n), n) for n in meta["names"]]
    real_pairs = [(d, n) for d, n in real_pairs if d]
    if real_pairs:
        common = min(d for d, _ in real_pairs)
        sim_L = [v["L_ci"][n][str(common)]["L"] for n in v["names"]
                 if str(common) in v["L_ci"][n]]
        real_L = [meta["L_ci"][n][str(common)]["L"] for _, n in real_pairs]
        real_L = [x for x in real_L if x == x]
        print(f"    L_t at depth {common} (deepest depth every real model "
              f"reached): simulated "
              f"[{min(sim_L):+.3f}, {max(sim_L):+.3f}] vs real "
              f"[{min(real_L):+.3f}, {max(real_L):+.3f}]")
        deepest = max(d for d, _ in real_pairs)
        if deepest != common:
            print(f"    (the deepest bin any real model reached is {deepest}; "
                  f"per-model depths differ because the sweep is incomplete)")
    sim_d1 = [v["delta_1"][n]["delta_1"] for n in v["names"]]
    real_d1 = [meta["delta_1"][n]["delta_1"] for n in meta["names"]]
    print(f"    delta_1: simulated [{min(sim_d1):+.3f}, {max(sim_d1):+.3f}], "
          f"real [{min(real_d1):+.3f}, {max(real_d1):+.3f}]")
    print()
    print("  Divergence between the two is a finding, not something to reconcile")
    print("  away: the simulated policies were constructed to have the propagation")
    print("  structure the model assumes, and real models were not.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="real")
    args = ap.parse_args()
    meta, idata = load(args.tag)

    print(f"REAL-RUN REPORT  tag={args.tag}  variant="
          f"{meta.get('task_variant', '?')}  models={len(meta['names'])}")
    if meta.get("achieved_shortfall"):
        print("\n  NOTE: some models did not reach their planned allocation. The")
        print("  achieved counts are nested prefixes and are used as such:")
        for name, info in meta["achieved_shortfall"].items():
            print(f"    {short(name)}: achieved {info['achieved']} of "
                  f"{info['requested']}"
                  + ("  (cap-stopped)" if info.get("capped") else ""))
    if meta.get("excluded_from_fit"):
        print(f"\n  EXCLUDED from the fit: "
              f"{ {short(k): v for k, v in meta['excluded_from_fit'].items()} }")
    if meta.get("structural_anomalies"):
        print("\n  !! STRUCTURAL ANOMALIES FLAGGED BY THE RUNNER:")
        for name, flags in meta["structural_anomalies"].items():
            for f in flags:
                print(f"    {short(name)}: {f}")

    report_rates(meta)
    report_control(meta)
    report_delta1(meta)
    report_fit(meta, idata)
    report_h2(meta, idata)
    report_h4(meta)
    report_agreement(meta)
    report_bias(meta, args.tag)


if __name__ == "__main__":
    main()
