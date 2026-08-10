"""Estimate call volume, token volume and rate-limit feasibility of a real run
BEFORE spending any API budget. Run this first, before scripts/run_real_suite.py.

This does not call any model. It counts exactly how many backend calls the
configured sweep will make (both run modes, with and without retries), counts
the prompt tokens the harness actually builds with a real tokenizer, and checks
the result against each provider's measured rate limits.

The rate-limit check is the point of this script. On free tiers the money cost
is zero and the binding constraint is throughput -- specifically tokens per
minute, which is far more restrictive than the headline requests-per-day figure
that provider docs lead with. A sweep can be well inside its daily request cap
and still take nine hours per model.

Config is the source of truth; CLI flags override it for what-if analysis.
"""

from __future__ import annotations

import argparse
import json

from tur.tasks.dag import generate_suite, generate_routing_suite
from tur.harness.executor import FeedbackMode
from tur.harness.runner import MockBackend, run_free, run_teacher_forced

# Rough $/1M tokens (input, output) for cost accounting only. Every model in the
# current suite is served on a free tier, hence 0.0 -- the entries exist so a
# paid model added later is still costed rather than silently reported as free.
PRICE_PER_1M = {
    "groq/": (0.0, 0.0),                 # Groq free tier
    "gemini/": (0.0, 0.0),               # Gemini API free tier
    "openrouter/": (0.0, 0.0),           # OpenRouter :free models
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-haiku": (0.80, 4.00),
    "claude-sonnet": (3.00, 15.00),
    "default_open_weight": (0.20, 0.20),
    "default_proprietary": (1.00, 3.00),
}

# {model: (requests_per_day, tokens_per_minute, tokens_per_day)}.
#
# RPD/TPM measured from x-ratelimit-limit-* response headers on 2026-08-09, not
# taken from documentation -- the docs quote an org-wide 14,400 req/day for Groq,
# but the enforced limit is per model and is 1,000/day for most of them.
#
# TPD is measured from the BODY of the 429 that enforces it, on 2026-08-10. It
# appears in no response header: every model returns "-" for
# x-ratelimit-limit-tokens-per-day, so there is no way to discover it except by
# being refused. It is also NOT uniform across the suite -- llama-3.3-70b gets
# half what the others get -- so it cannot be inferred from one observation.
# TPD is the binding constraint for this workload by a wide margin.
RATE_LIMITS = {
    "groq/llama-3.1-8b-instant":    (14400, 6000, None),
    "groq/llama-3.3-70b-versatile": (1000, 12000, 100_000),
    "groq/qwen/qwen3.6-27b":        (1000, 8000, 200_000),
    "groq/openai/gpt-oss-20b":      (1000, 8000, 200_000),
    "groq/openai/gpt-oss-120b":     (1000, 8000, 200_000),
    "groq/allam-2-7b":              (7000, 6000, None),
    "gemini/gemini-2.5-flash":      (None, None, None),
}

_OUTPUT_TOKENS_PER_CALL = 40   # one short JSON call


def _price_for(model: str) -> tuple[float, float]:
    for key, price in PRICE_PER_1M.items():
        if key in model:
            return price
    return PRICE_PER_1M["default_proprietary" if "gpt" in model or "claude" in model
                        else "default_open_weight"]


def _tokenizer():
    """Real tokenizer if available, else a chars/4 fallback.

    The fallback is flagged in the output because it materially changes the
    TPM feasibility verdict, which is the number this script exists to produce.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return (lambda s: len(enc.encode(s))), "tiktoken/o200k_base"
    except Exception:
        return (lambda s: len(s) // 4), "chars/4 APPROXIMATION"


def count_sweep(depths, per_depth, seeds, max_retries, distractor_level,
                variant="routing"):
    """Exact call counts and measured prompt-token totals for one model.

    Drives the REAL run loops against a mock backend and counts tokens on the
    messages they actually build, rather than reimplementing the prompt
    assembly here. An earlier version duplicated that logic and silently went
    stale the moment the run loop changed, under-reporting the sweep.

    `variant` matters for cost, not just for the science: a routing task carries
    TWO candidate tools per step plus distractors, so its schema block is about
    twice the size of a linear task's at the same depth. Estimating the routing
    sweep with the linear generator understated the token bill.
    """
    ntok, tok_name = _tokenizer()
    gen = generate_routing_suite if variant == "routing" else generate_suite
    suite = gen(depths, per_depth, distractor_level)

    calls_min = 0
    prompt_tokens = 0

    def perfect(task, step, ref, attempt):
        return task.gold[step].tool, {"ref": ref}, True

    for task in suite:
        for runner in (run_free, run_teacher_forced):
            counted = {"calls": 0, "tokens": 0}
            backend = MockBackend(perfect)
            inner = backend.complete

            def complete(messages, tools, mode, _c=counted, _inner=inner):
                text = "".join(str(m.get("content", "")) for m in messages)
                _c["tokens"] += ntok(text)
                _c["calls"] += 1
                return _inner(messages, tools, mode)

            backend.complete = complete
            runner(task, backend, "uniform", FeedbackMode.STRUCTURED, max_retries)
            calls_min += counted["calls"]
            prompt_tokens += counted["tokens"]

    return {
        "tasks": len(suite),
        "tokenizer": tok_name,
        "calls_min": calls_min * seeds,
        "calls_worst": calls_min * (max_retries + 1) * seeds,
        "prompt_tokens": prompt_tokens * seeds,
        "output_tokens": calls_min * seeds * _OUTPUT_TOKENS_PER_CALL,
    }


def estimate(depths, per_depth, seeds, max_retries, models, distractor_level=1,
             days=3, headroom=0.80, variant="routing", control=None):
    s = count_sweep(depths, per_depth, seeds, max_retries, distractor_level,
                    variant)
    if control:
        # The control arm is part of the bill, so it is part of the estimate.
        # Reporting only the primary arm would understate the sweep by exactly
        # the amount that later shows up as an unexpected cap-stop.
        c = count_sweep(control["depths"], control["per_depth"], seeds,
                        max_retries, distractor_level, control["variant"])
        print(f"Control arm ({control['variant']}, depths={control['depths']}): "
              f"{c['calls_min']:,} calls, {c['prompt_tokens'] + c['output_tokens']:,} "
              f"tokens -- included in the totals below")
        for k in ("tasks", "calls_min", "calls_worst", "prompt_tokens",
                  "output_tokens"):
            s[k] += c[k]
    total_tokens = s["prompt_tokens"] + s["output_tokens"]

    print(f"Sweep: variant={variant} depths={depths} seeds={seeds} "
          f"max_retries={max_retries} distractor_level={distractor_level}")
    print(f"per_depth: {per_depth}")
    print(f"Token counts via: {s['tokenizer']}")
    print(f"Tasks per seed: {s['tasks']}  |  total tasks: {s['tasks'] * seeds}")
    print()
    print(f"PER MODEL:")
    print(f"  backend calls, no retries : {s['calls_min']:,}")
    print(f"  backend calls, worst case : {s['calls_worst']:,}  "
          f"(every step exhausts {max_retries} retry)")
    print(f"  prompt tokens             : {s['prompt_tokens']:,}")
    print(f"  output tokens (est.)      : {s['output_tokens']:,}")
    print(f"  total tokens              : {total_tokens:,}")
    print(f"  avg tokens/call           : {total_tokens / max(s['calls_min'], 1):,.0f}")
    print()

    # ---- cost ----
    print(f"{'model':<34}{'worst-case $':>14}{'typical $':>12}")
    print("-" * 60)
    grand = 0.0
    for m in models:
        name = m["name"] if isinstance(m, dict) else m
        pin, pout = _price_for(name)
        worst = (s["prompt_tokens"] * (max_retries + 1) / 1e6) * pin \
            + (s["output_tokens"] * (max_retries + 1) / 1e6) * pout
        typical = (s["prompt_tokens"] / 1e6) * pin + (s["output_tokens"] / 1e6) * pout
        grand += typical
        print(f"{name:<34}{'$' + format(worst, ',.2f'):>14}"
              f"{'$' + format(typical, ',.2f'):>12}")
    print("-" * 60)
    print(f"{'TOTAL (typical)':<34}{'':>14}{'$' + format(grand, ',.2f'):>12}")
    print()

    # ---- rate-limit feasibility: the actual constraint ----
    # TPD is checked FIRST and reported as the headline, because it is the
    # constraint that decides whether the run is possible and it is the one this
    # script used to omit entirely. RPD and TPM both looked comfortable while the
    # sweep was in fact an 18-day run: 2.9M tokens per model against a daily
    # token allowance of 100,000-200,000. TPD appears in no response header, so
    # the figures below come from the body of the 429 that enforces them.
    print("RATE-LIMIT FEASIBILITY (TPD from 429 bodies; RPD/TPM from headers)")
    print(f"  pacing at {headroom:.0%} of each ceiling, target {days} day(s)")
    print()
    print(f"{'model':<32}{'TPD':>9}{'tok/day':>10}{'DAYS':>6}"
          f"{'RPD':>7}{'TPM':>7}{'hours':>7}  binding")
    print("-" * 92)
    infeasible, warnings, day_needs = [], [], []
    for m in models:
        name = m["name"] if isinstance(m, dict) else m
        limits = RATE_LIMITS.get(name)
        per_day_calls = s["calls_min"] / days
        per_day_tokens = total_tokens / days
        if not limits or limits[0] is None:
            print(f"{name:<32}{'?':>9}{per_day_tokens:>10,.0f}{'?':>6}{'?':>7}"
                  f"{'?':>7}{'?':>7}  UNKNOWN -- verify before running")
            warnings.append(name)
            continue
        rpd, tpm = limits[0], limits[1]
        tpd = limits[2] if len(limits) > 2 else None

        days_rpd = s["calls_min"] / rpd
        hours = total_tokens / (tpm * headroom) / 60
        if tpd:
            days_tpd = total_tokens / (tpd * headroom)
            days_needed = max(days_rpd, days_tpd)
            binding = "TPD" if days_tpd >= days_rpd else "RPD"
            tpd_txt = f"{tpd:,}"
        else:
            days_tpd, days_needed, binding = 0.0, days_rpd, "RPD (TPD unmeasured)"
            tpd_txt = "?"
            warnings.append(name)
        day_needs.append((name, days_needed))
        if days_needed > days:
            infeasible.append((name, days_needed))
            binding += f" -> needs {days_needed:.1f}d"
        print(f"{name:<32}{tpd_txt:>9}{per_day_tokens:>10,.0f}"
              f"{days_needed:>6.1f}{rpd:>7,}{tpm:>7,}{hours:>7.1f}  {binding}")
    print("-" * 92)
    print()
    if infeasible:
        worst = max(d for _, d in infeasible)
        print(f"!! {len(infeasible)} model(s) cannot finish in {days} day(s).")
        for name, d in sorted(infeasible, key=lambda x: -x[1]):
            print(f"     {name:<34} needs {d:.1f} days")
        print(f"   Models are swept CONCURRENTLY and caps are per model, so")
        print(f"   calendar time is the slowest model: ~{worst:.1f} days, not the sum.")
        print("   Options: cut per_depth, shorten the depth grid (cost is")
        print("   superlinear in depth -- the deepest bin dominates), raise")
        print("   --days, or move off the free tier.")
    if warnings:
        print(f"!  unmeasured TPD or worst-case retry volume for: {warnings}")
        print("   An unmeasured TPD is NOT an absent one. Every model in this")
        print("   suite has one and none of them publish it in a header, so a")
        print("   '?' here means the estimate below is a lower bound on days.")
    if not infeasible and not warnings:
        print("All models fit inside their measured caps at this configuration.")
    print()
    print("Wall-clock hours assume the run is TPM-bound within a day, which on")
    print("these free tiers it is. DAYS assumes it is TPD-bound across days,")
    print("which it also is -- those are two different ceilings and the run has")
    print("to clear both.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--depths", type=int, nargs="+", default=None)
    ap.add_argument("--per-depth", type=int, default=None,
                    help="override the config's per-depth allocation with a "
                         "single uniform count (what-if analysis)")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--max-retries", type=int, default=None)
    ap.add_argument("--distractors", type=int, default=None)
    ap.add_argument("--days", type=int, default=3,
                    help="days the sweep is spread over (default: %(default)s)")
    ap.add_argument("--headroom", type=float, default=0.80,
                    help="fraction of each TPM ceiling to pace at")
    args = ap.parse_args()

    import yaml
    try:
        cfg = yaml.safe_load(open(args.config))
    except FileNotFoundError:
        raise SystemExit(f"no config at {args.config}")

    per_depth = cfg.get("per_depth", 200)
    if isinstance(per_depth, dict):
        per_depth = {int(k): int(v) for k, v in per_depth.items()}
    if args.per_depth is not None:
        per_depth = args.per_depth

    ctrl = cfg.get("control_arm") or None
    if ctrl:
        c_pd = ctrl.get("per_depth", 20)
        if isinstance(c_pd, dict):
            c_pd = {int(k): int(v) for k, v in c_pd.items()}
        ctrl = {"variant": ctrl.get("task_variant", "linear"),
                "depths": ctrl.get("depths", [1, 4, 8]), "per_depth": c_pd}

    estimate(depths=args.depths or cfg.get("depths", [1, 2, 4, 6, 8]),
             per_depth=per_depth,
             seeds=args.seeds if args.seeds is not None else cfg.get("seeds", 2),
             max_retries=(args.max_retries if args.max_retries is not None
                          else cfg.get("max_retries", 1)),
             models=cfg.get("models", []),
             distractor_level=(args.distractors if args.distractors is not None
                               else cfg.get("distractor_level", 1)),
             days=args.days, headroom=args.headroom,
             variant=cfg.get("task_variant", "routing"), control=ctrl)


if __name__ == "__main__":
    main()
