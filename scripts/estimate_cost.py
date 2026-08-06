"""Estimate the call volume and rough cost of a real run BEFORE spending any
API budget. Run this first in Claude Code, before scripts/run_real_suite.py.

This does not call any model. It counts exactly how many backend calls the
configured sweep will make (accounting for retries and both run modes) and
multiplies by rough per-1k-token prices, using approximate token counts from
the actual prompts the harness builds.
"""

from __future__ import annotations

import argparse

from tur.tasks.dag import generate_suite
from tur.harness.runner import _task_intro

# Rough, approximate $/1M tokens (input, output). Update before a real run --
# these are ballpark figures for budgeting only, not billing-accurate.
PRICE_PER_1M = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-haiku": (0.80, 4.00),
    "claude-sonnet": (3.00, 15.00),
    "groq/qwen2.5-7b-instruct": (0.05, 0.08),
    "groq/qwen2.5-32b-instruct": (0.29, 0.39),
    "groq/qwen2.5-72b-instruct": (0.59, 0.79),
    "groq/llama-3.1-8b-instruct": (0.05, 0.08),
    "default_open_weight": (0.20, 0.20),
    "default_proprietary": (1.00, 3.00),
}


def _price_for(model: str) -> tuple[float, float]:
    for key, price in PRICE_PER_1M.items():
        if key in model:
            return price
    return PRICE_PER_1M["default_proprietary" if "gpt" in model or "claude" in model
                        else "default_open_weight"]


def estimate(depths: list[int], per_depth: int, seeds: int, max_retries: int,
            models: list[dict], distractor_level: int = 1) -> None:
    suite = generate_suite(depths, per_depth, distractor_level)
    # rough token estimate: ~4 chars/token, prompt grows with depth and
    # schema size; output is short (one JSON call)
    total_prompt_chars = 0
    total_calls_per_seed = 0
    for task in suite:
        intro_len = len(_task_intro(task))
        # each step of the free run re-sends the growing message history;
        # teacher-forced does the same but rebuilt fresh each step
        for step in range(task.depth):
            # both run modes call once per step in the best case, up to
            # (max_retries+1) times if every attempt fails
            total_calls_per_seed += 2 * (max_retries + 1)
            total_prompt_chars += 2 * (intro_len + 200 * step)  # +history growth

    total_calls = total_calls_per_seed * seeds
    est_input_tokens = total_prompt_chars * seeds / 4
    est_output_tokens = total_calls * 40  # a short JSON call is ~40 tokens

    print(f"Sweep: depths={depths} per_depth={per_depth} seeds={seeds} "
         f"max_retries={max_retries} distractor_level={distractor_level}")
    print(f"Tasks per seed: {len(suite)}  |  Total tasks: {len(suite) * seeds}")
    print(f"Worst-case backend calls PER MODEL: {total_calls:,}")
    print(f"Approx input tokens PER MODEL: {est_input_tokens:,.0f}  "
         f"(worst case, assumes every retry fires)")
    print(f"Approx output tokens PER MODEL: {est_output_tokens:,.0f}\n")

    print(f"{'model':<32}{'est. cost (worst case)':>26}{'est. cost (typical, ~40%)':>28}")
    grand_total = 0.0
    for m in models:
        name = m["name"] if isinstance(m, dict) else m
        in_price, out_price = _price_for(name)
        worst = (est_input_tokens / 1e6) * in_price + (est_output_tokens / 1e6) * out_price
        typical = worst * 0.4  # most retries don't fire; this is a rough haircut
        grand_total += typical
        print(f"{name:<32}{'$' + format(worst, ',.2f'):>26}{'$' + format(typical, ',.2f'):>28}")
    print(f"\nEstimated TOTAL across all models (typical case): ${grand_total:,.2f}")
    print("\nThese are rough estimates for budgeting only. Actual cost depends on\n"
         "real retry rates, actual token counts, and current provider pricing.\n"
         "Update PRICE_PER_1M in this file with current prices before relying on this.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    ap.add_argument("--per-depth", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--distractors", type=int, default=1)
    args = ap.parse_args()

    import yaml
    try:
        cfg = yaml.safe_load(open("config/default.yaml"))
        models = cfg.get("models", [])
    except FileNotFoundError:
        models = [{"name": "gpt-4o-mini"}, {"name": "groq/qwen2.5-7b-instruct"}]

    estimate(args.depths, args.per_depth, args.seeds, args.max_retries,
             models, args.distractors)


if __name__ == "__main__":
    main()
