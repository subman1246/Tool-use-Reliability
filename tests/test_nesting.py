"""The nested unequal-n design rests on one property: asking a suite generator
for fewer tasks at a depth must return exactly a PREFIX of what asking for more
returns, same ids and same tasks.

If that fails, models with different n are running different tasks, every
cross-model contrast silently stops being paired, and the paired bootstrap is
resampling ids that do not mean the same thing in both arms. Nothing would raise;
the numbers would just quietly stop being comparable. Hence a test.

Also checked: the budget planner produces allocations that are themselves nested
across models (richer model >= poorer model at every depth), since that is what
makes the common prefix well-defined for the whole suite rather than pairwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tur.harness.budget import plan_suite                            # noqa: E402
from tur.tasks.dag import generate_routing_suite, generate_suite     # noqa: E402

DEPTHS = [1, 2, 4]
BIG = {1: 12, 2: 10, 4: 8}
SMALL = {1: 5, 2: 4, 4: 3}


def signature(task):
    """Everything about a task that the run depends on."""
    return (task.task_id, task.depth, task.seed_value,
            tuple((s.tool, tuple(sorted(s.args.items())), s.output)
                  for s in task.gold))


def check_prefix(gen, label: str) -> None:
    big = gen(DEPTHS, BIG, 1, base_seed=1000)
    small = gen(DEPTHS, SMALL, 1, base_seed=1000)

    by_depth_big: dict[int, list] = {}
    by_depth_small: dict[int, list] = {}
    for t in big:
        by_depth_big.setdefault(t.depth, []).append(t)
    for t in small:
        by_depth_small.setdefault(t.depth, []).append(t)

    for d in DEPTHS:
        b = [signature(t) for t in by_depth_big[d]]
        s = [signature(t) for t in by_depth_small[d]]
        assert len(s) == SMALL[d], f"{label} d{d}: got {len(s)} tasks, want {SMALL[d]}"
        assert len(b) == BIG[d], f"{label} d{d}: got {len(b)} tasks, want {BIG[d]}"
        assert s == b[:len(s)], (
            f"{label} d{d}: the smaller allocation is NOT a prefix of the larger "
            f"one -- nested unequal n would be comparing different tasks.\n"
            f"  small[0]={s[0][0]} big[0]={b[0][0]}")
    print(f"  ok  {label}: smaller allocation is an exact prefix at every depth")


def check_plans_nested() -> None:
    models = [
        {"name": "rich", "tpd": 500_000},
        {"name": "mid", "tpd": 200_000},
        {"name": "poor", "tpd": 100_000},
    ]
    plans, _, _ = plan_suite(models, days=8, headroom=0.80,
                             ref_primary={1: 60, 2: 60, 4: 65, 6: 65, 8: 47},
                             ref_control={1: 20, 4: 20, 8: 5})
    by_name = {p.name: p for p in plans}
    order = ["rich", "mid", "poor"]
    for a, b in zip(order, order[1:]):
        pa, pb = by_name[a], by_name[b]
        assert pa.scale >= pb.scale, f"{a} scale {pa.scale} < {b} {pb.scale}"
        for d in pa.primary:
            assert pa.primary[d] >= pb.primary[d], (
                f"depth {d}: {a} runs {pa.primary[d]} but {b} runs "
                f"{pb.primary[d]} -- allocations are not nested across models")
    for p in plans:
        assert p.cost_tokens <= p.budget_tokens or p.notes, (
            f"{p.name}: allocation costs {p.cost_tokens:,} against a budget of "
            f"{p.budget_tokens:,} with no explanatory note")
    print("  ok  planner: allocations are nested across models and fit budgets")
    print("      " + "  ".join(f"{p.name}={sum(p.primary.values())}" for p in plans))


def main() -> None:
    check_prefix(generate_suite, "linear")
    check_prefix(generate_routing_suite, "routing")
    check_plans_nested()
    print("\nnested unequal-n design holds: prefixes are exact and plans nest")


if __name__ == "__main__":
    main()
