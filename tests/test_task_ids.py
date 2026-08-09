"""Task ids must be unique across generator seeds.

They were not: ids were f"d{depth}_{k}" / f"r{depth}_{k}", with no reference to
base_seed, so every seed emitted the identical id set. Nothing raised. What
happened instead is that every statistic keyed on task_id quietly merged two
different tasks -- measure_recovery read transitions across the join between two
unrelated trajectories, delta_1's per-id dict let the later seed overwrite the
earlier one outright, and the paired bootstrap resampled pairs instead of single
tasks. A configured r_syn of 0.35 measured 0.10 at seeds=2 and 0.34 at seeds=1,
which is the only reason it was noticed at all.

Also asserted here: the id is what distinguishes the suites, so two seeds must
produce genuinely different TASKS, not just different labels on the same ones.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tur.tasks.dag import generate_routing_suite, generate_suite   # noqa: E402

DEPTHS = [1, 2, 4]
N = 6


def check(gen, label: str) -> None:
    a = gen(DEPTHS, N, 1, base_seed=1000)
    b = gen(DEPTHS, N, 1, base_seed=1031)

    ids_a = [t.task_id for t in a]
    ids_b = [t.task_id for t in b]
    assert len(set(ids_a)) == len(ids_a), \
        f"{label}: ids collide WITHIN one seed: {len(ids_a) - len(set(ids_a))} dupes"
    overlap = set(ids_a) & set(ids_b)
    assert not overlap, (f"{label}: {len(overlap)} ids shared across generator "
                         f"seeds, e.g. {sorted(overlap)[:3]}")

    # and the tasks themselves must differ, or unique ids would just be
    # relabelling the same suite
    gold_a = {t.task_id.rsplit("_s", 1)[0]: [s.output for s in t.gold] for t in a}
    gold_b = {t.task_id.rsplit("_s", 1)[0]: [s.output for s in t.gold] for t in b}
    shared = set(gold_a) & set(gold_b)
    assert shared, f"{label}: id scheme changed shape; cannot compare suites"
    identical = [k for k in shared if gold_a[k] == gold_b[k]]
    assert not identical, (f"{label}: {len(identical)} tasks are identical across "
                          f"seeds despite distinct ids")
    print(f"  ok  {label}: {len(ids_a)} ids/seed, no overlap, all trajectories "
          f"differ across seeds")


def main() -> None:
    check(generate_suite, "linear")
    check(generate_routing_suite, "routing")
    print("\ntask ids are unique across generator seeds")


if __name__ == "__main__":
    main()
