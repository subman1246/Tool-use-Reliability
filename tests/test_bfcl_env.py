"""The vendored BFCL environment must agree with the BFCL data it is scored against.

Why it exists. Every real-benchmark number depends on being able to execute a task's gold
trajectory and get the state the benchmark's authors intended. If the vendored API classes
and the dataset come from different BFCL versions, some gold trajectories raise, and the
natural response -- curate to the tasks that happen to run -- silently selects the subset
on OUR bug rather than on any property of the tasks.

That is not hypothetical. It is what happened: the HuggingFace copy of the dataset serves
BFCL v3, the repository's source code is v4, and pairing them replayed 147 of 200 gold
trajectories, with 41 of the 53 failures a single changed signature
(TravelAPI.book_flight). Curating around that would have dropped a quarter of the
benchmark and called the remainder a subset. With v4 data against v4 source, all 200
replay.

So the test is the invariant, not the number: EVERY gold trajectory in the split must
execute cleanly. A single failure means the environment and the data have drifted apart
again, and the right fix is to re-pin them, not to filter.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tur.tasks.bfcl import CLASSES, instantiate, parse_call, DATA, ANSWERS  # noqa: E402


def test_every_gold_trajectory_replays() -> None:
    entries = DATA()
    answers = ANSWERS()
    assert entries, "no BFCL entries loaded"

    failures = []
    depths = {}
    for e in entries:
        calls = [c for turn in answers[e["id"]] for c in turn]
        try:
            insts = instantiate(e)
            for c in calls:
                name, pos, kw = parse_call(c)
                fn = next((getattr(i, name) for i in insts.values()
                           if hasattr(i, name)), None)
                if fn is None:
                    raise AttributeError("no involved class exposes " + name)
                fn(*pos, **kw)
        except Exception as ex:                      # noqa: BLE001
            failures.append((e["id"], type(ex).__name__, str(ex)[:80]))
            continue
        depths[e["id"]] = len(calls)

    assert not failures, (
        "%d of %d gold trajectories did not replay against the vendored environment. "
        "The environment and the dataset have drifted apart -- re-pin them to the same "
        "BFCL version rather than curating to whatever still runs, which would select "
        "the subset on this bug. First few: %s"
        % (len(failures), len(entries), failures[:3]))
    print("  ok  all %d gold trajectories replay against the vendored environment"
          % len(entries))
    return depths


def test_the_subset_is_deep_enough_to_be_worth_running() -> None:
    depths = test_every_gold_trajectory_replays()
    deep = [k for k, v in depths.items() if v >= 6]
    assert len(deep) >= 28, (
        "only %d tasks reach depth 6; the routing arm's depth-6 cell for this model has "
        "n=28, and a real-benchmark cell smaller than that cannot be compared to it"
        % len(deep))
    print("  ok  %d tasks at depth >= 6 (routing arm's depth-6 cell is n=28)"
          % len(deep))


def test_classes_cover_the_split() -> None:
    needed = {c for e in DATA() for c in e["involved_classes"]}
    missing = needed - set(CLASSES)
    assert not missing, (
        "the split involves API classes that are not vendored: %s. Tasks using them "
        "would be excluded for an infrastructure reason, not a task property."
        % sorted(missing))
    print("  ok  every involved class is vendored (%d classes)" % len(needed))


def main() -> None:
    test_every_gold_trajectory_replays()
    test_the_subset_is_deep_enough_to_be_worth_running()
    test_classes_cover_the_split()
    print("\nvendored BFCL environment is version-consistent with its dataset")


if __name__ == "__main__":
    main()
