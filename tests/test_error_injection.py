"""Controlled error injection: the intervention must be surgical, and inert on the scorer.

Why it exists. Injection is the only arm in this project that makes a CAUSAL claim --
that corrupting the context degrades the next call -- and the claim is only as good as
the construction. Two ways it could be quietly wrong:

  1. the two members of a pair differ somewhere other than the injected value, in which
     case the measured effect includes whatever else drifted, and
  2. the corrupted member is scored against an answer that is no longer correct, in
     which case the "effect" is an artefact of the scorer and would appear at full
     strength against a model that behaves perfectly.

The second is the dangerous one, because it produces a large, clean, entirely spurious
effect. Under a corrupted ref the routing rule selects a DIFFERENT tool, so gold is the
wrong yardstick; a perfect agent that applies the rule to the value it actually holds
must score 1.000 on the corrupted branch too. That is what test_perfect_policy_is_inert
pins down, and it is the test that would catch the injection measuring nothing but its
own substitution.

Asserted here:

  1. a perfect conditional-on-state policy scores 1.000 on BOTH members of every pair,
     so any effect measured later is a property of the model, not of the scoring
  2. the corrupted prompt differs from the clean one in EXACTLY ONE message, and that
     message is the result line for step inject_at-1
  3. parity_flip changes which tool is correct; parity_preserving does not, while still
     changing the value -- otherwise the two modes would not separate the channels they
     are meant to separate
  4. the injected value is deterministic across processes (crc32, not hash())
  5. the guard rails reject inject_at=0, inject_at>=depth, and linear tasks
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tur.harness.executor import FeedbackMode                          # noqa: E402
from tur.harness.runner import (MockBackend, run_injection_pair,       # noqa: E402
                                _corrupt_value, _expected_args_given_held,
                                _expected_tool_given_ref)
from tur.tasks.dag import (generate_routing_suite, generate_suite,     # noqa: E402
                           generate_routing_task, MOD)

DEPTH = 6
N = 12
SEED = 4242


def perfect(task, step, ref, attempt):
    """Applies the routing rule to the value actually held -- not to gold."""
    return (_expected_tool_given_ref(task, step, ref),
            _expected_args_given_held(task, step, ref), True)


def _tasks(n=N, depth=DEPTH):
    return generate_routing_suite([depth], n, 1, base_seed=SEED)


def test_perfect_policy_is_inert() -> None:
    """The scorer must not manufacture an effect against a perfect agent."""
    n_pairs = 0
    for mode in ("parity_flip", "parity_preserving"):
        for task in _tasks():
            for j in (1, 3, 5):
                recs = run_injection_pair(task, MockBackend(perfect), j, mode,
                                          "uniform", FeedbackMode.STRUCTURED, 1)
                assert len(recs) == 2
                clean, corrupt = recs
                assert clean.run_mode == "inj_clean"
                assert corrupt.run_mode == "inj_corrupt"
                for r in recs:
                    assert r.correct_given_state, (
                        f"{mode} j={j} {r.run_mode}: a perfect conditional-on-state "
                        f"policy scored INCORRECT. The corrupted branch is being "
                        f"scored against an answer that is not correct given the "
                        f"value held, so any severity measured here would be a "
                        f"scorer artefact, not a model effect.")
                    assert r.selection_correct and r.args_correct_given_state
                assert clean.pair_id == corrupt.pair_id
                n_pairs += 1
    print(f"  ok  perfect policy scores 1.000 on both members of {n_pairs} pairs")


def test_prompt_differs_in_exactly_one_message() -> None:
    """The intervention is surgical: one substituted value, nothing else."""
    checked = 0
    for mode in ("parity_flip", "parity_preserving"):
        for task in _tasks(n=6):
            for j in (1, 3, 5):
                seen: list[list[dict]] = []
                backend = MockBackend(perfect)
                inner = backend.complete

                def complete(messages, tools, m, _s=seen, _i=inner):
                    # snapshot: the runner mutates the list across retries
                    _s.append([dict(x) for x in messages])
                    return _i(messages, tools, m)

                backend.complete = complete
                run_injection_pair(task, backend, j, mode, "uniform",
                                   FeedbackMode.STRUCTURED, 1)
                assert len(seen) == 2, "expected one call per member"
                a, b = seen
                assert len(a) == len(b), (
                    f"{mode} j={j}: prompts have different message COUNTS "
                    f"({len(a)} vs {len(b)}); the prefix is not identical")
                diffs = [i for i, (x, y) in enumerate(zip(a, b))
                         if x.get("content") != y.get("content")]
                assert len(diffs) == 1, (
                    f"{mode} j={j}: prompts differ in {len(diffs)} messages "
                    f"{diffs}, expected exactly 1. Anything beyond the injected "
                    f"value confounds the causal contrast.")
                i = diffs[0]
                true_val = task.gold[j - 1].output
                assert a[i]["content"] == f"result: {true_val}", (
                    f"the differing message is {a[i]['content']!r}, not the result "
                    f"line for step {j - 1}")
                assert b[i]["content"].startswith("result: ")
                assert b[i]["content"] != a[i]["content"]
                checked += 1
    print(f"  ok  exactly one message differs, and it is the step j-1 result line "
          f"({checked} pairs)")


def test_modes_load_different_channels() -> None:
    """parity_flip must change the correct tool; parity_preserving must not."""
    flips = same = 0
    for task in _tasks(n=25):
        for j in (1, 3, 5):
            true_val = task.gold[j - 1].output
            gold_tool = _expected_tool_given_ref(task, j, true_val)

            bad = _corrupt_value(true_val, "parity_flip", 7)
            assert bad % 2 != true_val % 2, "parity_flip did not flip parity"
            assert _expected_tool_given_ref(task, j, bad) != gold_tool, (
                "parity_flip left the correct tool unchanged, so it does not "
                "load the selection channel")
            flips += 1

            bad = _corrupt_value(true_val, "parity_preserving", 7)
            assert bad % 2 == true_val % 2, "parity_preserving flipped parity"
            assert bad != true_val, (
                "parity_preserving returned the ORIGINAL value: the 'corrupted' "
                "member would be identical to the clean one and the pair would "
                "measure nothing")
            assert _expected_tool_given_ref(task, j, bad) == gold_tool
            same += 1
    print(f"  ok  parity_flip changes the correct tool ({flips} cases); "
          f"parity_preserving changes only the argument ({same} cases)")


def test_injected_value_is_deterministic() -> None:
    """Reproducible across processes -- crc32, not PYTHONHASHSEED-randomised hash()."""
    snippet = (
        "import sys; sys.path.insert(0, r'%s');"
        "from tur.tasks.dag import generate_routing_task;"
        "from tur.harness.runner import _corrupt_value;"
        "import zlib;"
        "t = generate_routing_task('r6_0_s%d', 6, 1, 99);"
        "v = t.gold[2].output;"
        "s = zlib.crc32(('r6_0_s%d|3|parity_flip').encode());"
        "print(_corrupt_value(v, 'parity_flip', s))"
    ) % (str(ROOT / "src"), SEED, SEED)
    outs = set()
    for seed_env in ("0", "1", "12345"):
        r = subprocess.run([sys.executable, "-c", snippet], capture_output=True,
                           text=True, env={**__import__("os").environ,
                                           "PYTHONHASHSEED": seed_env})
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout.strip())
    assert len(outs) == 1, (
        f"injected value varied across PYTHONHASHSEED settings: {outs}. It must be "
        f"reproducible, or a replay injects a different error than the original run.")
    print(f"  ok  injected value stable across PYTHONHASHSEED (= {outs.pop()})")


def test_guard_rails() -> None:
    task = _tasks(n=1)[0]
    for bad_j in (0, -1, DEPTH, DEPTH + 1):
        try:
            run_injection_pair(task, MockBackend(perfect), bad_j)
        except ValueError:
            pass
        else:
            raise AssertionError(f"inject_at={bad_j} was accepted; step 0 has no "
                                 f"previous result and >=depth has no step to score")
    try:
        run_injection_pair(generate_suite([DEPTH], 1, 1)[0],
                           MockBackend(perfect), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("a LINEAR task was accepted: corrupting a value there "
                             "cannot change which tool is correct, so the selection "
                             "channel does not exist and the arm is meaningless")
    try:
        run_injection_pair(task, MockBackend(perfect), 2, "not_a_mode")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown corruption mode was accepted")
    print("  ok  guard rails reject step 0, out-of-range, linear tasks, bad modes")


def main() -> None:
    test_perfect_policy_is_inert()
    test_prompt_differs_in_exactly_one_message()
    test_modes_load_different_channels()
    test_injected_value_is_deterministic()
    test_guard_rails()
    print("\nerror injection is surgical and inert with respect to the scorer")


if __name__ == "__main__":
    main()
