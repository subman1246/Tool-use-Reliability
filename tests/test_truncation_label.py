"""A truncated completion is a third category and must be labelled as one.

finish_reason == "length" means the provider cut the response at a token ceiling.
That produces output which LOOKS malformed, so without the label it lands in one of
two wrong buckets: scored as a syntactic error (blaming the model for a
configuration fault) or, if it happens to still parse, scored as a perfectly good
call. Neither is right, and the difference is invisible in the text alone.

Also asserted: absent finish_reason encodes as None (UNKNOWN), never False.
Completions cached before the field existed cannot say whether they were truncated,
and recording False there would assert a verification that never happened.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tur.harness.executor import FeedbackMode                      # noqa: E402
from tur.harness.runner import run_free, run_teacher_forced        # noqa: E402
from tur.tasks.dag import generate_routing_suite                   # noqa: E402


class Backend:
    """Returns a fixed finish_reason, or omits the field entirely."""

    model = "stub"

    def __init__(self, finish_reason, omit=False):
        self.finish_reason = finish_reason
        self.omit = omit

    def complete(self, messages, tools, mode):
        ctx = messages[-1]["_ctx"]
        task, step, ref = ctx["task"], ctx["step"], ctx["ref"]
        import json
        out = {"text": json.dumps({"tool": task.gold[step].tool,
                                   "args": {"ref": ref}})}
        if not self.omit:
            out["finish_reason"] = self.finish_reason
        return out


def run(backend):
    task = generate_routing_suite([2], 1, 1, base_seed=1000)[0]
    rows = [r.__dict__ for r in
            run_free(task, backend, "uniform", FeedbackMode.STRUCTURED, 1)]
    rows += [r.__dict__ for r in
             run_teacher_forced(task, backend, "uniform",
                                FeedbackMode.STRUCTURED, 1)]
    return rows


def main() -> None:
    rows = run(Backend("length"))
    assert rows and all(r["truncated"] is True for r in rows), \
        f"finish_reason 'length' must set truncated=True, got " \
        f"{[r['truncated'] for r in rows]}"
    print(f"  ok  finish_reason='length' -> truncated=True on all "
          f"{len(rows)} records (both arms)")

    rows = run(Backend("stop"))
    assert all(r["truncated"] is False for r in rows), \
        "finish_reason 'stop' must set truncated=False"
    print("  ok  finish_reason='stop'   -> truncated=False")

    rows = run(Backend(None, omit=True))
    assert all(r["truncated"] is None for r in rows), (
        "a completion with NO finish_reason must record truncated=None (unknown), "
        "never False -- False would assert a check that never ran")
    print("  ok  finish_reason absent   -> truncated=None (unknown, not False)")
    print("\ntruncation is labelled as its own category in both run modes")


if __name__ == "__main__":
    main()
