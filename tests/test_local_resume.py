"""Resume-on-reconnect must resume at the right step: no redone work, none skipped.

Why it exists. An unattended Colab run dies mid-sweep as a matter of course, and the
whole value of checkpointing is the claim that restarting neither wastes the work already
paid for nor silently drops a step. Both failure modes are quiet. Redoing work looks like
a slow run; skipping a step looks like a complete trajectory that is missing a call, and
nothing downstream would flag it, because a short trajectory is indistinguishable from a
task that ended early.

So the assertions are about the SPLIT between replayed and newly generated calls, not
just about the final record count:

  1. a clean run records every call and marks the task done
  2. a process killed mid-task keeps exactly the calls that completed
  3. on restart, calls before the kill point are REPLAYED (the model is never re-invoked
     for them) and calls after it are GENERATED -- asserted by counting invocations of
     the underlying backend, which is the only way to tell a cheap replay from an
     expensive redo
  4. replayed responses are byte-identical to the originals, so a resumed trajectory is
     the same trajectory
  5. an interrupted write leaves the previous good checkpoint, not a truncated file
  6. the real runner (run_free) resumes correctly through the wrapper, since the whole
     design rests on the runner needing no changes

A GPU is not required and neither is transformers: the wrapper is exercised against a
mock backend, which is the point of putting checkpointing in the backend layer.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tur.harness.checkpoint import (CheckpointingBackend,          # noqa: E402
                                    TrajectoryStore)
from tur.harness.executor import FeedbackMode                      # noqa: E402
from tur.harness.runner import MockBackend, run_free               # noqa: E402
from tur.harness.sim_policy import _gold_arg_for                   # noqa: E402
from tur.tasks.dag import generate_routing_suite                   # noqa: E402


class Killed(RuntimeError):
    pass


class CountingBackend:
    """Deterministic responses, and a count of how often it was actually invoked."""

    def __init__(self, kill_at: int | None = None):
        self.calls = 0
        self.kill_at = kill_at
        self.n_calls = 0
        self.n_cache_hits = 0

    def complete(self, messages, tools, mode):
        if self.kill_at is not None and self.calls >= self.kill_at:
            raise Killed("simulated process death")
        self.calls += 1
        self.n_calls = self.calls
        return {"text": json.dumps({"tool": "t%d" % self.calls,
                                    "args": {"ref": self.calls}}),
                "finish_reason": "stop"}

    def stats(self):
        return {"n_calls": self.calls}


def _store(root, tag="res", model="local/test"):
    return TrajectoryStore(root, tag, model)


def test_clean_run_records_everything() -> None:
    root = tempfile.mkdtemp()
    try:
        be = CheckpointingBackend(CountingBackend(), _store(root))
        be.begin_task("task-a")
        for _ in range(6):
            be.complete([{"role": "user", "content": "x"}], None, "uniform")
        be.end_task()
        st = _store(root).load("task-a")
        assert len(st["steps"]) == 6, st
        assert st["done"] is True
        assert be.n_generated == 6 and be.n_replayed == 0
        print("  ok  clean run: 6 calls recorded, task marked done")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_kill_midtask_keeps_completed_calls() -> None:
    root = tempfile.mkdtemp()
    try:
        inner = CountingBackend(kill_at=4)
        be = CheckpointingBackend(inner, _store(root))
        be.begin_task("task-b")
        done = 0
        try:
            for _ in range(8):
                be.complete([{"role": "user", "content": "x"}], None, "uniform")
                done += 1
        except Killed:
            pass
        assert done == 4, done
        st = _store(root).load("task-b")
        assert len(st["steps"]) == 4, "expected 4 flushed calls, got %d" % len(st["steps"])
        assert st["done"] is False, "an interrupted task must not be marked done"
        print("  ok  killed after 4 of 8 calls: exactly 4 survive, task not marked done")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_resume_replays_prefix_and_generates_only_the_rest() -> None:
    """The load-bearing test: no redone work, no skipped work."""
    root = tempfile.mkdtemp()
    try:
        first = CountingBackend(kill_at=4)
        be = CheckpointingBackend(first, _store(root))
        be.begin_task("task-c")
        originals = []
        try:
            for _ in range(8):
                originals.append(
                    be.complete([{"role": "user", "content": "x"}], None, "uniform"))
        except Killed:
            pass
        assert first.calls == 4

        # restart: fresh backend, fresh wrapper, same store
        second = CountingBackend()
        be2 = CheckpointingBackend(second, _store(root))
        already = be2.begin_task("task-c")
        assert already == 4, "resume point should be step 4, got %d" % already

        got = [be2.complete([{"role": "user", "content": "x"}], None, "uniform")
               for _ in range(8)]
        be2.end_task()

        assert be2.n_replayed == 4, ("expected 4 replayed calls, got %d -- the prefix "
                                     "was re-issued to the model" % be2.n_replayed)
        assert be2.n_generated == 4, ("expected 4 newly generated calls, got %d"
                                      % be2.n_generated)
        assert second.calls == 4, ("the underlying backend was invoked %d times on "
                                   "resume; it must only be invoked for the 4 steps "
                                   "that never completed" % second.calls)
        for i in range(4):
            assert got[i] == originals[i], (
                "replayed call %d differs from the original: a resumed trajectory "
                "must be the same trajectory" % i)
        st = _store(root).load("task-c")
        assert len(st["steps"]) == 8 and st["done"] is True
        print("  ok  resume: 4 replayed byte-identically, 4 generated, 8 recorded, "
              "backend invoked exactly 4 times")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_interrupted_write_leaves_previous_good_file() -> None:
    root = tempfile.mkdtemp()
    try:
        store = _store(root)
        be = CheckpointingBackend(CountingBackend(), store)
        be.begin_task("task-d")
        for _ in range(3):
            be.complete([{"role": "user", "content": "x"}], None, "uniform")
        good = json.loads(store.path_for("task-d").read_text())

        # a save that dies partway through must not corrupt what is on disk
        orig_dump = json.dump

        def exploding(obj, fh, *a, **k):
            fh.write('{"task_id": "task-d", "steps": [')   # truncated on purpose
            raise OSError("disk full")

        json.dump = exploding
        try:
            store.save({"task_id": "task-d", "model": "local/test", "tag": "res",
                        "steps": [1, 2, 3, 4], "done": False})
        except OSError:
            pass
        finally:
            json.dump = orig_dump

        after = json.loads(store.path_for("task-d").read_text())
        assert after == good, "a failed write clobbered a good checkpoint"
        leftovers = list(Path(root).glob("*.tmp"))
        assert not leftovers, "temp files left behind: %s" % leftovers
        print("  ok  interrupted write: previous checkpoint intact, no temp litter")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_resume_through_the_real_runner() -> None:
    """The design claims the runner needs no changes. Verify that against run_free."""
    root = tempfile.mkdtemp()
    try:
        task = generate_routing_suite([6], 1, 1, base_seed=777)[0]

        def perfect(t, step, ref, attempt):
            return t.gold[step].tool, {"ref": _gold_arg_for(t, step, ref)}, True

        class Counting(MockBackend):
            def __init__(self, policy, kill_at=None):
                super().__init__(policy)
                self.calls = 0
                self.kill_at = kill_at

            def complete(self, messages, tools, mode):
                if self.kill_at is not None and self.calls >= self.kill_at:
                    raise Killed("simulated death")
                self.calls += 1
                return super().complete(messages, tools, mode)

        inner = Counting(perfect, kill_at=3)
        be = CheckpointingBackend(inner, _store(root))
        be.begin_task(task.task_id)
        try:
            run_free(task, be, "uniform", FeedbackMode.STRUCTURED, 1)
        except Killed:
            pass
        assert inner.calls == 3

        inner2 = Counting(perfect)
        be2 = CheckpointingBackend(inner2, _store(root))
        assert be2.begin_task(task.task_id) == 3
        recs = run_free(task, be2, "uniform", FeedbackMode.STRUCTURED, 1)
        be2.end_task()

        assert len(recs) == 6, "resumed run produced %d step records, expected 6" % len(recs)
        assert be2.n_replayed == 3, be2.n_replayed
        assert inner2.calls == 3, ("model invoked %d times on resume; only the 3 "
                                   "incomplete steps should reach it" % inner2.calls)
        print("  ok  run_free resumes through the wrapper unmodified: 6 records, "
              "3 replayed, 3 generated")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    test_clean_run_records_everything()
    test_kill_midtask_keeps_completed_calls()
    test_resume_replays_prefix_and_generates_only_the_rest()
    test_interrupted_write_leaves_previous_good_file()
    test_resume_through_the_real_runner()
    print("\nresume-on-reconnect resumes at the correct step, redoing and skipping none")


if __name__ == "__main__":
    main()
