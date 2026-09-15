"""Durable trajectory state with resume-on-reconnect, for long unattended runs.

WHY

A Colab session dies: the tab closes, the runtime is reclaimed, the 12-hour cap hits. A
sweep that holds everything in memory loses all of it. This project has already paid that
price once in a different form -- run_real_suite printed that raw rows were written and
then exited before writing them, losing an entire cap-stopped injection run that had to
be reconstructed from the response cache by hand.

WHERE THE CHECKPOINT LIVES: THE BACKEND, NOT THE RUNNER

The obvious design is a resumable copy of run_free that can start from a saved step
prefix. That is the wrong place. run_free, run_teacher_forced and run_injection_pair each
build their prompts in a specific way, and a second implementation of that assembly is
exactly the duplication that made estimate_cost silently understate the sweep until it
was rewritten to drive the real loops. Any resume logic living in a forked runner would
drift the same way.

Instead the checkpoint wraps the BACKEND. The backend sees every call of every task, in
order, and nothing else in the harness needs to change. On resume the runner re-enters
the task and re-issues its calls; the wrapper serves the already-completed ones from the
checkpoint, so they cost nothing and return byte-identical results, and real generation
resumes exactly at the first step that never completed. No step is recomputed by the
model and none is skipped.

This is the same replay-from-cache pattern already trusted in this project to recover the
injection results, with one addition: the checkpoint records explicit per-task step
progress, so a test can assert the resume point rather than infer it from cost.

WHAT IS PERSISTED

One JSON file per (tag, model, task), flushed after every single call:

    {"task_id":..., "model":..., "tag":..., "steps":[{...}, ...], "done": bool}

Flushing every call rather than every task is the point -- a task interrupted at step 5
of 8 keeps its first five. Writes go through a temp file and os.replace so a process
killed mid-write leaves the previous good file rather than a truncated one.

SINGLE ACCOUNT ONLY. No cross-account sharding, by explicit instruction.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _safe(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


class TrajectoryStore:
    """Per-task durable state under a directory (a mounted Drive path on Colab)."""

    def __init__(self, root: str | Path, tag: str, model: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.model = model

    def path_for(self, task_id: str) -> Path:
        return self.root / ("%s_%s_%s.json" % (_safe(self.tag), _safe(self.model),
                                               _safe(task_id)))

    def load(self, task_id: str) -> dict:
        p = self.path_for(task_id)
        if not p.exists():
            return {"task_id": task_id, "model": self.model, "tag": self.tag,
                    "steps": [], "done": False}
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            # A corrupt checkpoint is treated as absent rather than fatal: the task
            # is simply recomputed. Crashing here would make one bad file block a
            # whole resumed sweep.
            return {"task_id": task_id, "model": self.model, "tag": self.tag,
                    "steps": [], "done": False}

    def save(self, state: dict) -> None:
        """Atomic write: temp file in the same directory, then os.replace."""
        p = self.path_for(state["task_id"])
        fd, tmp = tempfile.mkstemp(dir=str(self.root), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(state, fh)
            os.replace(tmp, p)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def completed_steps(self, task_id: str) -> int:
        return len(self.load(task_id).get("steps", []))

    def is_done(self, task_id: str) -> bool:
        return bool(self.load(task_id).get("done"))

    def mark_done(self, task_id: str) -> None:
        st = self.load(task_id)
        st["done"] = True
        self.save(st)

    def resume_summary(self) -> dict:
        """What a restarting process finds on disk: done / partial / total calls."""
        done, partial, calls = [], {}, 0
        for p in sorted(self.root.glob("%s_%s_*.json" % (_safe(self.tag),
                                                         _safe(self.model)))):
            try:
                st = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            calls += len(st.get("steps", []))
            if st.get("done"):
                done.append(st["task_id"])
            elif st.get("steps"):
                partial[st["task_id"]] = len(st["steps"])
        return {"done": done, "partial": partial, "recorded_calls": calls}


class CheckpointingBackend:
    """Wraps any backend with the complete(messages, tools, mode) contract.

    Serves calls already recorded for the current task from the checkpoint, and flushes
    each newly generated call before returning it. `begin_task` must be called before a
    task's calls; the harness's runners do not know about checkpointing, so the sweep
    driver is responsible for that one line.
    """

    def __init__(self, inner, store: TrajectoryStore):
        self.inner = inner
        self.store = store
        self._task_id: str | None = None
        self._state: dict | None = None
        self._cursor = 0
        # counted separately from inner.n_calls: inner never sees a replayed call
        self.n_replayed = 0
        self.n_generated = 0

    # ------------------------------------------------------------------ task

    def begin_task(self, task_id: str) -> int:
        """Point the wrapper at a task. Returns how many calls are already done."""
        self._task_id = task_id
        self._state = self.store.load(task_id)
        self._cursor = 0
        return len(self._state.get("steps", []))

    def end_task(self) -> None:
        if self._task_id is not None:
            self.store.mark_done(self._task_id)
        self._task_id = None
        self._state = None
        self._cursor = 0

    # -------------------------------------------------------------- contract

    def complete(self, messages, tools, mode):
        if self._state is None:
            # Un-checkpointed call: pass through rather than guess a task id. Silently
            # inventing one would scatter orphan files that resume_summary then counts.
            return self.inner.complete(messages, tools, mode)

        steps = self._state["steps"]
        if self._cursor < len(steps):
            # Already completed before the interruption: replay, do not regenerate.
            rec = steps[self._cursor]
            self._cursor += 1
            self.n_replayed += 1
            return rec["response"]

        result = self.inner.complete(messages, tools, mode)
        steps.append({"i": self._cursor, "response": result})
        self._cursor += 1
        self.n_generated += 1
        self.store.save(self._state)          # flush after EVERY call
        return result

    # ----------------------------------------------------------------- stats

    def __getattr__(self, name: str) -> Any:
        # Counters and stats() live on the wrapped backend; forward anything this
        # wrapper does not define itself so it stays a transparent drop-in.
        return getattr(self.__dict__["inner"], name)

    def stats(self) -> dict:
        s = dict(self.inner.stats())
        s.update({"n_replayed": self.n_replayed, "n_generated": self.n_generated})
        return s
