"""Loaders for the real, primary benchmarks (BFCL and tau-bench).

The synthetic DAG tasks are integer chains with a built-in executor. Real
benchmarks bring their own tools and gold calls, so they are mapped into a
neutral NormTask that the scorer can consume directly (score_step compares
tool name and argument dicts and is source-agnostic). Executing real tools is
the benchmark's own simulator and lives outside this module.

These are scaffolds: the exact field names differ across benchmark versions,
so the mapping points are marked. Fetching the data needs network and is left
to the run environment, not hardcoded here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NormStep:
    tool: str
    args: dict[str, Any]


@dataclass
class NormTask:
    task_id: str
    source: str
    depth: int
    tools_schema: list[dict[str, Any]]
    gold_steps: list[NormStep]
    prompt: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def _depth_of(steps: list[NormStep]) -> int:
    return len(steps)


def load_bfcl(path: str | Path, category_filter: tuple[str, ...] =
              ("multi_turn", "multi_step")) -> list[NormTask]:
    """Parse a BFCL jsonl export into NormTasks.

    BFCL entries typically carry a `function` list (the tool schemas) and a
    ground-truth answer with one or more calls. Multi-turn and multi-step
    splits are the ones with real dependency depth. Adjust the field names to
    the BFCL version in use where marked TODO.
    """
    tasks: list[NormTask] = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        cat = row.get("category") or row.get("test_category") or ""
        if category_filter and not any(c in cat for c in category_filter):
            continue
        # TODO: map to the exact BFCL schema field for available functions
        tools = row.get("function") or row.get("tools") or []
        schema = [{"name": t.get("name"), "description": t.get("description", ""),
                   "parameters": t.get("parameters", {})} for t in tools]
        # TODO: BFCL ground truth may be nested per turn; flatten to a call list
        gold_raw = row.get("ground_truth") or row.get("answer") or []
        gold = [NormStep(tool=c.get("name") or c.get("tool"),
                         args=c.get("arguments") or c.get("args") or {})
                for c in _iter_calls(gold_raw)]
        tasks.append(NormTask(task_id=f"bfcl_{i}", source="bfcl",
                              depth=_depth_of(gold), tools_schema=schema,
                              gold_steps=gold, prompt=row.get("question", ""),
                              meta={"category": cat}))
    return tasks


def load_tau_bench(path: str | Path) -> list[NormTask]:
    """Parse a tau-bench task dump into NormTasks.

    tau-bench grades final environment state rather than per-call gold, so a
    gold call trace is not always provided. Where a reference trajectory exists
    (annotation or a strong reference policy rollout), map it here; otherwise
    these tasks support end-to-end success only and are excluded from the
    invocation-level fit. Marked TODO accordingly.
    """
    data = json.loads(Path(path).read_text())
    tasks: list[NormTask] = []
    for i, row in enumerate(data):
        schema = row.get("tools", [])
        # TODO: attach a reference trajectory if available for this task
        gold_raw = row.get("reference_trajectory", [])
        gold = [NormStep(tool=c.get("name"), args=c.get("arguments", {}))
                for c in _iter_calls(gold_raw)]
        tasks.append(NormTask(task_id=f"tau_{i}", source="tau_bench",
                              depth=_depth_of(gold), tools_schema=schema,
                              gold_steps=gold, prompt=row.get("instruction", ""),
                              meta={"domain": row.get("domain", "")}))
    return tasks


def _iter_calls(raw: Any):
    """Yield {name/tool, arguments/args} dicts from the varied gold formats."""
    if isinstance(raw, dict):
        yield raw
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                yield item
            elif isinstance(item, list):
                yield from _iter_calls(item)
