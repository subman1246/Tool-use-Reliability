"""Executes a parsed tool call against a task and returns a tool result.

The executor is where the syntactic-versus-semantic distinction is made
operational. A call that cannot be parsed or violates the schema is a
syntactic failure and can be surfaced as a structured exception. A call that
names a valid tool with well-formed arguments always executes and returns a
number, even when that tool or argument is wrong, which is the silent
semantic case.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from tur.tasks.dag import Task


class FeedbackMode(str, Enum):
    STRUCTURED = "structured"
    OPAQUE = "opaque"


@dataclass
class ExecResult:
    ok: bool                 # did the call execute (schema-valid, known tool)
    output: int | None       # tool return value if executed
    feedback: str            # message handed back to the model
    schema_valid: bool
    known_tool: bool


def _validate(task: Task, tool_name: str, args: dict[str, Any]) -> tuple[bool, bool, str]:
    """Return (known_tool, schema_valid, reason)."""
    spec = task.tool_by_name.get(tool_name)
    if spec is None:
        return False, False, f"unknown tool '{tool_name}'"
    required = [p["name"] for p in spec.params if p["required"]]
    for r in required:
        if r not in args:
            return True, False, f"missing required parameter '{r}'"
    for p in spec.params:
        if p["name"] in args and p["type"] == "integer":
            v = args[p["name"]]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return True, False, (f"parameter '{p['name']}' expected integer, "
                                     f"got {type(v).__name__}")
    return True, True, "ok"


def execute(task: Task, tool_name: str, args: dict[str, Any],
            feedback: FeedbackMode) -> ExecResult:
    known, valid, reason = _validate(task, tool_name, args)
    if not known or not valid:
        # syntactic failure path: the executor can name the problem
        if feedback is FeedbackMode.STRUCTURED:
            msg = f"ToolError: {reason}"
        else:
            msg = "Error: the tool call could not be completed."
        return ExecResult(ok=False, output=None, feedback=msg,
                          schema_valid=valid, known_tool=known)
    spec = task.tool_by_name[tool_name]
    out = spec.run(args)
    # semantic errors execute normally: the feedback is just the returned value
    return ExecResult(ok=True, output=out, feedback=f"result: {out}",
                      schema_valid=True, known_tool=True)
