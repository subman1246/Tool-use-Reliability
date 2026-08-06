"""Per-call scoring against a gold step.

Correct invocation = correct tool selection AND correct arguments. Arguments
are scored strictly (exact) and softly (numeric tolerance, normalised strings)
and both are reported. Errors are bucketed into two families that map onto the
two recovery channels in the model:

  syntactic : the call could not be parsed, or violated the schema.
  semantic  : the call parsed and was schema-valid but named the wrong tool
              or carried a wrong value; it executes and returns junk silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from tur.tasks.dag import Step


class ErrorType(str, Enum):
    NONE = "none"
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"


@dataclass
class ParsedCall:
    tool: str | None
    args: dict[str, Any] | None
    parse_ok: bool
    raw: str = ""


@dataclass
class StepScore:
    selection_correct: bool        # matches the CORRECT tool given the ref held
    selection_matches_gold: bool   # matches the gold trajectory's tool
    args_correct_strict: bool
    args_correct_soft: bool
    schema_valid: bool
    error_type: ErrorType

    @property
    def correct(self) -> bool:
        """Globally correct: on the gold trajectory, with gold arguments."""
        return self.selection_matches_gold and self.args_correct_strict


def _soft_equal(a: Any, b: Any, rtol: float = 1e-9) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        denom = max(1.0, abs(b))
        return abs(a - b) / denom <= rtol
    return str(a).strip().casefold() == str(b).strip().casefold()


def _args_match(pred: dict[str, Any], gold: dict[str, Any], soft: bool) -> bool:
    if set(pred.keys()) != set(gold.keys()):
        return False
    for k, gv in gold.items():
        pv = pred[k]
        if soft:
            if not _soft_equal(pv, gv):
                return False
        else:
            if pv != gv:
                return False
    return True


def score_step(call: ParsedCall, gold: Step, schema_valid: bool,
               known_tool: bool, expected_tool: str | None = None) -> StepScore:
    """Score one call.

    expected_tool is the tool that is CORRECT given the ref the agent actually
    held. For linear tasks it equals the gold tool. For routing tasks under a
    poisoned context it can differ, and scoring against gold alone would blame
    the selection channel for an argument-propagation failure. Both notions are
    recorded: selection_correct (conditional) and selection_matches_gold.
    """
    if not call.parse_ok or call.tool is None or call.args is None:
        return StepScore(False, False, False, False, False, ErrorType.SYNTACTIC)
    if not known_tool or not schema_valid:
        return StepScore(False, False, False, False, schema_valid, ErrorType.SYNTACTIC)

    matches_gold = call.tool == gold.tool
    target = expected_tool if expected_tool is not None else gold.tool
    selection_ok = call.tool == target

    strict_ok = matches_gold and _args_match(call.args, gold.args, soft=False)
    soft_ok = matches_gold and _args_match(call.args, gold.args, soft=True)

    etype = ErrorType.NONE if strict_ok else ErrorType.SEMANTIC
    return StepScore(selection_ok, matches_gold, strict_ok, soft_ok, True, etype)
