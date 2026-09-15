"""Recovery-enabled routing task: a task in which returning to the goal is POSSIBLE.

THE GAP THIS CLOSES

Section 7 of the paper says, in its own words, that no task design currently gives the
agent any path back to the original goal after a divergence. That is not a small
omission: it means "recovery" has never been measurable in this project, only assumed
absent. Every recovery rate reported so far is either a scorer artifact (canonical gold
agreement pins it at zero, because after divergence the gold value is unreachable) or a
different quantity wearing the same word (conditional-on-state scoring measures whether
the agent continues COMPETENTLY from a wrong state, which is not the same as getting back
to the right one).

With no route back in the task, those two are indistinguishable. This variant adds the
route, so they can be told apart.

THE MECHANISM

One extra tool, `check_state`, which returns the true canonical value for the current
step. Two properties make it a real test rather than a give-away:

  it costs a call     using it consumes a backend call and does not advance the task, so
                      an agent that spams it pays for it in the call budget. The step is
                      then re-attempted with the correct value in hand.

  it is one-shot      usable once per task. This is the cleanest of the options: a budget
                      of one makes "did the agent use it" binary, so usage is a clean
                      covariate rather than a count that needs its own model, and it
                      removes the degenerate policy of resyncing before every step, which
                      would convert the free-running arm into the teacher-forced arm and
                      measure nothing.

The property that matters for the science: after a successful resync the agent holds the
canonical value again, so it can rejoin the GOLD trajectory and finish the ORIGINAL task.
Without that, a repair tool would only let the agent continue tidily from a wrong state,
which is the thing we already cannot distinguish.

WHAT IS DELIBERATELY NOT DONE

`check_state` is not added to tool_by_name and therefore never reaches executor.execute.
Its return value depends on run-state (which step the agent is on), not on its arguments,
and ToolSpec.run is a pure function of args. Threading run-state into the executor to
support one tool would put mutable state into the one component that is currently a pure
validator. The run loop intercepts the call instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tur.tasks.dag import RoutingTask

RESYNC_TOOL = "check_state"


@dataclass
class RecoveryRoutingTask(RoutingTask):
    """A RoutingTask whose prompt also offers a one-shot state-repair tool."""

    resync_budget: int = 1

    def schema_view(self) -> list[dict[str, Any]]:
        base = super().schema_view()
        base.append({
            "name": RESYNC_TOOL,
            "description": (
                "Returns the correct current ref value for the step you are on. "
                "Use it if you believe the value you are carrying may be wrong. "
                "It does not perform a step: after calling it you must still make "
                "the step's tool call."),
            "parameters": {},
        })
        return base

    def recovery_rule_text(self) -> str:
        return (
            f"You may call '{RESYNC_TOOL}' at most {self.resync_budget} time(s) in this "
            f"task. It returns the correct ref for the current step and does not count "
            f"as performing that step, so you must still make the step's own tool call "
            f"afterwards.")


def to_recovery_task(task: RoutingTask, resync_budget: int = 1) -> RecoveryRoutingTask:
    """Clone a RoutingTask into its recovery-enabled counterpart.

    Built by copying fields rather than regenerating from a seed so the gold trajectory,
    branch structure and seed value are IDENTICAL to the plain routing task with the same
    id. That is what makes the recovery arm comparable to the existing free-running arm
    task-for-task instead of merely at the same depth.
    """
    return RecoveryRoutingTask(
        task_id=task.task_id,
        depth=task.depth,
        branches=task.branches,
        gold=task.gold,
        seed_value=task.seed_value,
        distractor_level=task.distractor_level,
        arg_shift=task.arg_shift,
        present_odd_first=list(task.present_odd_first),
        resync_budget=resync_budget,
    )


def canonical_ref_at(task: RoutingTask, step: int) -> int:
    """The value a correct agent would be carrying INTO `step`.

    This is what check_state returns, and it is also the yardstick for whether a
    trajectory has diverged. Defined off the gold trajectory, so it exists whether or not
    the agent is anywhere near it.
    """
    return task.seed_value if step == 0 else task.gold[step - 1].output
