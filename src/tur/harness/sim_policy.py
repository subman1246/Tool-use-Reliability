"""A parametric mock 'model' that stands in for a real LLM during pipeline
validation. Unlike model.state.simulate() (which samples straight from the
analytic g_t formula), this drives the actual harness and scoring code: it
decides one call at a time from whatever ref it is handed, so context poisoning
happens because of what earlier steps in THIS run actually produced, not
because we told it to. That is what makes a run through this policy a real
test of the harness and scorer, not just a test of the state-model math.

Ground truth per policy (used later to check the fitted posteriors against a
known answer):
  p0, p_slope   clean baseline at depth 1 and its linear decline with depth
  pi            fraction of the clean rate lost while poisoned
  syntax_share  fraction of fresh errors that are syntactic vs semantic
  r_syn         chance a syntactic failure is fixed on the in-step retry
  r_sem         chance a semantic corruption is self-corrected on a later step
                (implemented by the policy substituting the true gold ref)
"""

from __future__ import annotations

import random
from dataclasses import dataclass


def _gold_tool_for(task, step: int, ref: int) -> str:
    if hasattr(task, "branches"):  # RoutingTask
        even_t, odd_t = task.branches[step]
        return (even_t if ref % 2 == 0 else odd_t).name
    return task.gold[step].tool  # linear Task: fixed order


@dataclass
class SimPolicyConfig:
    name: str
    p0: float
    p_slope: float
    pi: float
    syntax_share: float
    r_syn: float
    r_sem: float
    seed: int = 0


class SimPolicy:
    """Stateful simulated policy.

    `stateless` must be set True while driving a teacher-forced run. Teacher
    forcing presents a CORRECT upstream history at every step by construction,
    so the policy must treat every such call as entering on a clean context and
    must not carry corruption between steps.

    Without that flag the poisoned state -- keyed by task_id, on a single policy
    instance serving both run modes for a task -- leaked from the free run into
    the clean baseline. Measured effect: at depth 1, where no upstream history
    exists and poisoning is impossible, 31% of teacher-forced calls entered
    believing they were poisoned, and L_1 came out at -0.113 when it is 0 by
    construction. The bias scaled with pi, so it was structure and not noise.
    Real models are unaffected -- they hold no cross-call state, and at depth 1
    both run modes build an identical prompt and share one cached response, so
    L_1 is exactly 0 there.
    """

    def __init__(self, cfg: SimPolicyConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        # per-task state: whether poisoned entering this step, and its origin
        self._state: dict[str, tuple[bool, str | None]] = {}
        self.stateless = False

    def clean_p(self, step: int) -> float:
        return min(0.99, max(0.02, self.cfg.p0 - self.cfg.p_slope * step))

    def _set_state(self, key: str, value: tuple[bool, str | None]) -> None:
        if not self.stateless:
            self._state[key] = value

    def __call__(self, task, step: int, ref: int, attempt: int):
        cfg, rng = self.cfg, self.rng
        key = task.task_id
        if self.stateless:
            poisoned, origin = False, None
        else:
            poisoned, origin = self._state.get(key, (False, None))
        p_t = self.clean_p(step)

        if attempt > 0:
            # this is an in-step retry after a syntactic (parse/schema) failure
            success = rng.random() < cfg.r_syn
            if success:
                self._set_state(key, (False, None))
                gold_ref = task.gold[step].args["ref"] if not hasattr(task, "branches") else ref
                tool = _gold_tool_for(task, step, ref)
                return tool, {"ref": ref}, True
            # still failing: keep sending a malformed call
            return _gold_tool_for(task, step, ref), {}, True

        eff_p = p_t if not poisoned else (1.0 - cfg.pi) * p_t
        gold_tool = _gold_tool_for(task, step, ref)

        # every step while poisoned gets a chance to recover, using the rate
        # matching the origin of the corruption. Without this, an unresolved
        # syntax failure (retry budget exhausted, never executed) had no way
        # to ever clear on later steps, which produced permanent contamination
        # unrelated to pi and confounded the fit. This is the fix.
        if poisoned:
            r = cfg.r_syn if origin == "syntax" else cfg.r_sem
            if rng.random() < r:
                true_ref = task.gold[step].args.get("ref", ref)
                self._set_state(key, (False, None))
                return gold_tool, {"ref": true_ref}, True

        if rng.random() < eff_p:
            args = {"ref": ref}
            if not poisoned:
                self._set_state(key, (False, None))
            return gold_tool, args, True

        # a fresh error occurs this step
        if rng.random() < cfg.syntax_share:
            self._set_state(key, (True, "syntax"))
            return gold_tool, {}, True   # missing required 'ref' -> syntactic
        else:
            self._set_state(key, (True, "semantic"))
            if rng.random() < 0.4:
                # wrong tool (selection error) -- only meaningful on RoutingTask,
                # harmless mislabel on a linear Task (still counts as wrong)
                wrong_tool = f"{gold_tool}_x"
                return wrong_tool if hasattr(task, "branches") else gold_tool, \
                       {"ref": ref if not hasattr(task, "branches") else ref + rng.randint(1, 9)}, True
            offset = rng.randint(1, 11)
            return gold_tool, {"ref": ref + offset}, True


# a small suite spanning weak to strong, across two "families", mirroring the
# scale/family axes in the methodology's real model suite
DEFAULT_SUITE = [
    SimPolicyConfig("fam0-weak",   p0=0.78, p_slope=0.020, pi=0.70, syntax_share=0.5, r_syn=0.35, r_sem=0.05, seed=1),
    SimPolicyConfig("fam0-mid",    p0=0.88, p_slope=0.012, pi=0.45, syntax_share=0.5, r_syn=0.55, r_sem=0.15, seed=2),
    SimPolicyConfig("fam0-strong", p0=0.95, p_slope=0.006, pi=0.22, syntax_share=0.5, r_syn=0.75, r_sem=0.30, seed=3),
    SimPolicyConfig("fam1-weak",   p0=0.75, p_slope=0.022, pi=0.65, syntax_share=0.35, r_syn=0.30, r_sem=0.05, seed=4),
    SimPolicyConfig("fam1-mid",    p0=0.86, p_slope=0.013, pi=0.40, syntax_share=0.35, r_syn=0.50, r_sem=0.12, seed=5),
    SimPolicyConfig("fam1-strong", p0=0.94, p_slope=0.007, pi=0.18, syntax_share=0.35, r_syn=0.70, r_sem=0.25, seed=6),
]
FAMILY_INDEX = {"fam0-weak": 0, "fam0-mid": 0, "fam0-strong": 0,
               "fam1-weak": 1, "fam1-mid": 1, "fam1-strong": 1}
