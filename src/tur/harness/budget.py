"""Nested unequal-n allocation under per-model daily token caps.

The problem this solves: on Groq's free tier the daily TOKEN allowance (TPD) is
the binding constraint, it differs by model (100,000 for llama-3.3-70b against
500,000 for allam-2-7b), and it appears in no response header. Sizing every model
to the smallest allowance wastes most of the capacity of the others and was
costing the study its statistical power.

The design instead is **nested unequal n**. `generate_suite` and
`generate_routing_suite` emit tasks in a deterministic order for a fixed
base_seed -- depth by depth, k ascending, with ids `r{d}_{k}_s{base_seed}` that
do not depend on how many tasks were asked for. So asking for fewer tasks at a
depth returns exactly a PREFIX of what asking for more returns. If the binding
model runs n_min tasks at each depth and a richer model runs n_i >= n_min, the
first n_min are the same tasks, by construction rather than by convention.

What that buys:

  * cross-model comparison stays exactly paired on the common prefix, which is
    what the paired bootstrap and every scale/family contrast rely on
  * each model's own L_t, p_t and g_t use its full n, so richer models get
    tighter intervals instead of being truncated to the poorest one
  * the binomial likelihood in the hierarchical fit already takes per-model
    trial counts, so unequal n needs no change there
  * pooled per-step error counts (H4) draw most of their mass from the
    high-allowance models, which is where the tokens actually are

The cost model is measured, not assumed: routing cost is roughly quadratic in
depth because the schema scales with depth and the whole prompt is re-sent at
every step, so a depth-8 task costs ~37x a depth-1 task rather than 8x.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tur.harness.executor import FeedbackMode
from tur.harness.runner import MockBackend, run_free, run_teacher_forced
from tur.tasks.dag import generate_routing_suite, generate_suite

OUTPUT_TOKENS_PER_CALL = 40


def _tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return (lambda s: len(enc.encode(s))), "tiktoken/o200k_base"
    except Exception:
        return (lambda s: len(s) // 4), "chars/4 APPROXIMATION"


def measure_unit_costs(depths: list[int], variant: str = "routing",
                       distractor_level: int = 1, max_retries: int = 1,
                       realistic: bool = True, n_tasks: int = 40
                       ) -> dict[int, int]:
    """Tokens for ONE task at each depth, both arms.

    Driven through the real run loops against a mock backend so it cannot go
    stale when prompt assembly changes -- the same reason estimate_cost.py stopped
    reimplementing the prompt build.

    `realistic=True` drives an ERRING policy calibrated to the accuracy actually
    measured on routing (p ~ 0.45-0.50 at depth 4), averaged over `n_tasks` tasks
    per depth. A perfect policy is a floor, not an estimate: every failure adds a
    feedback turn and possibly a retry, and each of those re-sends the whole
    transcript. Measured across the reference allocation the erring policy costs
    **1.27x** the perfect-policy floor, uniformly ~1.25-1.33x per depth. Budgeting
    against the floor is therefore a 27% under-estimate, which on a tier where the
    token allowance is the binding constraint converts directly into a cap-stop
    partway through.
    """
    ntok, _ = _tokenizer()
    gen = generate_routing_suite if variant == "routing" else generate_suite
    out: dict[int, int] = {}

    def perfect(task, step, ref, attempt):
        return task.gold[step].tool, {"ref": ref}, True

    policy = None
    if realistic:
        from tur.harness.sim_policy import SimPolicy, SimPolicyConfig
        policy = SimPolicy(SimPolicyConfig(
            "cost-probe", p0=0.55, p_slope=0.02, pi=0.55, syntax_share=0.45,
            r_syn=0.35, r_sem=0.15, seed=5))

    for d in depths:
        reps = n_tasks if realistic else 1
        tasks = gen([d], reps, distractor_level, base_seed=1000)
        total = 0
        for task in tasks:
            for runner in (run_free, run_teacher_forced):
                c = {"t": 0, "n": 0}
                backend = MockBackend(policy if policy is not None else perfect)
                inner = backend.complete

                def complete(messages, tools, mode, _c=c, _i=inner):
                    _c["t"] += ntok("".join(str(m.get("content", ""))
                                           for m in messages))
                    _c["n"] += 1
                    return _i(messages, tools, mode)

                backend.complete = complete
                if policy is not None:
                    # teacher forcing presents a correct history by construction,
                    # so the policy must not carry corruption into it
                    policy.stateless = runner is run_teacher_forced
                runner(task, backend, "uniform", FeedbackMode.STRUCTURED,
                       max_retries)
                total += c["t"] + c["n"] * OUTPUT_TOKENS_PER_CALL
        out[d] = int(round(total / reps))
    return out


def allocation_cost(per_depth: dict[int, int], unit: dict[int, int]) -> int:
    return sum(n * unit[d] for d, n in per_depth.items() if n)


@dataclass
class ModelPlan:
    name: str
    tpd: int
    budget_tokens: int
    scale: float
    primary: dict[int, int]
    control: dict[int, int]
    cost_tokens: int
    calls: int
    notes: list[str] = field(default_factory=list)

    @property
    def utilisation(self) -> float:
        return self.cost_tokens / self.budget_tokens if self.budget_tokens else 0.0


def _scaled(reference: dict[int, int], f: float, floor: int) -> dict[int, int]:
    return {d: max(floor, int(round(n * f))) for d, n in reference.items()}


def plan_model(name: str, tpd: int, days: float, headroom: float,
               ref_primary: dict[int, int], ref_control: dict[int, int],
               unit_primary: dict[int, int], unit_control: dict[int, int],
               floor: int = 5, control_floor: int = 4) -> ModelPlan:
    """Largest nested allocation for one model that fits its own token budget.

    Both arms scale by the same factor, so the control arm shrinks alongside the
    primary rather than crowding it out on the poorest model -- at the reference
    size it is 17% of the bill, which is more than a null control is worth when
    the binding model has 100,000 tokens a day.

    Floors exist because a depth bin with one or two tasks is not a measurement.
    A model whose budget cannot cover the floors is reported at the floor with a
    note rather than silently rounded to nothing: that is a finding about the
    tier, not something to hide in an allocation table.
    """
    budget = int(days * tpd * headroom)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        cost = (allocation_cost(_scaled(ref_primary, mid, floor), unit_primary)
                + allocation_cost(_scaled(ref_control, mid, control_floor),
                                  unit_control))
        if cost <= budget:
            lo = mid
        else:
            hi = mid
    f = lo
    primary = _scaled(ref_primary, f, floor)
    control = _scaled(ref_control, f, control_floor)
    cost = (allocation_cost(primary, unit_primary)
            + allocation_cost(control, unit_control))
    notes = []
    if cost > budget:
        notes.append(f"floors ({floor} primary / {control_floor} control) exceed "
                     f"this model's {days:g}-day budget of {budget:,} tokens by "
                     f"{cost / budget:.2f}x; it will cap-stop before finishing")
    if f >= 0.999:
        notes.append("budget covers the full reference allocation")
    calls = (sum(n * 2 * d for d, n in primary.items())
             + sum(n * 2 * d for d, n in control.items()))
    return ModelPlan(name=name, tpd=tpd, budget_tokens=budget, scale=f,
                     primary=primary, control=control, cost_tokens=cost,
                     calls=calls, notes=notes)


def plan_suite(models: list[dict], days: float, headroom: float,
               ref_primary: dict[int, int], ref_control: dict[int, int],
               control_variant: str = "linear",
               primary_variant: str = "routing",
               distractor_level: int = 1,
               max_retries: int = 1) -> tuple[list[ModelPlan], dict, dict]:
    unit_primary = measure_unit_costs(sorted(ref_primary), primary_variant,
                                      distractor_level, max_retries)
    unit_control = measure_unit_costs(sorted(ref_control), control_variant,
                                      distractor_level, max_retries)
    plans = []
    for m in models:
        tpd = m.get("tpd")
        if not tpd:
            # An unmeasured TPD is not an absent one -- every model on this tier
            # has one. Planning as if it were unlimited is how a run ends up
            # cap-stopping on day 1, so fall back to the smallest MEASURED value
            # in the suite and say so.
            measured = [x.get("tpd") for x in models if x.get("tpd")]
            tpd = min(measured) if measured else 100_000
            plan = plan_model(m["name"], tpd, days, headroom, ref_primary,
                              ref_control, unit_primary, unit_control)
            plan.notes.append(f"TPD not measured; planned against the smallest "
                              f"measured value in the suite ({tpd:,}) as a "
                              f"conservative stand-in")
        else:
            plan = plan_model(m["name"], tpd, days, headroom, ref_primary,
                              ref_control, unit_primary, unit_control)
        plans.append(plan)
    return plans, unit_primary, unit_control


def pooled_step_errors(plans: list[ModelPlan], p_correct: float = 0.50,
                       recovery: float = 0.25, max_step: int = 8
                       ) -> dict[int, float]:
    """Projected FRESH-error count at each step index, pooled across models.

    A step index i exists only in tasks of depth > i, and only trajectories still
    holding a clean context at step i can contribute a fresh error -- that is the
    restriction aggregate_by_step applies. Clean-context survival is propagated as
    c_{i+1} = c_i * p + (1 - c_i) * r, with p the measured routing accuracy.
    """
    c, surv = 1.0, []
    for _ in range(max_step):
        surv.append(c)
        c = c * p_correct + (1 - c) * recovery
    out = {}
    for i in range(max_step):
        deeper = sum(n for plan in plans for d, n in plan.primary.items() if d > i)
        out[i] = deeper * surv[i] * (1 - p_correct)
    return out
