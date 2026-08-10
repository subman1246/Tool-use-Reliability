"""Contamination-free synthetic multi-step tool-use tasks.

A task is a linear dependency chain of deterministic integer functions. Each
node consumes the output of its predecessor, so dependency depth equals the
chain length. Because every function is deterministic and the gold trajectory
is computed here, we have exact per-call ground truth, which lets us label
semantic errors that execute without raising an exception.

Semantic noise is added at the prompt surface: verbose, paraphrased tool
descriptions and distractor tools whose names overlap with the correct one.
The underlying values stay exact so scoring is unambiguous.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

MOD = 100000  # keep integer outputs bounded and comparable


@dataclass
class ToolSpec:
    name: str
    description: str
    params: list[dict[str, Any]]  # [{name, type, required}]
    # true operation constants, not exposed to the model
    a: int = 1
    b: int = 0

    def run(self, args: dict[str, Any]) -> int:
        """Execute the true operation on integer arguments.

        Any well-formed integer call returns a value, which is what makes a
        wrong-but-valid (semantic) call return plausible, irrelevant data.
        """
        vals = [int(v) for k, v in sorted(args.items()) if isinstance(v, (int, float))]
        s = sum(vals)
        return (self.a * s + self.b) % MOD


@dataclass
class Step:
    tool: str
    args: dict[str, Any]
    output: int


@dataclass
class Task:
    task_id: str
    depth: int
    tools: list[ToolSpec]           # correct tools plus distractors, shuffled
    gold: list[Step]                # ordered gold trajectory
    seed_value: int
    distractor_level: int
    tool_by_name: dict[str, ToolSpec] = field(default_factory=dict)

    def __post_init__(self):
        self.tool_by_name = {t.name: t for t in self.tools}

    def schema_view(self) -> list[dict[str, Any]]:
        """Model-facing schema: names, descriptions, params only."""
        return [
            {"name": t.name, "description": t.description,
             "parameters": {p["name"]: {"type": p["type"], "required": p["required"]}
                            for p in t.params}}
            for t in self.tools
        ]


_DESC_TEMPLATES = [
    "Retrieve the {noun} record associated with the provided upstream identifier.",
    "Compute the {noun} value derived from the previous step's result.",
    "Look up the {noun} entry keyed by the incoming reference number.",
    "Transform the supplied reference into its corresponding {noun} figure.",
]
_NOUNS = ["ledger", "settlement", "manifest", "invoice", "roster", "quota",
          "balance", "shipment", "allocation", "reconciliation"]


def _make_tool(rng: random.Random, idx: int, arg_name: str) -> ToolSpec:
    noun = rng.choice(_NOUNS)
    desc = rng.choice(_DESC_TEMPLATES).format(noun=noun)
    return ToolSpec(
        name=f"op_{noun}_{idx}",
        description=desc,
        params=[{"name": arg_name, "type": "integer", "required": True}],
        a=rng.randint(2, 9),
        b=rng.randint(1, 999),
    )


def generate_task(task_id: str, depth: int, distractor_level: int = 1,
                  seed: int = 0) -> Task:
    """Build one task: a chain of `depth` dependent calls plus distractors.

    distractor_level controls how many similarly named unused tools appear in
    the schema (0 = none, 1 = depth, 2 = 2*depth).
    """
    rng = random.Random(seed)
    arg_name = "ref"
    chain: list[ToolSpec] = [_make_tool(rng, i, arg_name) for i in range(depth)]

    # compute gold trajectory
    gold: list[Step] = []
    seed_value = rng.randint(1, MOD - 1)
    prev = seed_value
    for t in chain:
        args = {arg_name: prev}
        out = t.run(args)
        gold.append(Step(tool=t.name, args=args, output=out))
        prev = out

    # distractors: extra tools with overlapping names, never part of gold
    n_distract = distractor_level * depth
    distractors = [_make_tool(rng, 1000 + i, arg_name) for i in range(n_distract)]

    tools = chain + distractors
    rng.shuffle(tools)
    return Task(task_id=task_id, depth=depth, tools=tools, gold=gold,
                seed_value=seed_value, distractor_level=distractor_level)


def generate_suite(depths: list[int], per_depth: "int | dict[int, int]",
                   distractor_level: int = 1, base_seed: int = 0) -> list[Task]:
    """Generate `per_depth` tasks at each depth in `depths`.

    `per_depth` may be a single count applied to every depth, or a per-depth
    mapping {depth: count}. The mapping form exists because task cost is linear
    in depth (a depth-8 task issues 8x the calls of a depth-1 task) while the
    propagation signal is concentrated at depth -- L_1 is 0 by construction --
    so a uniform allocation spends most of a rate-limited budget on the least
    informative bin. Depths absent from the mapping fall back to 0 tasks.
    """
    tasks: list[Task] = []
    for d in depths:
        n = per_depth[d] if isinstance(per_depth, dict) else per_depth
        for k in range(n):
            seed = base_seed + d * 100003 + k
            # base_seed belongs in the id: it is what distinguishes one
            # generator seed's suite from another's. See generate_routing_suite
            # for what went wrong when it was left out.
            tasks.append(generate_task(f"d{d}_{k}_s{base_seed}", d,
                                       distractor_level, seed))
    return tasks


@dataclass
class RoutingTask:
    """A dependency chain where the NEXT tool is not fixed in advance.

    Unlike Task (a linear, pre-announced chain), here each step's correct tool
    is a deterministic function of the previous output (odd -> tool A, even ->
    tool B, out of two candidates at that position). The model is told the
    routing rule, not the resulting tool sequence, so a wrong tool choice at
    step t changes which branch is "correct" at step t+1 as well as the value
    carried forward. This is what lets a selection error propagate, which the
    linear Task cannot exercise (there, the order is given and only values
    drift).
    """
    task_id: str
    depth: int
    branches: list[tuple[ToolSpec, ToolSpec]]   # (even_tool, odd_tool) per step
    gold: list[Step]
    seed_value: int
    distractor_level: int
    tool_by_name: dict[str, ToolSpec] = field(default_factory=dict)

    def __post_init__(self):
        self.tool_by_name = {t.name: t for pair in self.branches for t in pair}

    def schema_view(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description,
             "parameters": {p["name"]: {"type": p["type"], "required": p["required"]}
                            for p in t.params}}
            for pair in self.branches for t in pair
        ]

    def routing_rule_text(self) -> str:
        lines = []
        for i, (even_t, odd_t) in enumerate(self.branches):
            lines.append(f"  step {i}: if the incoming ref is even call "
                         f"'{even_t.name}', if odd call '{odd_t.name}'")
        return "\n".join(lines)


def generate_routing_task(task_id: str, depth: int, distractor_level: int = 1,
                          seed: int = 0) -> RoutingTask:
    """Build one routing task.

    `distractor_level` is accepted, recorded on the task, and DELIBERATELY not
    used to add tools. It is inert here, which the config's `distractor_level: 1`
    does not convey on its own, hence this note.

    The reason is that routing needs no separate distractors: the schema already
    contains 2*depth tools, exactly one of which is correct at each step, and the
    wrong one at each step is a genuine confusable rather than a tool that is
    never correct anywhere. That is stronger selection pressure than the linear
    variant's distractors provide, not weaker. Adding linear-style distractors on
    top would change the arms in different directions and make the selection-error
    rates non-comparable between them.

    The consequence to keep in mind when reading results: the two arms differ in
    schema composition as well as in routing, so cross-arm differences in
    selection-error rate are not attributable to the routing rule alone. We
    therefore do not draw cross-arm inferences from selection rates; the control
    arm is used only as a null for L_t.
    """
    rng = random.Random(seed)
    arg_name = "ref"
    branches: list[tuple[ToolSpec, ToolSpec]] = []
    for i in range(depth):
        even_t = _make_tool(rng, i * 2, arg_name)
        odd_t = _make_tool(rng, i * 2 + 1, arg_name)
        branches.append((even_t, odd_t))

    gold: list[Step] = []
    seed_value = rng.randint(1, MOD - 1)
    prev = seed_value
    for even_t, odd_t in branches:
        chosen = even_t if prev % 2 == 0 else odd_t
        args = {arg_name: prev}
        out = chosen.run(args)
        gold.append(Step(tool=chosen.name, args=args, output=out))
        prev = out

    task = RoutingTask(task_id=task_id, depth=depth, branches=branches, gold=gold,
                       seed_value=seed_value, distractor_level=distractor_level)
    return task


def generate_routing_suite(depths: list[int], per_depth: "int | dict[int, int]",
                           distractor_level: int = 1,
                           base_seed: int = 0) -> list[RoutingTask]:
    """As generate_suite, for the routing variant.

    task_id embeds base_seed. It did not, and the consequence was severe: every
    generator seed produced the SAME ids, so a multi-seed run handed two
    genuinely different tasks the same identifier. Every statistic that groups by
    task_id then silently merged them:

      measure_recovery  interleaved two unrelated trajectories under one id and
                        read transitions across the join, which is how a
                        configured r_syn of 0.35 measured 0.10 at seeds=2 while
                        measuring 0.34 at seeds=1
      delta_1           built a {step: correct} dict per id, so the second seed
                        OVERWROTE the first and half the data was discarded
                        without any warning
      bootstrap_L_ci    resampled ids, so each draw pulled a pair of tasks
                        rather than one, changing the interval width

    None of it raised. The real run uses seeds: 1, where ids happen not to
    collide, so this is a validation-path defect -- but every multi-seed
    validation number computed before the fix is affected, which is why the
    validation was re-run rather than patched in the write-up.
    """
    tasks: list[RoutingTask] = []
    for d in depths:
        n = per_depth[d] if isinstance(per_depth, dict) else per_depth
        for k in range(n):
            seed = base_seed + 500000 + d * 100003 + k
            tasks.append(generate_routing_task(f"r{d}_{k}_s{base_seed}", d,
                                               distractor_level, seed))
    return tasks
