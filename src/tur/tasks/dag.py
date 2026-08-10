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
    # When non-zero, the argument to send is a stated TRANSFORMATION of the value
    # carried in, not a verbatim copy of it: ref = (carried + arg_shift) % MOD.
    #
    # This exists because H4 -- the claim that error composition shifts from
    # selection-dominated to argument-dominated along a chain -- turned out to be
    # untestable on the copy-argument variant. Measured on real models, 302 of 302
    # clean-context errors were SELECTION errors and zero were argument errors:
    # applying the routing rule is hard, transcribing a number that is written in
    # the previous observation is not, so the argument channel was empty exactly
    # where the composition is defined. A stated transformation makes an argument
    # wrong-able independently of tool choice, which is the minimum needed for the
    # hypothesis to have two categories to shift between.
    arg_shift: int = 0
    # Per-step presentation order for the rule text: True means the ODD branch is
    # listed first at that step. Exists as a CONTROL, because with a fixed order
    # "picks the first-listed tool" and "applies the rule correctly for even refs"
    # predict overlapping data -- the correct tool for an even ref just is the
    # first-listed one. Randomising the order breaks that confound directly rather
    # than relying on a discrimination statistic to separate the two.
    present_odd_first: list[bool] = field(default_factory=list)
    tool_by_name: dict[str, ToolSpec] = field(default_factory=dict)

    def __post_init__(self):
        self.tool_by_name = {t.name: t for pair in self.branches for t in pair}

    def arg_rule_text(self) -> str:
        """Model-facing description of the argument transformation, if any."""
        if not self.arg_shift:
            return ""
        return (f"IMPORTANT: the 'ref' you pass is NOT the previous result "
                f"verbatim. At every step after the first, pass "
                f"(previous result + {self.arg_shift}) mod {MOD}. "
                f"The routing rule above is applied to the previous result "
                f"itself, before that addition.")

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
            odd_first = (self.present_odd_first[i]
                         if i < len(self.present_odd_first) else False)
            if odd_first:
                lines.append(f"  step {i}: if the incoming ref is odd call "
                             f"'{odd_t.name}', if even call '{even_t.name}'")
            else:
                lines.append(f"  step {i}: if the incoming ref is even call "
                             f"'{even_t.name}', if odd call '{odd_t.name}'")
        return "\n".join(lines)


def generate_routing_task(task_id: str, depth: int, distractor_level: int = 1,
                          seed: int = 0, arg_shift: int = 0,
                          shuffle_branch_order: bool = False) -> RoutingTask:
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
    for i, (even_t, odd_t) in enumerate(branches):
        # the branch is chosen on the value carried IN, before any transformation,
        # so the rule the model is told stays a rule about the previous result
        chosen = even_t if prev % 2 == 0 else odd_t
        # the first step has no "previous result", so the transformation applies
        # only from step 1 onward -- otherwise the seed value stated in the prompt
        # would itself have to be transformed, which the rule text does not say
        ref = prev if (i == 0 or not arg_shift) else (prev + arg_shift) % MOD
        args = {arg_name: ref}
        out = chosen.run(args)
        gold.append(Step(tool=chosen.name, args=args, output=out))
        prev = out

    # presentation order only changes how the rule is WORDED, never which tool is
    # correct, so the gold trajectory above is untouched by it
    order = ([rng.random() < 0.5 for _ in range(depth)] if shuffle_branch_order
             else [False] * depth)
    task = RoutingTask(task_id=task_id, depth=depth, branches=branches, gold=gold,
                       seed_value=seed_value, distractor_level=distractor_level,
                       arg_shift=arg_shift, present_odd_first=order)
    return task


def generate_routing_suite(depths: list[int], per_depth: "int | dict[int, int]",
                           distractor_level: int = 1,
                           base_seed: int = 0,
                           arg_shift: int = 0,
                           shuffle_branch_order: bool = False) -> list[RoutingTask]:
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
                                               distractor_level, seed,
                                               arg_shift=arg_shift,
                                               shuffle_branch_order=shuffle_branch_order))
    return tasks
