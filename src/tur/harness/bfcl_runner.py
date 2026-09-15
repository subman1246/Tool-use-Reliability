"""L_d on BFCL multi-turn tasks: the same two arms, against a real executable environment.

SCOPE, FIXED BEFORE ANY SPEND

This measures ONE thing: the depth-dependent propagation loss L_d = 1 - g_d/p_d, where
p_d is per-call gold agreement under a correct baseline history (teacher-forced) and g_d
is the same under the model's own history (free-running). It deliberately does NOT
implement a conditional-on-state scorer. BFCL has no rule mapping "the state I am actually
in" to "the call that would be correct from here" -- its gold expresses a user's intent,
and after a divergence there is no oracle for the right continuation. Building one would
mean inventing the oracle whose absence is the point.

THE SCORER HERE IS OURS, NOT BFCL'S

BFCL's own multi-turn checker compares final environment state and accepts the ground
truth's execution results as an UNORDERED SUBSET of the model's, permitting reordering and
extra calls; its trajectory-order check is commented out. See
docs/NOTE_bfcl_scorer_structure.md, which reads this off the scoring source. So these
numbers are our fixed-gold per-call scorer applied to BFCL's tasks, and are not comparable
to published BFCL leaderboard scores. The tasks are being used as a realistic task SOURCE.

WHY THE ENVIRONMENT IS REALLY EXECUTED

g_d requires the model's own history, which means its own wrong calls must actually run and
produce whatever results they produce, against its own environment instance. Feeding back
invented results would make the free arm a simulation of a divergence rather than one.
Each arm therefore gets a freshly instantiated environment loaded from initial_config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tur.harness.runner import StepRecord, _SYSTEM, _render_schema, parse_response
from tur.tasks.bfcl import (ANSWERS, DATA, BFCL_DIR, instantiate, parse_call)

_DOC_DIR = BFCL_DIR / "func_doc"
_DOC_FILE = {
    "GorillaFileSystem": "gorilla_file_system.json", "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json", "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json", "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json", "VehicleControlAPI": "vehicle_control.json",
}

_TF = "bfcl_teacher_forced"
_FREE = "bfcl_free"


def _load_docs(class_name: str) -> list[dict]:
    p = _DOC_DIR / _DOC_FILE[class_name]
    if not p.exists():
        raise SystemExit("missing %s -- run scripts/fetch_bfcl.py" % p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


@dataclass
class BFCLTask:
    task_id: str
    depth: int
    entry: dict
    gold: list[dict]                      # [{tool, args, call}]
    turn_of: list[int]                    # which user turn each gold call belongs to
    questions: list[list[dict]]
    _schema: list[dict] = field(default_factory=list)

    def schema_view(self) -> list[dict[str, Any]]:
        return self._schema


def _param_order(schema: list[dict], tool: str) -> list[str]:
    for d in schema:
        if d["name"] == tool:
            return list(d["parameters"].keys())
    return []


def _flatten_gold(entry: dict) -> tuple[list[dict], list[int]]:
    """Gold calls across all turns, with the turn each belongs to.

    Depth is the total gold-call count. A per-turn notion of depth would be useless here:
    476 of the split's 742 turns contain exactly one call, so almost every task would sit
    at depth 1 and no propagation could be observed.
    """
    gold, turn_of = [], []
    for ti, turn in enumerate(ANSWERS()[entry["id"]]):
        for c in turn:
            name, pos, kw = parse_call(c)
            gold.append({"tool": name, "args": dict(kw), "call": c, "pos": pos})
            turn_of.append(ti)
    return gold, turn_of


def build_task(entry: dict) -> BFCLTask:
    gold, turn_of = _flatten_gold(entry)
    schema: list[dict] = []
    for cn in entry["involved_classes"]:
        for d in _load_docs(cn):
            params = d.get("parameters", {}).get("properties", {})
            required = set(d.get("parameters", {}).get("required", []))
            schema.append({
                "name": d["name"], "description": d.get("description", ""),
                "parameters": {k: {"type": v.get("type", "string"),
                                   "required": k in required}
                               for k, v in params.items()},
            })
    # 33 gold calls in the curated subset use positional arguments (sort('x')). Folding
    # them onto their parameter names from the schema means gold["args"] is the complete
    # call in one form: the free arm can execute the model's keyword-only output, and the
    # scorer compares like with like instead of an empty dict against a filled one.
    for g in gold:
        if g["pos"]:
            names = _param_order(schema, g["tool"])
            for i, v in enumerate(g["pos"]):
                if i < len(names) and names[i] not in g["args"]:
                    g["args"][names[i]] = v
            g["pos"] = []
    return BFCLTask(task_id=entry["id"], depth=len(gold), entry=entry, gold=gold,
                    turn_of=turn_of, questions=entry.get("question", []),
                    _schema=schema)


def _intro(task: BFCLTask, schema_style: str = "verbose") -> str:
    schema = _render_schema(task.schema_view(), schema_style)
    return (f"Tools available:\n{schema}\n\n"
            f"You will be given a user request across {len(task.questions)} turn(s). "
            f"Carry it out by calling tools one at a time. "
            f"Emit one JSON object per step: "
            f'{{"tool": name, "args": {{...}}}}.')


def _turn_text(task: BFCLTask, ti: int) -> str:
    msgs = task.questions[ti] if ti < len(task.questions) else []
    return "\n".join(str(m.get("content", "")) for m in msgs)


def _exec(insts: dict, name: str, pos: list, kw: dict):
    fn = next((getattr(i, name) for i in insts.values() if hasattr(i, name)), None)
    if fn is None:
        return False, "ToolError: unknown tool " + str(name)
    try:
        return True, fn(*pos, **kw)
    except Exception as ex:                                  # noqa: BLE001
        # A wrong-but-well-formed call that the API rejects is a real outcome, not a
        # harness failure: the model sees the error and continues from it, exactly as it
        # would against a live API.
        return False, "ToolError: %s: %s" % (type(ex).__name__, str(ex)[:200])


def _score(call, gold: dict) -> tuple[bool, bool]:
    """(selection_matches_gold, args_correct_strict) against the canonical call."""
    if not call.parse_ok or call.tool is None:
        return False, False
    sel = call.tool == gold["tool"]
    args_ok = sel and (call.args or {}) == gold["args"]
    return sel, args_ok


def _run(task: BFCLTask, backend, mode: str, call_mode: str = "uniform",
         schema_style: str = "verbose") -> list[StepRecord]:
    insts = instantiate(task.entry)
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _intro(task, schema_style)}]
    records: list[StepRecord] = []
    seen_turn = -1

    for k, gold in enumerate(task.gold):
        ti = task.turn_of[k]
        if ti != seen_turn:
            messages.append({"role": "user", "content": _turn_text(task, ti)})
            seen_turn = ti

        ctx = {"task": task, "step": k, "ref": None, "attempt": 0}
        messages.append({"role": "user", "content": "[step %d]" % k, "_ctx": ctx})
        resp = backend.complete(messages, task.schema_view(), call_mode)
        fr = resp.get("finish_reason") if isinstance(resp, dict) else None
        call = parse_response(resp, call_mode, coerce_ints=False)
        sel, args_ok = _score(call, gold)

        if mode == _TF:
            # Baseline history: whatever the model just said, the history continues along
            # the GOLD trajectory, so step k+1 is asked under a correct prefix. This is
            # what makes p_d a per-call capability measure rather than a trajectory one.
            ok, out = _exec(insts, gold["tool"], gold["pos"], gold["args"])
            messages.append({"role": "assistant",
                             "content": json.dumps({"tool": gold["tool"],
                                                    "args": gold["args"]})})
            messages.append({"role": "user", "content": "result: " + str(out)[:400]})
        else:
            ok, out = _exec(insts, call.tool or "", [], call.args or {})
            messages.append({"role": "assistant",
                             "content": json.dumps({"tool": call.tool,
                                                    "args": call.args})})
            messages.append({"role": "user", "content": "result: " + str(out)[:400]})

        records.append(StepRecord(
            task.task_id, task.depth, k, mode, call_mode,
            call.tool, sel, sel, args_ok, args_ok,
            "none" if args_ok else ("parse" if not call.parse_ok else "semantic"),
            1, bool(ok), context_clean_in=(mode == _TF), recovered=False,
            stalled_in=False,
            backend_error=bool(call.is_backend_error),
            args_correct_given_state=False, correct_given_state=False,
            held_ref=None, first_listed_even=None,
            truncated=(fr == "length") if fr is not None else None))
    return records


def run_bfcl_teacher_forced(task, backend, call_mode="uniform", schema_style="verbose"):
    return _run(task, backend, _TF, call_mode, schema_style)


def run_bfcl_free(task, backend, call_mode="uniform", schema_style="verbose"):
    return _run(task, backend, _FREE, call_mode, schema_style)


def curated_subset(min_depth: int = 6, max_depth: int = 10,
                   n: int | None = None) -> list[BFCLTask]:
    """Tasks in the split whose gold depth falls in range, in dataset order.

    Dataset order rather than a sample: the selection criterion is depth alone, stated up
    front, so the subset is reproducible without a seed and cannot be tuned after seeing
    results.
    """
    tasks = [build_task(e) for e in DATA()]
    sel = [t for t in tasks if min_depth <= t.depth <= max_depth]
    return sel[:n] if n else sel
