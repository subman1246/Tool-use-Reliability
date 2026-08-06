"""Run loop over tasks.

Two calling modes:
  uniform : schema is serialised into the prompt, model emits one JSON object,
            one shared parser reads every model. This is the headline protocol.
  native  : provider-native tool calling via LiteLLM tools=, parsed from the
            structured tool_call. Used only for the calling-mode ablation.

Two run modes:
  free           : the model threads its own outputs forward (measures g_t).
  teacher_forced : each step is presented with the correct upstream history at
                   its true length (measures the depth-varying baseline p_t).

Syntactic failures (non-executing calls) trigger bounded within-step retries,
which is where syntactic recovery (r_syn) can occur. Semantic errors execute
and are carried forward, which is how propagation happens.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

from tur.eval.scoring import ParsedCall, ErrorType, score_step
from tur.harness.executor import FeedbackMode, execute
from tur.tasks.dag import Task


# --------------------------- backends ---------------------------

class Backend(Protocol):
    model: str
    def complete(self, messages: list[dict], tools: list[dict] | None,
                 mode: str) -> dict: ...


class MockBackend:
    """Offline backend driven by a scripted policy.

    policy(task, step_index, current_ref, attempt) -> (tool, args, parse_ok)
    Lets tests inject selection errors, wrong values, schema violations, and
    retry-recovery deterministically without any network.
    """

    def __init__(self, policy: Callable[[Task, int, int, int], tuple], model: str = "mock"):
        self.policy = policy
        self.model = model

    def complete(self, messages, tools, mode):
        ctx = messages[-1]["_ctx"]  # runner stashes structured context here
        tool, args, parse_ok = self.policy(ctx["task"], ctx["step"],
                                            ctx["ref"], ctx["attempt"])
        if not parse_ok:
            return {"text": "not valid json {{"}
        if mode == "native":
            return {"tool_call": {"name": tool, "arguments": json.dumps(args)}}
        return {"text": json.dumps({"tool": tool, "args": args})}


class LiteLLMBackend:
    """Provider-agnostic backend via LiteLLM. Imported lazily."""

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def complete(self, messages, tools, mode):
        import litellm  # lazy so the package imports without the dep
        clean = [{k: v for k, v in m.items() if not k.startswith("_")}
                 for m in messages]
        kwargs: dict[str, Any] = dict(model=self.model, messages=clean,
                                      temperature=self.temperature)
        if mode == "native" and tools:
            kwargs["tools"] = [{"type": "function",
                                "function": {"name": t["name"],
                                             "description": t["description"],
                                             "parameters": _json_schema(t)}}
                               for t in tools]
        resp = litellm.completion(**kwargs)
        msg = resp["choices"][0]["message"]
        if mode == "native" and msg.get("tool_calls"):
            tc = msg["tool_calls"][0]["function"]
            return {"tool_call": {"name": tc["name"], "arguments": tc["arguments"]}}
        return {"text": msg.get("content") or ""}


def _json_schema(tool_view: dict) -> dict:
    props, required = {}, []
    for name, meta in tool_view["parameters"].items():
        jt = "integer" if meta["type"] == "integer" else "string"
        props[name] = {"type": jt}
        if meta["required"]:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}


# --------------------------- parsing ---------------------------

def parse_response(resp: dict, mode: str) -> ParsedCall:
    if mode == "native":
        tc = resp.get("tool_call")
        if not tc:
            return ParsedCall(None, None, parse_ok=False, raw=str(resp))
        try:
            args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
            return ParsedCall(tc["name"], _coerce_ints(args), True, str(resp))
        except (json.JSONDecodeError, TypeError):
            return ParsedCall(None, None, False, str(resp))
    text = resp.get("text", "")
    try:
        obj = json.loads(_extract_json(text))
        return ParsedCall(obj.get("tool"), _coerce_ints(obj.get("args", {})),
                          True, text)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return ParsedCall(None, None, False, text)


def _extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end != -1 else text


def _coerce_ints(args: dict | None) -> dict:
    if not isinstance(args, dict):
        return {}
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and v.lstrip("-").isdigit():
            out[k] = int(v)
        else:
            out[k] = v
    return out


# --------------------------- records ---------------------------

@dataclass
class StepRecord:
    task_id: str
    depth: int
    step: int
    run_mode: str
    call_mode: str
    tool: str | None
    selection_correct: bool          # correct given the ref actually held
    selection_matches_gold: bool     # matches the gold trajectory
    args_correct_strict: bool
    args_correct_soft: bool
    error_type: str
    n_attempts: int
    executed: bool
    context_clean_in: bool
    recovered: bool
    stalled_in: bool = False         # entered this step on a stalled chain


# --------------------------- prompt ---------------------------

_SYSTEM = ("You are a tool-using agent. At each step call exactly one tool. "
           "Respond in the requested format only.")


def _task_intro(task) -> str:
    schema = json.dumps(task.schema_view(), indent=0)
    if hasattr(task, "routing_rule_text"):  # RoutingTask
        return (f"Tools available:\n{schema}\n\n"
                f"Perform {task.depth} steps. At each step, choose the tool "
                f"according to this rule, applied to the incoming ref value:\n"
                f"{task.routing_rule_text()}\n\n"
                f"The first step's ref={task.seed_value}. Each later step's ref "
                f"equals the numeric result returned by the previous tool. "
                f"Emit one JSON object per step: {{\"tool\": name, \"args\": {{\"ref\": value}}}}.")
    order = " -> ".join(s.tool for s in task.gold)
    return (f"Tools available:\n{schema}\n\n"
            f"Perform {task.depth} steps in this order: {order}. "
            f"The first tool takes ref={task.seed_value}. Each later tool takes "
            f"ref equal to the numeric result returned by the previous tool. "
            f"Emit one JSON object per step: {{\"tool\": name, \"args\": {{\"ref\": value}}}}.")


# --------------------------- run loops ---------------------------

def _expected_tool_given_ref(task, step: int, ref):
    """What tool SHOULD be called at this step given the ref actually held.

    For a linear Task the answer is fixed (the announced order), so this equals
    the gold tool. For a RoutingTask the correct tool is a function of the
    incoming value, so once the context is poisoned the gold tool is no longer
    the right yardstick: an agent that applies the routing rule perfectly to a
    corrupted ref will legitimately call a different tool than gold. Scoring
    that as a selection error would attribute an argument-propagation failure
    to the selection channel and corrupt the error-type decomposition.
    """
    if hasattr(task, "branches"):
        even_t, odd_t = task.branches[step]
        if not isinstance(ref, (int, float)) or isinstance(ref, bool):
            return None  # cannot determine; fall back to gold
        return (even_t if int(ref) % 2 == 0 else odd_t).name
    return task.gold[step].tool


def run_free(task: Task, backend: Backend, call_mode: str = "uniform",
             feedback: FeedbackMode = FeedbackMode.STRUCTURED,
             max_retries: int = 1) -> list[StepRecord]:
    records: list[StepRecord] = []
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _task_intro(task)}]
    carried = task.seed_value
    stalled = False   # a prior step exhausted retries without ever executing
    for t, gold in enumerate(task.gold):
        expected_ref = task.gold[t].args["ref"]
        stalled_in = stalled
        context_clean = (carried == expected_ref) and not stalled
        attempts = 0
        recovered = False
        final_score = None
        executed = False
        call = None
        for attempt in range(max_retries + 1):
            attempts += 1
            ctx = {"task": task, "step": t, "ref": carried, "attempt": attempt}
            messages.append({"role": "user", "content": f"[step {t}]", "_ctx": ctx})
            resp = backend.complete(messages, task.schema_view(), call_mode)
            call = parse_response(resp, call_mode)
            ex = execute(task, call.tool or "", call.args or {}, feedback)
            score = score_step(call, gold, ex.schema_valid, ex.known_tool,
                               expected_tool=_expected_tool_given_ref(task, t, carried))
            final_score = score
            if ex.ok:
                executed = True
                if attempt > 0 and score.correct:
                    recovered = True
                carried = ex.output
                stalled = False   # chain is moving again
                break
            messages.append({"role": "user", "content": ex.feedback})
        if not executed:
            # retries exhausted with no successful execution: the chain stalls.
            # carried stays stale, which is a distinct corruption mode from a
            # wrong-value semantic error and is flagged so downstream steps are
            # not mislabelled as fresh semantic failures.
            stalled = True
        records.append(StepRecord(
            task.task_id, task.depth, t, "free", call_mode,
            call.tool if call else None,
            final_score.selection_correct, final_score.selection_matches_gold,
            final_score.args_correct_strict,
            final_score.args_correct_soft, final_score.error_type.value,
            attempts, executed, context_clean, recovered, stalled_in=stalled_in))
    return records


def run_teacher_forced(task: Task, backend: Backend, call_mode: str = "uniform",
                       feedback: FeedbackMode = FeedbackMode.STRUCTURED,
                       max_retries: int = 1) -> list[StepRecord]:
    """Measure p_t: present the correct history at its true length, ask step t.

    Uses the same within-step retry budget as run_free so that p_t and g_t are
    scored under identical rules. Without this, a model that retries its way
    to a correct call in the free run but gets only one shot here would make
    g_t look artificially close to (or above) p_t at shallow depth.
    """
    records: list[StepRecord] = []
    for t, gold in enumerate(task.gold):
        messages = [{"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _task_intro(task)}]
        for j in range(t):
            messages.append({"role": "assistant",
                             "content": json.dumps({"tool": task.gold[j].tool,
                                                    "args": task.gold[j].args})})
            messages.append({"role": "user",
                             "content": f"result: {task.gold[j].output}"})
        correct_ref = task.gold[t].args["ref"]
        attempts = 0
        recovered = False
        final_score = None
        executed = False
        final_call = None
        for attempt in range(max_retries + 1):
            attempts += 1
            ctx = {"task": task, "step": t, "ref": correct_ref, "attempt": attempt}
            messages.append({"role": "user", "content": f"[step {t}]", "_ctx": ctx})
            resp = backend.complete(messages, task.schema_view(), call_mode)
            call = parse_response(resp, call_mode)
            final_call = call
            ex = execute(task, call.tool or "", call.args or {}, feedback)
            score = score_step(call, gold, ex.schema_valid, ex.known_tool,
                               expected_tool=_expected_tool_given_ref(task, t, correct_ref))
            final_score = score
            if ex.ok:
                executed = True
                if attempt > 0 and score.correct:
                    recovered = True
                break
            messages.append({"role": "user", "content": ex.feedback})
        records.append(StepRecord(
            task.task_id, task.depth, t, "teacher_forced", call_mode,
            final_call.tool if final_call else None,
            final_score.selection_correct, final_score.selection_matches_gold,
            final_score.args_correct_strict,
            final_score.args_correct_soft, final_score.error_type.value,
            attempts, executed, True, recovered, stalled_in=False))
    return records


def dump_jsonl(records: list[StepRecord], path: str) -> None:
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
