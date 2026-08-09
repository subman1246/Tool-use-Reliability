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
import random
import re
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

from tur.eval.scoring import ParsedCall, ErrorType, score_step
from tur.harness.cache import Cache
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


class DailyCapReached(RuntimeError):
    """The model's daily request allowance is gone.

    Raised rather than absorbed. Continuing past this point would fill the log
    with backend_error records that look like model failures, producing a
    partial sweep that is indistinguishable from a complete one once it reaches
    the analysis. The caller is expected to stop this model, say so loudly, and
    exit non-zero.
    """


class RateLimiter:
    """Paces requests against a tokens-per-minute ceiling and a daily request cap.

    On free tiers TPM binds long before requests-per-day does: a sweep can sit
    well inside its daily allowance and still be throttled to a crawl. We pace
    against a fraction of the TPM ceiling (`headroom`) rather than riding it,
    because the token estimate is approximate and overshooting just converts
    into provider-side 429s and wasted retries.

    Limits are seeded from config but re-synced from the provider's own
    x-ratelimit-remaining-* response headers whenever they are available, which
    makes the limiter self-correcting: it tracks the provider's accounting
    rather than a static guess that drifts (and Groq's daily allowance is a
    continuously-refilling bucket, not a calendar-day counter).
    """

    def __init__(self, tpm: int | None = None, rpd: int | None = None,
                 headroom: float = 0.80, reserve_requests: int = 5,
                 tpd: int | None = None):
        self.tpm = tpm
        self.rpd = rpd
        # Tokens per day. Enforced by the provider but exposed in NO response
        # header -- it surfaces only in the 429 body -- so this is seeded as a
        # LOWER BOUND from prior observation and corrected upward the moment the
        # provider tells us the real figure (see discovered_tpd).
        self.tpd = tpd
        self.headroom = headroom
        self.reserve_requests = reserve_requests
        self._window: deque[tuple[float, int]] = deque()  # (ts, tokens)
        self.n_requests = 0
        self.tokens_today = 0
        self.discovered_tpd: int | None = None
        self.remaining_requests: int | None = None
        self.remaining_tokens: int | None = None
        self.sleep_seconds = 0.0

    @property
    def budget(self) -> float:
        return (self.tpm or 0) * self.headroom

    def _prune(self, now: float) -> int:
        while self._window and now - self._window[0][0] >= 60.0:
            self._window.popleft()
        return sum(t for _, t in self._window)

    def acquire(self, est_tokens: int) -> None:
        """Block until `est_tokens` fits in the trailing 60s budget."""
        if self.rpd is not None and self.n_requests >= self.rpd:
            raise DailyCapReached(
                f"local request count {self.n_requests} reached the configured "
                f"daily cap {self.rpd}")
        if self.remaining_requests is not None and \
                self.remaining_requests <= self.reserve_requests:
            raise DailyCapReached(
                f"provider reports only {self.remaining_requests} requests "
                f"remaining (reserve={self.reserve_requests})")
        effective_tpd = self.discovered_tpd or self.tpd
        if effective_tpd is not None and \
                self.tokens_today + est_tokens > effective_tpd * self.headroom:
            raise DailyCapReached(
                f"projected token use {self.tokens_today + est_tokens:,} would "
                f"exceed {self.headroom:.0%} of the daily token budget "
                f"{effective_tpd:,}"
                f"{' (discovered)' if self.discovered_tpd else ' (assumed lower bound)'}")
        if self.tpm:
            while True:
                now = time.time()
                used = self._prune(now)
                if used + est_tokens <= self.budget or not self._window:
                    break
                wait = 60.0 - (now - self._window[0][0]) + 0.05
                self.sleep_seconds += max(wait, 0.0)
                time.sleep(max(wait, 0.0))
            self._window.append((time.time(), est_tokens))
        # Counted whether or not a TPM ceiling is configured -- these are the
        # figures the daily caps are checked against.
        self.n_requests += 1
        self.tokens_today += est_tokens

    def sync_from_headers(self, headers: dict) -> None:
        """Adopt the provider's own remaining-quota accounting when exposed."""
        if not headers:
            return
        h = {str(k).lower(): v for k, v in headers.items()}
        rr = h.get("x-ratelimit-remaining-requests")
        rt = h.get("x-ratelimit-remaining-tokens")
        try:
            if rr is not None:
                self.remaining_requests = int(float(rr))
            if rt is not None:
                self.remaining_tokens = int(float(rt))
        except (TypeError, ValueError):
            pass

    def note_discovered_tpd(self, limit: int) -> None:
        """Record a tokens-per-day figure learned from a provider error."""
        if limit and (self.discovered_tpd is None or limit != self.discovered_tpd):
            self.discovered_tpd = int(limit)

    def stats(self) -> dict:
        return {"n_requests": self.n_requests, "tpm": self.tpm, "rpd": self.rpd,
                "tpd_assumed": self.tpd, "tpd_discovered": self.discovered_tpd,
                "tokens_today": self.tokens_today,
                "headroom": self.headroom,
                "remaining_requests": self.remaining_requests,
                "remaining_tokens": self.remaining_tokens,
                "paced_sleep_s": round(self.sleep_seconds, 1)}


def _count_tokens(messages: list[dict], fallback_divisor: int = 4) -> int:
    """Approximate prompt tokens for pacing purposes.

    Exactness is not required -- this feeds a throttle with headroom, not a
    billing figure -- so a tokenizer-free fallback is acceptable.
    """
    text = "".join(str(m.get("content", "")) for m in messages)
    try:
        import tiktoken
        global _ENC
        if _ENC is None:
            _ENC = tiktoken.get_encoding("o200k_base")
        return len(_ENC.encode(text)) + 40
    except Exception:
        return len(text) // fallback_divisor + 40


_ENC = None


# Phrases that indicate an allowance which will NOT clear within any reasonable
# backoff. Deliberately excludes bare "429"/"rate limit", which are usually the
# per-minute ceiling and should be waited out instead.
_DAILY_CAP_MARKERS = ("per day", "requests per day", "rpd", "daily limit",
                      "daily quota", "quota exceeded", "exceeded your current quota",
                      "resource_exhausted", "resource exhausted",
                      "insufficient_quota")


_TPD_LIMIT_RE = re.compile(r"limit\s+(\d[\d,_]*)", re.I)


def _parse_daily_token_limit(err: Exception) -> int | None:
    """Extract the tokens-per-day figure from a provider rate-limit message.

    Groq's 429 body is the only place this number appears -- it is absent from
    every x-ratelimit-* header and from the published docs -- and it reads e.g.
    "on tokens per day (TPD): Limit 200000, Used 199895, Requested 2890".
    Capturing it turns an opaque failure into a measured constraint that the
    next day's run can plan against.
    """
    msg = str(err)
    low = msg.lower()
    if "token" not in low or not any(k in low for k in ("per day", "tpd")):
        return None
    m = _TPD_LIMIT_RE.search(msg)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", "").replace("_", ""))
    except ValueError:
        return None


def _is_daily_cap_error(err: Exception) -> bool:
    """Distinguish an exhausted daily allowance from a transient per-minute 429.

    A per-minute 429 should be waited out; a daily one will not clear within any
    reasonable backoff, so retrying it just burns whatever allowance is left on
    calls that cannot succeed. Matching is on the marker phrases alone -- an
    earlier version also required "429"/"rate"/"quota" to appear, which silently
    failed to catch Google's bare RESOURCE_EXHAUSTED.
    """
    msg = str(err).lower()
    return any(m in msg for m in _DAILY_CAP_MARKERS)


class LiteLLMBackend:
    """Provider-agnostic backend via LiteLLM. Imported lazily.

    Wraps every call with retry-with-backoff for transient network/rate-limit
    errors. This is distinct from the harness's own within-step retry (which
    handles syntactic failures by re-prompting the model) -- this layer
    handles the call to the provider itself not going through at all.
    Without it, a single rate-limit blip partway through a real run kills the
    whole batch and wastes every call made so far that wasn't yet cached.
    """

    def __init__(self, model: str, temperature: float = 0.0,
                max_retries: int = 5, base_delay: float = 1.0,
                timeout: float = 60.0, cache: "Cache | None" = None,
                limiter: "RateLimiter | None" = None):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self.cache = cache
        self.limiter = limiter
        self.n_calls = 0
        self.n_cache_hits = 0
        self.n_retries = 0
        self.n_failures = 0
        self.n_rate_limited = 0

    def complete(self, messages, tools, mode):
        clean = [{k: v for k, v in m.items() if not k.startswith("_")}
                 for m in messages]
        cache_key = None
        if self.cache is not None:
            cache_key = Cache.key(self.model, mode, clean, tools)
            hit = self.cache.get(cache_key)
            if hit is not None:
                self.n_cache_hits += 1
                return hit

        import litellm
        kwargs: dict[str, Any] = dict(model=self.model, messages=clean,
                                      temperature=self.temperature,
                                      timeout=self.timeout)
        if mode == "native" and tools:
            kwargs["tools"] = [{"type": "function",
                                "function": {"name": t["name"],
                                             "description": t["description"],
                                             "parameters": _json_schema(t)}}
                               for t in tools]

        est_tokens = _count_tokens(clean) if self.limiter else 0

        last_err = None
        for attempt in range(self.max_retries + 1):
            # Pace before every attempt, including retries -- a retry is a real
            # request and counts against the same allowance. DailyCapReached
            # propagates deliberately; it is not a transient condition.
            if self.limiter is not None:
                self.limiter.acquire(est_tokens)
            self.n_calls += 1
            try:
                resp = litellm.completion(**kwargs)
                if self.limiter is not None:
                    self.limiter.sync_from_headers(_response_headers(resp))
                msg = resp["choices"][0]["message"]
                if mode == "native" and msg.get("tool_calls"):
                    tc = msg["tool_calls"][0]["function"]
                    result = {"tool_call": {"name": tc["name"],
                                            "arguments": tc["arguments"]}}
                else:
                    result = {"text": msg.get("content") or ""}
                if self.cache is not None:
                    self.cache.set(cache_key, result)
                return result
            except DailyCapReached:
                raise
            except Exception as e:  # noqa: broad except is intentional here --
                # litellm normalises errors across providers into its own
                # exception hierarchy, but network/timeout errors from the
                # underlying transport can still leak through untyped.
                last_err = e
                if _is_daily_cap_error(e):
                    # No backoff will clear this; retrying spends the little
                    # allowance left on calls that cannot succeed. Capture the
                    # daily token limit if the provider named it -- this is the
                    # only place it is ever stated.
                    if self.limiter is not None:
                        found = _parse_daily_token_limit(e)
                        if found:
                            self.limiter.note_discovered_tpd(found)
                    raise DailyCapReached(
                        f"provider signalled a daily/quota limit for "
                        f"{self.model}: {e}") from e
                if "429" in str(e):
                    self.n_rate_limited += 1
                if attempt < self.max_retries:
                    self.n_retries += 1
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(delay)
                else:
                    self.n_failures += 1
        # exhausted retries: return a sentinel the parser will treat as a
        # syntactic failure (empty/unparseable), so it is scored and logged
        # rather than crashing the whole run. Distinguishable in logs via the
        # "_backend_error" marker.
        return {"text": "", "_backend_error": str(last_err)}

    def stats(self) -> dict:
        s = {"model": self.model, "n_calls": self.n_calls,
             "n_cache_hits": self.n_cache_hits, "n_retries": self.n_retries,
             "n_failures": self.n_failures,
             "n_rate_limited": self.n_rate_limited}
        if self.limiter is not None:
            s["limiter"] = self.limiter.stats()
        return s


def _response_headers(resp: Any) -> dict:
    """Best-effort extraction of provider response headers from a LiteLLM
    response. LiteLLM stashes these inconsistently across versions and
    providers, so every access is guarded and a miss is simply no sync."""
    for getter in (
        lambda: resp._hidden_params.get("additional_headers"),
        lambda: resp._hidden_params.get("response_headers"),
        lambda: resp._response_headers,
    ):
        try:
            h = getter()
            if h:
                return dict(h)
        except Exception:
            continue
    return {}


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
    if "_backend_error" in resp:
        # the call to the provider never succeeded (exhausted retries); this
        # is a distinct failure mode from the model producing a bad response,
        # and is tagged in raw so it can be filtered out of f_syn/error-type
        # analysis rather than counted as a genuine model mistake.
        return ParsedCall(None, None, parse_ok=False,
                          raw=f"[BACKEND_ERROR] {resp['_backend_error']}")
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


_REASONING_CLOSE = re.compile(r"</think(?:ing)?>", re.I)
_REASONING_OPEN = re.compile(r"<think(?:ing)?>", re.I)


def _strip_reasoning(text: str) -> str:
    """Drop a reasoning preamble so it can't be mistaken for the answer.

    Reasoning models (qwen3.6 here) emit a <think> block before the call. That
    block routinely contains braces -- draft JSON, dict literals, prose about
    the schema -- so any brace-based extraction that sees it will splice
    reasoning into the parsed call. Everything after the LAST close tag is the
    answer. An unclosed block means the response was truncated mid-reasoning
    and there is no answer to find, which is a genuine failure, not a parsing
    artifact, so we leave nothing behind for the scanner to latch onto.
    """
    if _REASONING_CLOSE.search(text):
        return _REASONING_CLOSE.split(text)[-1]
    if _REASONING_OPEN.search(text):
        return _REASONING_OPEN.split(text)[0]
    return text


def _top_level_objects(text: str) -> list[str]:
    """Every balanced, top-level {...} span, in order of appearance.

    Brace counting is string-aware: braces inside JSON string values, and
    escaped quotes within them, do not affect depth. Nested objects are not
    returned separately -- only spans that open and close at depth zero -- so
    the args sub-object of a call is never mistaken for the call itself.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escaped = False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    spans.append(text[start:i + 1])
                    start = -1
    return spans


def _extract_json(text: str) -> str:
    """Pull the model's call out of a free-text response.

    Returns the LAST balanced top-level object that actually parses, rather
    than the old first-brace-to-last-brace slice. That slice was wrong in two
    common real-model cases: a reasoning preamble containing braces (it would
    span from a brace in the reasoning to the last brace of the answer, parsing
    as neither), and any response with more than one JSON object in it. Taking
    the last parseable object also handles markdown code fences for free, since
    the fence markers sit outside the braces.

    Preferring the last object reflects answer-after-reasoning ordering. It can
    in principle pick up a trailing commentary object if that object is itself
    valid JSON; that is rarer than the failure modes it fixes. Nothing here
    inspects the object's contents, so an unparseable response is still a parse
    failure and is still recorded as one.
    """
    cleaned = _strip_reasoning(text)
    for cand in reversed(_top_level_objects(cleaned)):
        try:
            json.loads(cand)
            return cand
        except json.JSONDecodeError:
            continue
    # Nothing balanced and parseable: hand back the cleaned text so the caller's
    # json.loads fails and the step is logged as a syntactic failure.
    return cleaned


_MAX_ECHO_CHARS = 2000


def _assistant_turn(call: ParsedCall) -> str:
    """Render the model's own turn for the conversation history.

    A well-formed call is echoed in canonical form; anything else is echoed
    verbatim so the model can see what it actually emitted and, on a retry,
    correct it. Truncated because a runaway response would otherwise grow the
    context for every later step of the task.
    """
    if call.parse_ok:
        return json.dumps({"tool": call.tool, "args": call.args})
    return (call.raw or "")[:_MAX_ECHO_CHARS]


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
    backend_error: bool = False      # the provider call itself failed, not the model


# --------------------------- prompt ---------------------------

_SYSTEM = ("You are a tool-using agent. At each step call exactly one tool. "
           "Respond in the requested format only.")


def _render_schema(tools: list[dict], style: str = "verbose") -> str:
    """Serialise the tool schema for the prompt.

    Two renderings carrying IDENTICAL information -- every tool name, parameter
    name, parameter type, requiredness, and the full description text:

      verbose : the original indented JSON dump.
      compact : one line per tool, `- name(param:type, other:type?): description`,
                where a trailing ? marks an optional parameter.

    This exists because the schema is re-sent with every API call (the provider
    is stateless, so each call carries the whole conversation, and the intro
    sits at the head of it). At depth 8 the schema is roughly three quarters of
    all tokens spent, so its serialisation -- not the number of steps -- is the
    dominant cost driver. Switching rendering changes what the model sees, so
    the choice is validated empirically rather than assumed neutral.
    """
    if style == "compact":
        lines = []
        for t in tools:
            params = ", ".join(
                f"{n}:{meta['type']}" + ("" if meta["required"] else "?")
                for n, meta in t["parameters"].items())
            lines.append(f"- {t['name']}({params}): {t['description']}")
        return "\n".join(lines)
    if style != "verbose":
        raise ValueError(f"unknown schema style {style!r}")
    return json.dumps(tools, indent=0)


def _task_intro(task, schema_style: str = "verbose") -> str:
    schema = _render_schema(task.schema_view(), schema_style)
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
             max_retries: int = 1,
             schema_style: str = "verbose") -> list[StepRecord]:
    records: list[StepRecord] = []
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _task_intro(task, schema_style)}]
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
            # Echo the model's own turn back into the history. Without this the
            # conversation carries no record of what was called or what came
            # back, so at step t the model is asked for a ref it was never
            # shown and can only guess -- g_t then measures the harness, not
            # propagation. Simulated backends never caught this because they
            # read the carried ref from _ctx instead of from the history.
            messages.append({"role": "assistant",
                             "content": _assistant_turn(call)})
            if ex.ok:
                executed = True
                if attempt > 0 and score.correct:
                    recovered = True
                carried = ex.output
                stalled = False   # chain is moving again
                # The observation is the model's OWN result, right or wrong.
                # That is exactly the channel a semantic error propagates
                # through: a plausible number that is not the gold one.
                messages.append({"role": "user", "content": f"result: {ex.output}"})
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
            attempts, executed, context_clean, recovered, stalled_in=stalled_in,
            backend_error=bool(call and call.is_backend_error)))
    return records


def run_teacher_forced(task: Task, backend: Backend, call_mode: str = "uniform",
                       feedback: FeedbackMode = FeedbackMode.STRUCTURED,
                       max_retries: int = 1,
                       schema_style: str = "verbose") -> list[StepRecord]:
    """Measure p_t: present the correct history at its true length, ask step t.

    Uses the same within-step retry budget as run_free so that p_t and g_t are
    scored under identical rules. Without this, a model that retries its way
    to a correct call in the free run but gets only one shot here would make
    g_t look artificially close to (or above) p_t at shallow depth.
    """
    records: list[StepRecord] = []
    for t, gold in enumerate(task.gold):
        messages = [{"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _task_intro(task, schema_style)}]
        for j in range(t):
            # The [step j] marker is included so the clean history is
            # structurally identical to a free run's history at the same depth.
            # p_t and g_t must differ only in whether that history is CORRECT;
            # if one mode also carries extra turns the other lacks, the
            # comparison picks up a prompt-shape difference as well.
            messages.append({"role": "user", "content": f"[step {j}]"})
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
            attempts, executed, True, recovered, stalled_in=False,
            backend_error=bool(final_call and final_call.is_backend_error)))
    return records


def dump_jsonl(records: list[StepRecord], path: str) -> None:
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
