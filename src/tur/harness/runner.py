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
from dataclasses import asdict, dataclass, replace
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
                choice = resp["choices"][0]
                msg = choice["message"]
                if mode == "native" and msg.get("tool_calls"):
                    tc = msg["tool_calls"][0]["function"]
                    result = {"tool_call": {"name": tc["name"],
                                            "arguments": tc["arguments"]}}
                else:
                    result = {"text": msg.get("content") or ""}
                # finish_reason is recorded because it is the only DIRECT evidence
                # that a completion was cut off at a token ceiling rather than
                # ending naturally. Without it, truncation can only be inferred
                # structurally (an unclosed reasoning tag, a response not ending
                # in a closing brace), and a truncated completion is a third thing
                # distinct from both a parse failure and a model error: it is a
                # harness or provider configuration fault that merely LOOKS like
                # malformed output. Storing it costs one field.
                try:
                    result["finish_reason"] = choice.get("finish_reason")
                except AttributeError:
                    result["finish_reason"] = None
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
    # Conditional-on-state scoring, recorded alongside gold-agreement rather than
    # instead of it. Gold-agreement pins severity at 1 and hides recovery, because
    # after divergence the gold value is unreachable; these two ask whether the call
    # correctly continued from the value the agent actually held.
    args_correct_given_state: bool = False
    correct_given_state: bool = False
    # The value actually carried INTO this step. Recorded because the routing rule
    # conditions on it, so any analysis of whether a model is following the rule or
    # defaulting to a fixed branch needs it. It was previously recoverable only by
    # replaying the whole sweep against the response cache.
    held_ref: int | None = None
    # None means UNKNOWN, not False. Completions cached before finish_reason was
    # recorded cannot say whether they were truncated, and encoding that as False
    # would assert a verification that never happened. True means the provider
    # reported finish_reason == "length": the completion was cut at a token
    # ceiling, which is a harness/provider configuration fault that merely looks
    # like malformed model output and must not be counted as either a parse
    # failure or a model error.
    truncated: bool | None = None
    # Which branch was presented first in the rule text at this step. With a fixed
    # presentation order, "picks the first-listed tool" and "applies the rule
    # correctly for even refs" predict overlapping data; recording the order lets
    # the two be separated directly rather than only via a discrimination statistic.
    first_listed_even: bool = True
    # Controlled error injection. None on every other run mode. `pair_id` ties the
    # clean and corrupted members of one injection pair together: the pairing is a
    # property of how the two prompts were BUILT (identical prefix, one substituted
    # value), so it is recorded at construction rather than reconstructed later by
    # matching on task_id and step, which would silently pair across corruption modes.
    inject_at: int | None = None
    injection: str | None = None
    pair_id: str | None = None
    # Recovery arm. None/False on every other run mode.
    used_resync: bool = False        # this step WAS the resync call
    resync_remaining: int | None = None
    resync_charged: bool | None = None   # did this resync consume the repair budget
    diverged_in: bool = False        # held value differed from canonical entering this step
    gold_ref: int | None = None      # canonical value for this step, for the three-way split
    # The value carried OUT of this step. held_ref is the value carried IN, matching
    # run_free's `carried_in`; recording only that would leave the final state of a
    # trajectory unrecoverable, which is exactly what the recovery verdict needs.
    held_out: int | None = None


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


def _first_listed_even(task, step: int) -> bool:
    """Was the even-ref branch listed first in the rule text at this step?"""
    order = getattr(task, "present_odd_first", None)
    if not order:
        return True
    return not order[step]


def _expected_args_given_held(task, step: int, held: int) -> dict:
    """What a correct call sends GIVEN the value the agent actually holds.

    On the copy-argument variant this is the held value itself; on the
    transformed-argument variant it is the stated function of it. Used for
    conditional-on-state scoring, which is what lets severity and recovery take
    interior values instead of being pinned by gold-agreement scoring.
    """
    shift = getattr(task, "arg_shift", 0)
    if not shift or step == 0:
        return {"ref": held}
    return {"ref": (held + shift) % 100000}


def _task_intro(task, schema_style: str = "verbose") -> str:
    schema = _render_schema(task.schema_view(), schema_style)
    if hasattr(task, "routing_rule_text"):  # RoutingTask
        arg_rule = getattr(task, "arg_rule_text", lambda: "")()
        if arg_rule:
            # transformed-argument variant: the ref sent is a function of the
            # previous result rather than a copy of it, so state that instead of
            # the copy rule rather than in addition to it
            ref_rule = arg_rule
        else:
            ref_rule = ("Each later step's ref equals the numeric result "
                        "returned by the previous tool.")
        # The repair rule, when the task has one, is stated HERE rather than as its own
        # message. Appended as the last turn before [step 0] it sat immediately before the
        # first decision, and a 7B model reflexively spent its one repair on step 0 in 14
        # of 16 pilot tasks -- where the held value equals the canonical value by
        # construction, so the call returned the number already in the prompt. Stating it
        # inside the task description, with the output format still last, describes the
        # tool without cueing it.
        # Inserted as its own paragraph ONLY when the task has a repair rule. A task
        # without one must produce a byte-identical intro to before this change:
        # otherwise every routing and injection prompt shifts, which would invalidate the
        # response cache and make the recovery arm incomparable to the frozen runs it
        # exists to be compared against.
        recovery_rule = getattr(task, "recovery_rule_text", lambda: "")()
        tail = (f"\n\n{recovery_rule}\n\n" if recovery_rule else " ")
        return (f"Tools available:\n{schema}\n\n"
                f"Perform {task.depth} steps. At each step, choose the tool "
                f"according to this rule, applied to the incoming ref value:\n"
                f"{task.routing_rule_text()}\n\n"
                f"The first step's ref={task.seed_value}. {ref_rule}{tail}"
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
        # "Clean context" is a property of the value CARRIED IN, which is the
        # previous tool's gold output -- not of the argument the model is supposed
        # to send. Those coincide only when the required argument is a verbatim
        # copy of the previous result. On a task variant where the argument is a
        # stated transformation of it, comparing the carried value against the
        # expected ARGUMENT would mark every clean step as poisoned.
        expected_carry = task.seed_value if t == 0 else task.gold[t - 1].output
        carried_in = carried      # `carried` is overwritten by execution below
        stalled_in = stalled
        context_clean = (carried == expected_carry) and not stalled
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
            fr = resp.get("finish_reason") if isinstance(resp, dict) else None
            was_truncated = (fr == "length") if fr is not None else None
            call = parse_response(resp, call_mode)
            ex = execute(task, call.tool or "", call.args or {}, feedback)
            score = score_step(
                call, gold, ex.schema_valid, ex.known_tool,
                expected_tool=_expected_tool_given_ref(task, t, carried),
                expected_args=_expected_args_given_held(task, t, carried))
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
            args_correct_given_state=final_score.args_correct_given_state,
            correct_given_state=final_score.correct_given_state,
            held_ref=carried_in,
            first_listed_even=_first_listed_even(task, t),
            truncated=was_truncated,
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
        # The policy is handed the value CARRIED IN, matching run_free's `carried`,
        # so a simulated policy sees the same kind of quantity in both run modes.
        # Under a transformed-argument variant this is not the argument itself.
        correct_ref = task.seed_value if t == 0 else task.gold[t - 1].output
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
            fr = resp.get("finish_reason") if isinstance(resp, dict) else None
            was_truncated = (fr == "length") if fr is not None else None
            call = parse_response(resp, call_mode)
            final_call = call
            ex = execute(task, call.tool or "", call.args or {}, feedback)
            score = score_step(
                call, gold, ex.schema_valid, ex.known_tool,
                expected_tool=_expected_tool_given_ref(task, t, correct_ref),
                expected_args=_expected_args_given_held(task, t, correct_ref))
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
            backend_error=bool(final_call and final_call.is_backend_error),
            args_correct_given_state=final_score.args_correct_given_state,
            correct_given_state=final_score.correct_given_state,
            held_ref=correct_ref,
            first_listed_even=_first_listed_even(task, t),
            truncated=was_truncated))
    return records


def dump_jsonl(records: list[StepRecord], path: str) -> None:
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


# --------------------- controlled error injection ---------------------
#
# What the observational arms cannot do. In the free run a corrupted context is
# something the model PRODUCED, so "models fail more after a corrupted context" is
# confounded: whatever made the model err at step j-1 (a hard branch, an unlucky
# value, a weak moment) is still present at step j. Severity is therefore estimated
# from data in which treatment assignment is decided by the model itself. That is the
# identifiability problem the paper's hierarchical fit runs into from the other side.
#
# Injection removes the confound by construction. Both members of a pair are built
# from a byte-identical gold prefix and differ in exactly one substituted number: the
# result reported for step inject_at-1. The model did not choose to be corrupted, so
# the difference in step-inject_at accuracy between the two members is a causal
# effect of context corruption rather than an association with it.
#
# Scoring is conditional-on-state, not gold-agreement. Under a corrupted ref the gold
# tool is no longer the correct answer -- an agent applying the routing rule perfectly
# to the value it holds will legitimately depart from gold -- so both members are
# scored against _expected_tool_given_ref / _expected_args_given_held at the value
# actually held. Scoring the corrupted member against gold would guarantee failure and
# measure nothing but the substitution.
#
# Two corruption modes, because they load different channels:
#
#   parity_flip         the substituted value has OPPOSITE parity, so the routing rule
#                       selects the other branch. The correct TOOL changes. This is the
#                       selection channel, and it is the mode that mimics a real
#                       propagated error in the routing task.
#   parity_preserving   the substituted value has the SAME parity, so the correct tool
#                       is unchanged and only the argument to send differs. This is the
#                       argument channel.
#
# Running both separates "a corrupted context degrades rule application" from "a
# corrupted context degrades value transcription". A single corruption mode cannot:
# parity_flip alone confounds the two, since it changes tool and argument together.

_INJ_CLEAN = "inj_clean"
_INJ_CORRUPT = "inj_corrupt"
_INJECTIONS = ("parity_flip", "parity_preserving")


def _corrupt_value(value: int, mode: str, seed: int) -> int:
    """Substitute a wrong value for `value` under the named corruption mode.

    MOD is even, so reducing mod MOD preserves the parity of the sum. An odd delta
    therefore always flips parity and an even delta always keeps it, with no need to
    check the result and resample.
    """
    from tur.tasks.dag import MOD
    rng = random.Random(seed)
    if mode == "parity_flip":
        delta = rng.randrange(1, MOD, 2)        # odd  -> parity flips
    elif mode == "parity_preserving":
        delta = rng.randrange(2, MOD, 2)        # even -> parity kept, value differs
    else:
        raise ValueError("unknown injection mode " + repr(mode))
    return (int(value) + delta) % MOD


def run_injection_pair(task, backend: Backend, inject_at: int,
                       corruption: str = "parity_flip",
                       call_mode: str = "uniform",
                       feedback: FeedbackMode = FeedbackMode.STRUCTURED,
                       max_retries: int = 1,
                       schema_style: str = "verbose") -> list[StepRecord]:
    """Run one clean/corrupted pair at `inject_at` and return both records.

    Both conditions are run here rather than in two passes so that the identical
    prefix is guaranteed by construction: the two message lists are built in the same
    call, from the same task object, and provably differ in one element.
    """
    if not hasattr(task, "branches"):
        raise ValueError("error injection requires a RoutingTask: on a linear task "
                         "the tool order is announced in the prompt, so corrupting a "
                         "value cannot change which tool is correct and the selection "
                         "channel the injection is meant to load does not exist.")
    if not 1 <= inject_at < task.depth:
        raise ValueError("inject_at must satisfy 1 <= inject_at < depth (" +
                         str(task.depth) + "); got " + str(inject_at) +
                         ". Step 0 has no previous result to corrupt.")
    if corruption not in _INJECTIONS:
        raise ValueError("unknown injection mode " + repr(corruption))

    true_val = task.gold[inject_at - 1].output
    # crc32 of a stable string, NOT hash(): hash() is randomised per interpreter by
    # PYTHONHASHSEED, which would make the injected value differ between a run and its
    # replay and quietly break cache reuse and reproducibility.
    import zlib
    seed = zlib.crc32((task.task_id + "|" + str(inject_at) + "|" + corruption).encode())
    bad_val = _corrupt_value(true_val, corruption, seed)
    pair_id = task.task_id + "|j" + str(inject_at) + "|" + corruption

    records: list[StepRecord] = []
    for mode, held in ((_INJ_CLEAN, true_val), (_INJ_CORRUPT, bad_val)):
        messages = [{"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _task_intro(task, schema_style)}]
        for j in range(inject_at):
            messages.append({"role": "user", "content": "[step " + str(j) + "]"})
            messages.append({"role": "assistant",
                             "content": json.dumps({"tool": task.gold[j].tool,
                                                    "args": task.gold[j].args})})
            # THE ONLY DIFFERENCE between the two members of the pair is this value,
            # and only on the final iteration of this loop.
            shown = held if j == inject_at - 1 else task.gold[j].output
            messages.append({"role": "user", "content": "result: " + str(shown)})

        gold = task.gold[inject_at]
        attempts = 0
        recovered = False
        final_score = None
        executed = False
        final_call = None
        for attempt in range(max_retries + 1):
            attempts += 1
            ctx = {"task": task, "step": inject_at, "ref": held, "attempt": attempt}
            messages.append({"role": "user", "content": "[step " + str(inject_at) + "]",
                             "_ctx": ctx})
            resp = backend.complete(messages, task.schema_view(), call_mode)
            fr = resp.get("finish_reason") if isinstance(resp, dict) else None
            was_truncated = (fr == "length") if fr is not None else None
            call = parse_response(resp, call_mode)
            final_call = call
            ex = execute(task, call.tool or "", call.args or {}, feedback)
            score = score_step(
                call, gold, ex.schema_valid, ex.known_tool,
                expected_tool=_expected_tool_given_ref(task, inject_at, held),
                expected_args=_expected_args_given_held(task, inject_at, held))
            final_score = score
            if ex.ok:
                executed = True
                if attempt > 0 and score.correct:
                    recovered = True
                break
            messages.append({"role": "user", "content": ex.feedback})

        records.append(StepRecord(
            task.task_id, task.depth, inject_at, mode, call_mode,
            final_call.tool if final_call else None,
            final_score.selection_correct, final_score.selection_matches_gold,
            final_score.args_correct_strict,
            final_score.args_correct_soft, final_score.error_type.value,
            attempts, executed,
            context_clean_in=(mode == _INJ_CLEAN),
            recovered=recovered, stalled_in=False,
            backend_error=bool(final_call and final_call.is_backend_error),
            args_correct_given_state=final_score.args_correct_given_state,
            correct_given_state=final_score.correct_given_state,
            held_ref=held,
            first_listed_even=_first_listed_even(task, inject_at),
            truncated=was_truncated,
            inject_at=inject_at, injection=corruption, pair_id=pair_id))
    return records


# --------------------------- recovery arm ---------------------------
#
# Scores the same trajectory THREE ways, which is the entire point of the arm. Section 7
# says the project cannot currently distinguish "continuing competently from an off-gold
# state" from "actually recovering". Those two differ only when the task offers a route
# back, so until now they have been the same number wearing two names.
#
#   canonical gold agreement   did the call match the gold trajectory
#                              (selection_matches_gold + args_correct_strict)
#   conditional-on-state       was the call correct GIVEN the value actually held
#                              (correct_given_state) -- the existing remedy
#   genuine recovery           did the trajectory get back to the canonical state and
#                              finish the ORIGINAL task, verified against the task's true
#                              final output, not merely continued plausibly
#
# The third is a TRAJECTORY-level property, not a per-call one, so it is computed by
# recovery_outcome() over a finished task rather than recorded per step.
#
# check_state is intercepted here rather than dispatched through executor.execute,
# because its value depends on which step the agent is on and ToolSpec.run is a pure
# function of its arguments. See tur/tasks/recovery.py.


def run_recovery(task, backend: Backend, call_mode: str = "uniform",
                 feedback: FeedbackMode = FeedbackMode.STRUCTURED,
                 max_retries: int = 1,
                 schema_style: str = "verbose") -> list[StepRecord]:
    """Free-running, but the agent may call check_state once to repair its state."""
    from tur.tasks.recovery import RESYNC_TOOL, canonical_ref_at

    records: list[StepRecord] = []
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _task_intro(task, schema_style)}]
    held = task.seed_value
    remaining = getattr(task, "resync_budget", 1)
    t = 0
    guard = 0
    free_used_this_step = False
    # A resync does not advance the step, so the loop is bounded by calls rather than by
    # steps: depth steps, plus one free no-op resync per step (see below), plus the
    # charged repairs allowed, plus slack for a refused resync being retried once. Without
    # it a model that emits check_state forever would spin.
    max_calls = 2 * task.depth + getattr(task, "resync_budget", 1) + 2

    while t < task.depth and guard < max_calls:
        guard += 1
        gold_ref = canonical_ref_at(task, t)
        diverged_in = (held != gold_ref)
        held_in = held          # snapshot: `held` is overwritten by execution below

        attempts = 0
        recovered_syn = False
        final_score = None
        executed = False
        final_call = None
        was_resync = False

        for attempt in range(max_retries + 1):
            attempts += 1
            ctx = {"task": task, "step": t, "ref": held, "attempt": attempt}
            messages.append({"role": "user", "content": "[step " + str(t) + "]",
                             "_ctx": ctx})
            resp = backend.complete(messages, task.schema_view(), call_mode)
            fr = resp.get("finish_reason") if isinstance(resp, dict) else None
            was_truncated = (fr == "length") if fr is not None else None
            call = parse_response(resp, call_mode)
            final_call = call

            if call.parse_ok and call.tool == RESYNC_TOOL:
                was_resync = True
                # A resync issued while the held value is ALREADY canonical repairs
                # nothing: it returns the number the agent is holding. Charging the budget
                # for it measured the model's prior about when to ask rather than its
                # ability to recover, and in the pilot it consumed every repair at step 0,
                # where divergence is impossible by construction. So a no-op resync is
                # free -- but only once per step, otherwise a model that emits check_state
                # unconditionally would never advance and never pay.
                #
                # This leaks nothing the agent can act on: the reply is "result: <value>"
                # whether or not the budget moved, and the moment the agent is genuinely
                # diverged the repair costs its one shot as before.
                no_op = (held == gold_ref)
                if no_op and not free_used_this_step:
                    free_used_this_step = True
                    charged, granted = False, True
                elif remaining > 0:
                    remaining -= 1
                    charged, granted = True, True
                else:
                    charged, granted = False, False

                if granted:
                    held = gold_ref          # state repaired: back on the canonical value
                    messages.append({"role": "assistant",
                                     "content": json.dumps({"tool": RESYNC_TOOL,
                                                            "args": {}})})
                    messages.append({"role": "user",
                                     "content": "result: " + str(gold_ref)})
                else:
                    messages.append({"role": "user",
                                     "content": "ToolError: check_state budget exhausted"})
                executed = True
                break

            ex = execute(task, call.tool or "", call.args or {}, feedback)
            score = score_step(
                call, task.gold[t], ex.schema_valid, ex.known_tool,
                expected_tool=_expected_tool_given_ref(task, t, held),
                expected_args=_expected_args_given_held(task, t, held))
            final_score = score
            if ex.ok:
                executed = True
                if attempt > 0 and score.correct:
                    recovered_syn = True
                messages.append({"role": "assistant",
                                 "content": json.dumps({"tool": call.tool,
                                                        "args": call.args})})
                messages.append({"role": "user", "content": ex.feedback})
                held = ex.output
                break
            messages.append({"role": "user", "content": ex.feedback})

        if was_resync:
            # The resync call itself is recorded so the cost is visible in the data, but
            # it is not scored as a step attempt: it is not an answer to the step.
            records.append(StepRecord(
                task.task_id, task.depth, t, "recovery", call_mode, RESYNC_TOOL,
                False, False, False, False, "resync", attempts, True, not diverged_in,
                False, stalled_in=False,
                backend_error=bool(final_call and final_call.is_backend_error),
                args_correct_given_state=False, correct_given_state=False,
                held_ref=held_in, first_listed_even=_first_listed_even(task, t),
                truncated=None, used_resync=True, resync_remaining=remaining,
                resync_charged=charged,
                diverged_in=diverged_in, gold_ref=gold_ref, held_out=held))
            continue        # same step, retried with repaired state

        if final_score is None:
            break
        records.append(StepRecord(
            task.task_id, task.depth, t, "recovery", call_mode,
            final_call.tool if final_call else None,
            final_score.selection_correct, final_score.selection_matches_gold,
            final_score.args_correct_strict, final_score.args_correct_soft,
            final_score.error_type.value, attempts, executed,
            context_clean_in=not diverged_in, recovered=recovered_syn, stalled_in=False,
            backend_error=bool(final_call and final_call.is_backend_error),
            args_correct_given_state=final_score.args_correct_given_state,
            correct_given_state=final_score.correct_given_state,
            held_ref=held_in, first_listed_even=_first_listed_even(task, t),
            truncated=was_truncated, used_resync=False, resync_remaining=remaining,
            diverged_in=diverged_in, gold_ref=gold_ref, held_out=held))
        t += 1
        free_used_this_step = False

    return records


def recovery_outcome(task, records: list[StepRecord]) -> dict:
    """Trajectory-level three-way verdict for one recovery task.

    `recovered` is the strict reading and the one the paper needs: the trajectory left the
    canonical path at some point AND finished holding the task's true final value. Merely
    ending on a value that is self-consistent with the agent's own wrong history does not
    count -- that is the confusion this whole arm exists to remove.
    """
    steps = [r for r in records if not r.used_resync]
    if not steps:
        return {"task_id": task.task_id, "depth": task.depth, "completed": False,
                "diverged": False, "recovered": False, "used_resync": False,
                "gold_agreement": 0.0, "conditional": 0.0}
    diverged = any(r.diverged_in for r in records)
    used = any(r.used_resync for r in records)
    completed = len(steps) == task.depth
    final_gold = task.gold[-1].output
    # the value carried out of the last scored step
    final_held = steps[-1].held_out
    recovered = bool(diverged and completed and final_held == final_gold)
    return {
        "task_id": task.task_id, "depth": task.depth, "completed": completed,
        "diverged": diverged, "used_resync": used, "recovered": recovered,
        "gold_agreement": sum(1 for r in steps
                              if r.selection_matches_gold and r.args_correct_strict)
                          / len(steps),
        "conditional": sum(1 for r in steps if r.correct_given_state) / len(steps),
        "n_steps": len(steps), "n_resync": sum(1 for r in records if r.used_resync),
    }


# --------------------------- repaired-trajectory arm ---------------------------
#
# WHY THIS REPLACED THE ELECTED-REPAIR ARM
#
# The recovery arm above offers check_state and lets the model decide when to use it. Two
# pilots on allam-2-7b showed that the decision is not made on evidence: with the rule
# stated as the last turn before step 0, the model spent its single repair immediately in
# 14 of 16 tasks -- at the one step where the held value equals the canonical value by
# construction, so every repair returned the number already in the prompt. With the same
# rule stated inside the task description, it never called the tool at all, in 16 of 16.
# Zero repairs landed on a diverged step either way. The model has no divergence signal in
# this task, so its only coherent policies are "always ask" and "never ask", and prompt
# salience picks between them. Raising the budget would just select "always ask", which
# collapses the free-running arm into the teacher-forced one.
#
# So repair is assigned here rather than elected, the same way corruption is assigned in
# the injection arm. That isolates the question Section 7 actually asks -- given a route
# back, does the agent return to the goal, or does it merely continue competently from
# where it stands -- from the question the elected arm kept answering instead, which is
# whether the model knows to ask.
#
# WHY IT IS NOT APPENDIX E AGAIN
#
# Appendix E's corrected branch already establishes that handing back the true value
# restores accuracy ON THE NEXT CALL (+0.000 against baseline at j=1,3,5). Re-measuring
# one call would duplicate it. This arm therefore scores the trajectory to the END of the
# task: after the repair the model runs free for the remaining steps, and the verdict is
# whether it finishes holding the task's TRUE final output. Returning to the canonical
# value for one call and drifting off again is not recovery, and only a downstream-to-
# completion measure can tell the two apart. repair_at is constrained so at least two
# steps remain to be scored, which is what makes the measure a trajectory property rather
# than Appendix E with extra steps.
#
# CONSTRUCTION
#
# One shared, genuinely free prefix, then a fork that differs in exactly one value:
#
#   1. teacher-forced gold up to inject_at, with the step inject_at-1 result corrupted
#   2. FREE running from inject_at to repair_at-1, run ONCE so both branches inherit the
#      identical history, including whatever the model did wrong in it
#   3. at repair_at both branches receive the same check_state exchange. The repaired
#      branch is handed the canonical value; the unrepaired branch is handed the value it
#      is already carrying, so its "repair" is a no-op by construction. The two message
#      lists differ in exactly one number, same discipline as run_injection_pair.
#   4. FREE running to the end of the task on each branch, scored to completion

_REP_REPAIRED = "rep_repaired"
_REP_UNREPAIRED = "rep_unrepaired"


def _free_segment(task, backend, messages, held, start, stop, mode, call_mode,
                  feedback, max_retries, pair_id, inject_at, corruption,
                  records, gold_ref_of):
    """Run steps [start, stop) free, appending one StepRecord each. Returns held value."""
    for t in range(start, stop):
        gold_ref = gold_ref_of(task, t)
        held_in = held
        attempts = 0
        recovered_syn = False
        final_score = None
        executed = False
        final_call = None
        was_truncated = None
        for attempt in range(max_retries + 1):
            attempts += 1
            ctx = {"task": task, "step": t, "ref": held, "attempt": attempt}
            messages.append({"role": "user", "content": "[step " + str(t) + "]",
                             "_ctx": ctx})
            resp = backend.complete(messages, task.schema_view(), call_mode)
            fr = resp.get("finish_reason") if isinstance(resp, dict) else None
            was_truncated = (fr == "length") if fr is not None else None
            call = parse_response(resp, call_mode)
            final_call = call
            ex = execute(task, call.tool or "", call.args or {}, feedback)
            score = score_step(
                call, task.gold[t], ex.schema_valid, ex.known_tool,
                expected_tool=_expected_tool_given_ref(task, t, held),
                expected_args=_expected_args_given_held(task, t, held))
            final_score = score
            if ex.ok:
                executed = True
                if attempt > 0 and score.correct:
                    recovered_syn = True
                messages.append({"role": "assistant",
                                 "content": json.dumps({"tool": call.tool,
                                                        "args": call.args})})
                messages.append({"role": "user", "content": ex.feedback})
                held = ex.output
                break
            messages.append({"role": "user", "content": ex.feedback})

        if final_score is None:
            break
        records.append(StepRecord(
            task.task_id, task.depth, t, mode, call_mode,
            final_call.tool if final_call else None,
            final_score.selection_correct, final_score.selection_matches_gold,
            final_score.args_correct_strict, final_score.args_correct_soft,
            final_score.error_type.value, attempts, executed,
            context_clean_in=(held_in == gold_ref), recovered=recovered_syn,
            stalled_in=False,
            backend_error=bool(final_call and final_call.is_backend_error),
            args_correct_given_state=final_score.args_correct_given_state,
            correct_given_state=final_score.correct_given_state,
            held_ref=held_in, first_listed_even=_first_listed_even(task, t),
            truncated=was_truncated, diverged_in=(held_in != gold_ref),
            gold_ref=gold_ref, held_out=held,
            inject_at=inject_at, injection=corruption, pair_id=pair_id))
    return held


def run_repair_pair(task, backend: Backend, inject_at: int, repair_at: int,
                    corruption: str = "parity_flip",
                    call_mode: str = "uniform",
                    feedback: FeedbackMode = FeedbackMode.STRUCTURED,
                    max_retries: int = 1,
                    schema_style: str = "verbose") -> list[StepRecord]:
    """Corrupt at inject_at, run free, hand the state back at repair_at, finish free.

    Returns the records of BOTH branches. Steps before repair_at appear once per branch
    with identical content, because they were produced by a single shared run.
    """
    from tur.tasks.recovery import RESYNC_TOOL, canonical_ref_at

    if not hasattr(task, "branches"):
        raise ValueError("the repair arm requires a RoutingTask: on a linear task the "
                         "tool order is announced in the prompt, so restoring a value "
                         "cannot change which tool is correct and there is nothing for "
                         "the repair to put back.")
    if not 1 <= inject_at < task.depth:
        raise ValueError("inject_at must satisfy 1 <= inject_at < depth (" +
                         str(task.depth) + "); got " + str(inject_at) +
                         ". Step 0 has no previous result to corrupt.")
    if not inject_at < repair_at <= task.depth - 2:
        # The upper bound is what stops this becoming Appendix E: at repair_at = depth-2
        # exactly two steps remain, which is the minimum that makes the verdict a
        # downstream trajectory property rather than a single corrected call.
        raise ValueError(
            "repair_at must satisfy inject_at < repair_at <= depth-2 (" +
            str(task.depth - 2) + "); got " + str(repair_at) + ". A repair with fewer "
            "than two steps left measures the next call, which Appendix E already "
            "reports, rather than whether the trajectory returns to the goal.")
    if corruption not in _INJECTIONS:
        raise ValueError("unknown injection mode " + repr(corruption))

    true_val = task.gold[inject_at - 1].output
    import zlib
    seed = zlib.crc32((task.task_id + "|" + str(inject_at) + "|" + corruption).encode())
    bad_val = _corrupt_value(true_val, corruption, seed)
    pair_id = (task.task_id + "|j" + str(inject_at) + "|r" + str(repair_at) + "|"
               + corruption)

    # ---- shared prefix: teacher-forced to inject_at with one corrupted result ----
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _task_intro(task, schema_style)}]
    for j in range(inject_at):
        messages.append({"role": "user", "content": "[step " + str(j) + "]"})
        messages.append({"role": "assistant",
                         "content": json.dumps({"tool": task.gold[j].tool,
                                                "args": task.gold[j].args})})
        shown = bad_val if j == inject_at - 1 else task.gold[j].output
        messages.append({"role": "user", "content": "result: " + str(shown)})

    # ---- shared free segment, run ONCE so both branches inherit the same history ----
    shared: list[StepRecord] = []
    held = _free_segment(task, backend, messages, bad_val, inject_at, repair_at,
                         "rep_shared", call_mode, feedback, max_retries, pair_id,
                         inject_at, corruption, shared, canonical_ref_at)

    records: list[StepRecord] = []
    for mode, handed in ((_REP_REPAIRED, canonical_ref_at(task, repair_at)),
                         (_REP_UNREPAIRED, held)):
        # The branches differ in exactly this number. The unrepaired branch pays the same
        # call and sees the same shape, so the contrast is the value, not the exchange.
        branch = [dict(m) for m in messages]
        branch.append({"role": "assistant",
                       "content": json.dumps({"tool": RESYNC_TOOL, "args": {}})})
        branch.append({"role": "user", "content": "result: " + str(handed)})

        for r in shared:
            records.append(replace(r, run_mode=mode))
        _free_segment(task, backend, branch, handed, repair_at, task.depth,
                      mode, call_mode, feedback, max_retries, pair_id,
                      inject_at, corruption, records, canonical_ref_at)
    return records


def repair_outcome(task, records: list[StepRecord], inject_at: int,
                   repair_at: int) -> dict:
    """Did the repair return the trajectory to the ORIGINAL goal, all the way to the end?

    `completed_original` is the load-bearing measure and deliberately strict: the branch
    finished every step AND finished holding the task's true final output. A branch that
    took the handed value, made one correct call and then drifted is NOT recovered, which
    is precisely the distinction a next-call-only check cannot draw.
    """
    out = {}
    for mode in (_REP_REPAIRED, _REP_UNREPAIRED):
        rs = sorted((r for r in records if r.run_mode == mode), key=lambda r: r.step)
        if not rs:
            continue
        # only the steps from the repair onward are attributable to the repair
        post = [r for r in rs if r.step >= repair_at]
        # Steps before inject_at are teacher-forced into the prefix and never scored, so
        # a branch that ran to the end holds depth - inject_at records, not depth.
        completed = (len(rs) == task.depth - inject_at
                     and rs[-1].step == task.depth - 1)
        final_held = rs[-1].held_out
        out[mode] = {
            "completed": completed,
            "completed_original": bool(completed and final_held == task.gold[-1].output),
            "returned_to_gold_steps": sum(1 for r in rs if not r.diverged_in),
            "downstream_gold_agreement": _rate_of(
                post, lambda r: r.selection_matches_gold and r.args_correct_strict),
            "downstream_conditional": _rate_of(post, lambda r: r.correct_given_state),
            "n_downstream": len(post),
        }
    return out


def _rate_of(rs, pred):
    return float("nan") if not rs else sum(1 for r in rs if pred(r)) / len(rs)
