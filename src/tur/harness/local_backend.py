"""Local quantized HuggingFace backend, drop-in for LiteLLMBackend.

WHY THIS EXISTS

Three models in the frozen suite have been retired from the Groq free tier mid-project:
llama-3.1-8b-instant and llama-3.3-70b-versatile (2026-08-27), and qwen/qwen3.6-27b
(observed gone 2026-08-29). Their frozen data stays valid, but no new condition can be
run on them through the provider. Running the open weights locally is the only route
back to those models.

WHAT "DROP-IN" MEANS HERE

This implements exactly the contract the harness already depends on:

    complete(messages, tools, mode) -> dict

returning {"text": str, "finish_reason": str|None} in uniform mode, or
{"text": "", "_backend_error": str} when generation fails, which the parser scores as a
syntactic failure rather than crashing the sweep. It exposes the same counters
(n_calls, n_cache_hits, n_retries, n_failures, n_rate_limited) and the same stats().
Message dicts have their _-prefixed private keys stripped before use, as LiteLLMBackend
does, so the harness's _ctx passthrough does not leak into the prompt.

Nothing in run_free / run_teacher_forced / run_injection_pair needs to change. That is
deliberate: the project has already been bitten once by a parallel implementation of
prompt assembly drifting out of sync with the real one (see estimate_cost's note), and a
second inference path is a far bigger surface for the same mistake.

CACHE NAMESPACING IS NOT OPTIONAL

The cache key is a hash of (model, mode, messages, tools). A local run MUST therefore
use a model string distinct from the Groq one, or its responses collide with frozen
Groq-hosted cache entries and the two conditions silently merge. This class enforces
that: the model name is required to carry the LOCAL_PREFIX, and __init__ refuses a bare
provider-style name. Getting this wrong would be unrecoverable after the fact, because
the merged entries are indistinguishable once written.

EQUIVALENCE IS AN OPEN QUESTION, NOT AN ASSUMPTION

4-bit or 8-bit quantized weights on different hardware, a different sampling stack and a
different chat template do not reproduce a hosted endpoint bit-for-bit, and may not
reproduce it distributionally either. Until scripts/compare_local_vs_groq.py has been run
and its verdict recorded, results from this backend are a SEPARATE CONDITION and must be
tagged as such wherever they appear. `condition_tag` is attached to stats() so that the
tag travels with the data rather than living only in someone's memory of how it was run.
"""

from __future__ import annotations

import time
from typing import Any

from tur.harness.cache import Cache

LOCAL_PREFIX = "local/"

# Chat templates and 4-bit kernels differ enough between hosts that we do not claim
# determinism; greedy decoding is still the right default because it removes sampling
# as one more source of divergence when comparing against the frozen numbers.
_DEFAULT_MAX_NEW_TOKENS = 96


class LocalHFBackend:
    """Quantized local inference with the LiteLLMBackend call contract.

    torch/transformers/bitsandbytes are imported lazily inside _ensure_model(), so this
    module imports and its logic is testable on a machine with no GPU and none of those
    packages installed. That matters: the resume and checkpoint tests must run in CI and
    on the development machine, neither of which has a GPU.
    """

    def __init__(self, model: str, temperature: float = 0.0,
                 max_retries: int = 2, base_delay: float = 1.0,
                 timeout: float = 60.0, cache: "Cache | None" = None,
                 limiter: Any = None,
                 hf_id: str | None = None,
                 load_in_4bit: bool = True,
                 max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
                 device_map: str = "auto",
                 condition_tag: str = "local-quantized"):
        if not model.startswith(LOCAL_PREFIX):
            raise ValueError(
                "local backend model name must start with %r so its cache entries "
                "cannot collide with frozen provider-hosted ones; got %r. A collision "
                "here silently merges two conditions and is unrecoverable afterwards."
                % (LOCAL_PREFIX, model))
        self.model = model
        self.hf_id = hf_id or model[len(LOCAL_PREFIX):]
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self.cache = cache
        self.limiter = limiter          # accepted and ignored: no provider quota locally
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.condition_tag = condition_tag

        self.n_calls = 0
        self.n_cache_hits = 0
        self.n_retries = 0
        self.n_failures = 0
        self.n_rate_limited = 0         # always 0 locally; kept for interface parity

        self._tok = None
        self._model = None

    # ------------------------------------------------------------------ model

    def _ensure_model(self):
        """Load tokenizer and quantized weights on first use."""
        if self._model is not None:
            return
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)

        if self.load_in_4bit:
            # nf4 + double quant + bf16 compute is the configuration that fits an 8B
            # model and a depth-8 KV cache inside a 16GB T4 with headroom. 8-bit fits
            # 8B too but leaves much less room for the cache at our context lengths.
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
                if torch.cuda.is_bf16_supported() else torch.float16)
        else:
            qcfg = BitsAndBytesConfig(load_in_8bit=True)

        self._tok = AutoTokenizer.from_pretrained(self.hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, quantization_config=qcfg, device_map=self.device_map)
        self._model.eval()
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token

    # ------------------------------------------------------------- generation

    def _generate(self, clean: list[dict]) -> tuple[str, str | None]:
        """Return (text, finish_reason) for one chat completion."""
        import torch
        self._ensure_model()
        prompt = self._tok.apply_chat_template(
            clean, tokenize=False, add_generation_prompt=True)
        enc = self._tok(prompt, return_tensors="pt").to(self._model.device)
        n_in = enc["input_ids"].shape[-1]
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self._tok.pad_token_id)
        gen = out[0][n_in:]
        text = self._tok.decode(gen, skip_special_tokens=True)
        # Mirror the provider's finish_reason semantics: "length" means the completion
        # hit the token ceiling. The harness treats that as a distinct category from a
        # parse failure, and conflating them would misattribute a configuration fault
        # to the model.
        finish = "length" if gen.shape[-1] >= self.max_new_tokens else "stop"
        return text, finish

    # ---------------------------------------------------------------- contract

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

        if mode == "native":
            # Refused rather than emulated. The uniform protocol is the paper's
            # headline and is what every frozen number uses; a hand-rolled local
            # imitation of provider-native tool calling would be a different
            # interface being compared against provider-native results, which is
            # precisely the confound the calling-mode ablation exists to measure.
            raise NotImplementedError(
                "native tool-calling mode is not supported by the local backend; "
                "run the uniform protocol, which is what the frozen suite uses")

        last_err = None
        for attempt in range(self.max_retries + 1):
            self.n_calls += 1
            try:
                text, finish = self._generate(clean)
                result = {"text": text, "finish_reason": finish}
                if self.cache is not None:
                    self.cache.set(cache_key, result)
                return result
            except Exception as e:                      # noqa: BLE001
                last_err = e
                if attempt < self.max_retries:
                    self.n_retries += 1
                    time.sleep(self.base_delay * (2 ** attempt))
                else:
                    self.n_failures += 1
        # Same sentinel LiteLLMBackend returns: scored as a syntactic failure and
        # logged, rather than killing a sweep that may be hours in.
        return {"text": "", "_backend_error": str(last_err)}

    def stats(self) -> dict:
        return {"model": self.model, "hf_id": self.hf_id,
                "condition_tag": self.condition_tag,
                "quantization": "4bit-nf4" if self.load_in_4bit else "8bit",
                "n_calls": self.n_calls, "n_cache_hits": self.n_cache_hits,
                "n_retries": self.n_retries, "n_failures": self.n_failures,
                "n_rate_limited": self.n_rate_limited}
