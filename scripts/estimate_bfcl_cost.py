"""Call and token counts for the BFCL L_d arm, measured on the real prompts.

Drives run_bfcl_teacher_forced and run_bfcl_free against a mock backend and counts tokens
on the messages they actually build, rather than reimplementing prompt assembly here --
the same discipline as estimate_cost.py, and for the same reason: a duplicated assembly
goes stale the moment a run loop changes and silently understates the bill.

Counted with an ORACLE policy. Token volume depends on the schema and the history, not on
whether a call is right, but an erring policy produces different execution results (error
strings instead of payloads) and would make the estimate depend on how badly the mock
happens to behave.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tur.harness.bfcl_runner import (curated_subset, run_bfcl_free,       # noqa: E402
                                     run_bfcl_teacher_forced)
from tur.harness.runner import MockBackend                                # noqa: E402

_OUT_TOKENS_PER_CALL = 40


def _tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return (lambda s: len(enc.encode(s))), "tiktoken/o200k_base"
    except Exception:                                        # noqa: BLE001
        return (lambda s: max(1, len(s) // 4)), "chars/4 (tiktoken unavailable)"


def _oracle(task, step, ref, attempt):
    g = task.gold[step]
    return g["tool"], g["args"], True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-depth", type=int, default=6)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument("--n", type=int, default=None, help="cap on tasks (pilot size)")
    ap.add_argument("--tpm", type=int, default=6000)
    ap.add_argument("--tpd", type=int, default=200000)
    ap.add_argument("--headroom", type=float, default=0.8)
    args = ap.parse_args()

    ntok, tok_name = _tokenizer()
    tasks = curated_subset(args.min_depth, args.max_depth, args.n)

    calls = 0
    prompt_tokens = 0
    for t in tasks:
        for fn in (run_bfcl_teacher_forced, run_bfcl_free):
            counted = {"calls": 0, "tokens": 0}
            backend = MockBackend(_oracle)
            inner = backend.complete

            def complete(messages, tools, mode, _c=counted, _inner=inner):
                text = "".join(str(m.get("content", "")) for m in messages)
                _c["tokens"] += ntok(text)
                _c["calls"] += 1
                return _inner(messages, tools, mode)

            backend.complete = complete
            fn(t, backend)
            calls += counted["calls"]
            prompt_tokens += counted["tokens"]

    out_tokens = calls * _OUT_TOKENS_PER_CALL
    total = prompt_tokens + out_tokens
    eff_tpm = args.tpm * args.headroom
    eff_tpd = args.tpd * args.headroom

    print("BFCL L_d arm -- SINGLE-MODEL CHECK (gpt-oss-120b alone), NOT comparative")
    print("  depth range      : %d-%d" % (args.min_depth, args.max_depth))
    print("  tasks            : %d" % len(tasks))
    print("  tokenizer        : %s" % tok_name)
    print()
    print("  backend calls    : %d   (both arms, one call per gold step per arm)" % calls)
    print("  prompt tokens    : %d" % prompt_tokens)
    print("  output tokens    : %d  (est. %d/call)" % (out_tokens, _OUT_TOKENS_PER_CALL))
    print("  total tokens     : %d" % total)
    print("  avg tokens/call  : %d" % (total // max(1, calls)))
    print()
    print("  at %d TPM / %s TPD, %.0f%% headroom:" % (args.tpm, "{:,}".format(args.tpd),
                                                      100 * args.headroom))
    print("    wall-clock hours (TPM-bound) : %.1f" % (total / eff_tpm / 60))
    print("    days (TPD-bound)             : %.2f" % (total / eff_tpd))

    import collections
    print("\n  depth distribution: %s"
          % dict(sorted(collections.Counter(t.depth for t in tasks).items())))


if __name__ == "__main__":
    main()
