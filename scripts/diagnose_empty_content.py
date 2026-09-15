"""Why did some completions come back with empty content?

18% of scored calls in the BFCL pilot returned `content == ""` with
`finish_reason == "stop"` -- not truncated, not backend errors -- and were scored as parse
failures, dragging both p_d and g_d. This reads the response cache to say WHICH of three
things happened, because the three have different consequences:

  budget exhausted   the model spent its output on reasoning and never emitted an answer.
                     A genuine non-answer. Scoring it wrong is correct, and p_d = 0.417
                     stands.

  answer stranded    the reasoning field contains a complete, parseable tool call that
                     never reached `content`. The model decided; the answer did not
                     surface. That is a provider or client routing fault, the score is an
                     artefact, and the honest p_d is nearer 0.484.

  genuinely blank    no content and no reasoning either. Provider-side anomaly; the call
                     should be treated as a failed request rather than a wrong answer.

The distinction is not cosmetic: it decides whether the n=28 scale-up is measuring the
model or the harness.

Only responses fetched AFTER commit 9f59a47 carry the empty_content marker. Cache keys are
built from the request, so earlier calls replay from cache and never acquire it -- this
reports coverage explicitly rather than implying it read everything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tur.harness.runner import parse_response                       # noqa: E402


def classify(entry: dict) -> tuple[str, str]:
    reasoning = (entry.get("reasoning") or "").strip()
    if not reasoning:
        return "genuinely blank", ""
    # does the scratchpad contain a call the model simply never emitted?
    call = parse_response({"text": reasoning}, "uniform", coerce_ints=False)
    if call.parse_ok and call.tool:
        return "answer stranded in reasoning", "%s(%s)" % (call.tool, call.args)
    # a trailing fragment without a closing brace reads as an interrupted thought
    tail = reasoning[-120:].replace("\n", " ")
    return "budget exhausted (no parseable call)", tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    cache = Path(args.cache)
    if not cache.exists():
        raise SystemExit("no cache at %s" % cache)

    total = marked = 0
    buckets: dict[str, list] = {}
    # rglob, not glob: the response cache is laid out per model
    # (data/cache/groq_allam-2-7b/...), and a top-level glob sees only the handful of
    # entries written by runners that point at the cache root.
    for f in cache.rglob("*.json"):
        try:
            e = json.loads(f.read_text(encoding="utf-8"))
        except Exception:                                            # noqa: BLE001
            continue
        if not isinstance(e, dict):
            continue
        total += 1
        if not e.get("empty_content"):
            continue
        marked += 1
        kind, detail = classify(e)
        buckets.setdefault(kind, []).append((f.name, detail))

    print("cache entries scanned : %d" % total)
    print("carrying empty_content: %d" % marked)
    if marked == 0:
        print("\nNone found. Either no empty response has been fetched since commit "
              "9f59a47, or the affected calls are still being served from cache entries "
              "written before it. Clear those entries to re-diagnose them -- a re-run "
              "alone will not, because cache keys come from the request.")
        return

    print()
    for kind, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        print("%-38s %d" % (kind, len(items)))
    print()
    for kind, items in buckets.items():
        print("--- %s ---" % kind)
        for name, detail in items[:args.show]:
            print("  %s  %s" % (name[:16], detail[:150]))
        print()

    stranded = len(buckets.get("answer stranded in reasoning", []))
    if stranded:
        print("VERDICT: %d response(s) contained a complete tool call that never reached "
              "`content`. Those scores are a harness artefact, not model error, and p_d "
              "and g_d are both understated. Fix the extraction before scaling."
              % stranded)
    else:
        print("VERDICT: no stranded answers. The empty responses look like genuine "
              "non-answers, so scoring them as wrong is correct and the measured p_d "
              "stands.")


if __name__ == "__main__":
    main()
