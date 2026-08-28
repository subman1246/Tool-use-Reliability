"""Score Phase 4's completed calls from the response cache, WITHOUT any API call.

The sweep holds records in memory and only writes jsonl when it finishes, so a partial
report has to come from the cache. This replays the same suite against a backend that
serves cached responses only and refuses to fall through to the network -- a miss raises,
the pair is skipped, and the achieved n is whatever completed. Nothing here can spend
budget or compete with the live run.
"""
import json, os, sys
sys.path.insert(0, "src")

from tur.harness.cache import Cache
from tur.harness.executor import FeedbackMode
from tur.harness.runner import run_injection_pair
from tur.tasks.dag import generate_routing_suite

MODEL = "groq/allam-2-7b"
CACHE = Cache("data/cache/" + MODEL.replace("/", "_"))


class CacheMiss(Exception):
    pass


class CacheOnlyBackend:
    """Serves cached responses; never calls a provider."""
    def __init__(self):
        self.hits = 0
        self.misses = 0

    def complete(self, messages, tools, mode):
        clean = [{k: v for k, v in m.items() if not k.startswith("_")}
                 for m in messages]
        hit = CACHE.get(Cache.key(MODEL, mode, clean, tools))
        if hit is None:
            self.misses += 1
            raise CacheMiss()
        self.hits += 1
        return hit


# Exactly the suite the live run is executing: _suite_for("injection", ...) with
# base_seed = seed*31 + 1000 at seed 0.
suite = generate_routing_suite([6], {6: 60}, 1, base_seed=1000)

records = []
done_tasks = set()
for task in suite:
    for j in (1, 3, 5):
        for mode in ("parity_flip", "parity_preserving"):
            be = CacheOnlyBackend()
            try:
                recs = run_injection_pair(task, be, j, mode, "uniform",
                                          FeedbackMode.STRUCTURED, 1)
            except CacheMiss:
                continue
            records += [r.__dict__ for r in recs]
            done_tasks.add(task.task_id)

out = "data/results/inject_partial_groq_allam-2-7b.jsonl"
with open(out, "w") as fh:
    for r in records:
        fh.write(json.dumps(r) + "\n")
print("replayed %d records from cache across %d tasks (0 API calls)"
      % (len(records), len(done_tasks)))
print("saved -> " + out)
