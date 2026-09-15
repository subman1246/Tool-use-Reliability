"""Colab cell: local-quantized llama-3.1-8b sweep, resumable, Drive-checkpointed.

Paste the whole file into one Colab cell and run it. Safe to re-run after a disconnect:
it resumes from the Drive checkpoint rather than restarting.

SUBSET AND WHY IT IS PAIRED WITH THE FROZEN RUN

The frozen Groq arm achieved {1:60, 2:60, 4:65, 6:65, 8:23} at base_seed 1000. This sweep
runs {1:24, 2:24, 4:24, 6:65, 8:23}: full n where the propagation signal lives (depths 6
and 8) and a cheaper prefix at the shallow depths, which are there to confirm the
implementation invariant rather than to carry the result.

generate_routing_suite emits k = 0..n-1 and task identity depends only on
(depth, k, base_seed), so n=24 yields task ids r{d}_0_s1000 .. r{d}_23_s1000 -- an exact
PREFIX of the frozen run's k=0..59. The comparison is therefore paired on the same tasks,
not two independent samples that happen to share a depth. Changing base_seed, or taking a
random subset instead of a prefix, breaks that and makes the equivalence check meaningless.

COST: 160 tasks, 1,484 generations (both arms, sum of 2*depth per task).
"""

# ----------------------------------------------------------------- Colab setup
# !pip install -q transformers accelerate bitsandbytes
# from google.colab import drive; drive.mount('/content/drive')
# !git clone https://github.com/<you>/Tool-use-Reliability.git /content/tur

import os
import sys
import json
from dataclasses import asdict

REPO = "/content/tur"                                  # where the repo is checked out
DRIVE = "/content/drive/MyDrive/tur_checkpoints"       # survives a disconnect
OUT_DIR = os.path.join(REPO, "data", "results")

sys.path.insert(0, os.path.join(REPO, "src"))
os.chdir(REPO)

from tur.harness.local_backend import LocalHFBackend      # noqa: E402
from tur.harness.checkpoint import (CheckpointingBackend,  # noqa: E402
                                    TrajectoryStore)
from tur.harness.cache import Cache                        # noqa: E402
from tur.harness.executor import FeedbackMode              # noqa: E402
from tur.harness.runner import run_free, run_teacher_forced  # noqa: E402
from tur.tasks.dag import generate_routing_suite           # noqa: E402

# --------------------------------------------------------------------- config
# The local/ prefix is REQUIRED. The response cache is keyed on
# (model, mode, messages, tools), so reusing the Groq model string would collide with
# frozen Groq-hosted cache entries and merge two conditions unrecoverably.
# LocalHFBackend refuses a name without it.
MODEL = "local/meta-llama/Llama-3.1-8B-Instruct"
TAG = "localval"

DEPTHS = [1, 2, 4, 6, 8]
PER_DEPTH = {1: 24, 2: 24, 4: 24, 6: 65, 8: 23}
BASE_SEED = 1000            # must match the frozen run, or the pairing is lost
DISTRACTORS = 1
MAX_RETRIES = 1

os.makedirs(DRIVE, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------- backend
backend = LocalHFBackend(
    MODEL,
    temperature=0.0,                 # greedy: removes sampling as a source of divergence
    load_in_4bit=True,               # nf4 + double quant fits 8B + depth-8 KV in 16GB
    cache=Cache(os.path.join(REPO, "data", "cache", MODEL.replace("/", "_"))),
    condition_tag="local-quantized-4bit-T4",
)
store = TrajectoryStore(DRIVE, TAG, MODEL)
be = CheckpointingBackend(backend, store)

resume = store.resume_summary()
print("resuming: %d tasks already done, %d partial, %d calls on disk"
      % (len(resume["done"]), len(resume["partial"]), resume["recorded_calls"]))

# --------------------------------------------------------------------- sweep
suite = generate_routing_suite(DEPTHS, PER_DEPTH, DISTRACTORS, base_seed=BASE_SEED)
print("suite: %d tasks  %s" % (len(suite), PER_DEPTH))

out_path = os.path.join(OUT_DIR, "%s_%s.jsonl" % (TAG, MODEL.replace("/", "_")))
records = []

for i, task in enumerate(suite):
    # Two run modes share one checkpoint id, so a task is only "done" once BOTH arms
    # have completed. Keying them separately would let a disconnect between the arms
    # leave a task that looks finished with only half its calls.
    if store.is_done(task.task_id):
        continue
    already = be.begin_task(task.task_id)
    if already:
        print("  [%d/%d] %s resuming at call %d" % (i + 1, len(suite), task.task_id, already))
    f = run_free(task, be, "uniform", FeedbackMode.STRUCTURED, MAX_RETRIES)
    t = run_teacher_forced(task, be, "uniform", FeedbackMode.STRUCTURED, MAX_RETRIES)
    be.end_task()

    rows = [asdict(r) for r in f] + [asdict(r) for r in t]
    for r in rows:
        r["condition_tag"] = backend.condition_tag     # the tag travels with the data
        r["backend"] = "local-hf-4bit"
    records += rows
    # Append as we go: if the runtime dies, the jsonl already holds finished tasks.
    with open(out_path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    if (i + 1) % 10 == 0:
        print("  [%d/%d] generated=%d replayed=%d failures=%d"
              % (i + 1, len(suite), be.n_generated, be.n_replayed, backend.n_failures))

print()
print("done. %d new records -> %s" % (len(records), out_path))
print("stats:", json.dumps(be.stats(), indent=2))
print()
print("COPY BACK: %s" % out_path)
print("Then run locally:")
print("  py scripts/compare_local_vs_groq.py \\")
print("     --local-tag %s --local-model %s \\" % (TAG, MODEL))
print("     --frozen-model groq/llama-3.1-8b-instant")
