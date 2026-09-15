#!/usr/bin/env bash
# Re-invoke the BFCL sweep until it completes its planned n.
# Every finished call is in the content-addressed cache, so each attempt replays the
# done prefix for free and only pays for new ground. The daily token allowance is the
# binding constraint, so the loop sleeps and retries rather than running once.
TAG="${1:-bfclpilot}"; N="${2:-8}"; MAX_ATTEMPTS="${3:-60}"
for i in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "[bfcl-driver] attempt $i"
  py -3 scripts/run_bfcl_suite.py --tag "$TAG" --n "$N" || echo "[bfcl-driver] attempt $i exited nonzero"
  done_n=$(py -3 -c "import json;print(json.load(open('data/results/${TAG}_meta.json'))['tasks_completed'])" 2>/dev/null || echo 0)
  echo "[bfcl-driver] completed ${done_n}/${N}"
  if [ "$done_n" -ge "$N" ]; then echo "[bfcl-driver] done"; break; fi
  echo "[bfcl-driver] waiting 1200s for token refill"
  sleep 1200
done
