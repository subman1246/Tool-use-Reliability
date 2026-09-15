"""Download the BFCL split and the API source it is scored against, from ONE version.

Both come from the repository at the same ref. The HuggingFace mirror of the dataset is a
version behind the repository's source code, and mixing them breaks 53 of 200 gold
trajectories -- see tests/test_bfcl_env.py.
"""
from __future__ import annotations

import pathlib
import urllib.request

REF = "main"
REPO = "https://raw.githubusercontent.com/ShishirPatil/gorilla/%s/" % REF
BFCL = REPO + "berkeley-function-call-leaderboard/bfcl_eval/"

DATA = ["data/BFCL_v4_multi_turn_base.json",
        "data/possible_answer/BFCL_v4_multi_turn_base.json"]
SRC = ["gorilla_file_system.py", "math_api.py", "message_api.py", "posting_api.py",
       "ticket_api.py", "trading_bot.py", "travel_booking.py", "vehicle_control.py",
       "long_context.py"]
SRC_DIR = "eval_checker/multi_turn_eval/func_source_code/"

ROOT = pathlib.Path(__file__).resolve().parents[1]


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "tur-research/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> None:
    out = ROOT / "data" / "bfcl"
    out.mkdir(parents=True, exist_ok=True)
    for w in DATA:
        b = get(BFCL + w)
        (out / w.replace("data/", "").replace("/", "__")).write_bytes(b)
        print("%-50s %8d bytes" % (w, len(b)))

    env = ROOT / "src" / "tur" / "tasks" / "bfcl_env"
    env.mkdir(parents=True, exist_ok=True)
    for f in SRC:
        b = get(BFCL + SRC_DIR + f)
        (env / f).write_bytes(b)
        print("%-50s %8d bytes" % (SRC_DIR + f, len(b)))
    print("\nvendored source and dataset are from the same ref (%s)" % REF)


if __name__ == "__main__":
    main()
