"""Loading and executing BFCL multi-turn tasks.

This module is the only adaptation layer over the vendored environment in bfcl_env/, which
is kept byte-identical to upstream so that a re-vendor is a file copy rather than a merge.

VERSION CONSISTENCY IS THE WHOLE GAME HERE

The dataset and the API source code must come from the same BFCL release. The HuggingFace
mirror serves v3 while the repository's source is v4; pairing them replays 147 of 200 gold
trajectories, with 41 of the 53 failures caused by one changed signature. Filtering to
"tasks that run" would have selected a subset on that mismatch and called it curation.
Both are therefore taken from the repository at v4, and tests/test_bfcl_env.py asserts
that ALL gold trajectories replay -- an invariant, not a statistic.

WHAT BFCL CAN AND CANNOT MEASURE FOR THIS PAPER

BFCL multi-turn tasks chain through SIMULATOR STATE, not through a value carried from one
call's output into the next call's argument. `cd(folder='temp')` changes what a later
`mv` means without handing anything to it. A few tasks do pass values (multi_turn_base_15
ends with mean(numbers=[3,16,60]), those three numbers being the outputs of three earlier
wc calls), but they are the exception.

That matters because the synthetic suite's conditional-on-state scorer is a RULE: given
the value actually held, the routing rule says which tool is correct. BFCL has no such
rule. Its gold is a fixed trajectory expressing a user's intent, and after a divergence
there is no oracle for "what should this agent do now, from where it actually is". So
conditional-on-state scoring does not transfer, and claiming it did would be inventing an
oracle. What does transfer is the baseline/free-running contrast -- p_d, g_d and the
propagation loss L_d -- which needs only the gold trajectory and the ability to execute.
"""

from __future__ import annotations

import ast
import functools
import json
from pathlib import Path
from typing import Any

from tur.tasks.bfcl_env import (gorilla_file_system, math_api, message_api,
                                posting_api, ticket_api, trading_bot,
                                travel_booking, vehicle_control)

ROOT = Path(__file__).resolve().parents[3]
BFCL_DIR = ROOT / "data" / "bfcl"

# v4 to match the vendored source. See the module docstring.
SPLIT = "BFCL_v4_multi_turn_base.json"
ANSWER_SPLIT = "possible_answer__BFCL_v4_multi_turn_base.json"

CLASSES: dict[str, Any] = {
    "GorillaFileSystem": gorilla_file_system.GorillaFileSystem,
    "MathAPI": math_api.MathAPI,
    "MessageAPI": message_api.MessageAPI,
    "TwitterAPI": posting_api.TwitterAPI,
    "TicketAPI": ticket_api.TicketAPI,
    "TradingBot": trading_bot.TradingBot,
    "TravelAPI": travel_booking.TravelAPI,
    "VehicleControlAPI": vehicle_control.VehicleControlAPI,
}


@functools.lru_cache(maxsize=1)
def DATA() -> list[dict]:
    p = BFCL_DIR / SPLIT
    if not p.exists():
        raise SystemExit("missing %s -- run scripts/fetch_bfcl.py" % p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


@functools.lru_cache(maxsize=1)
def ANSWERS() -> dict[str, list[list[str]]]:
    p = BFCL_DIR / ANSWER_SPLIT
    if not p.exists():
        raise SystemExit("missing %s -- run scripts/fetch_bfcl.py" % p)
    return {json.loads(l)["id"]: json.loads(l)["ground_truth"]
            for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def instantiate(entry: dict) -> dict[str, Any]:
    """Fresh API instances for one task, loaded with its initial_config."""
    insts = {}
    for cn in entry["involved_classes"]:
        inst = CLASSES[cn]()
        # MathAPI is stateless and exposes no _load_scenario; calling it unconditionally
        # excluded every task involving it for a reason that was purely structural.
        if hasattr(inst, "_load_scenario"):
            inst._load_scenario(dict(entry["initial_config"].get(cn, {})))
        insts[cn] = inst
    return insts


def parse_call(call: str) -> tuple[str, list, dict]:
    """`mv(source='a', destination='b')` -> ("mv", [], {"source": "a", ...}).

    Parsed with ast rather than eval: the ground-truth strings come from a downloaded
    file, and eval on downloaded text executes whatever is in it.
    """
    node = ast.parse(call.strip(), mode="eval").body
    name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
    pos = [ast.literal_eval(a) for a in node.args]
    kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    return name, pos, kw


def gold_calls(entry: dict) -> list[str]:
    """The task's gold trajectory, flattened across turns.

    Depth is the total number of gold calls. Within a single turn BFCL rarely exceeds two
    calls (476 of 742 turns in the split hold exactly one), so a per-turn notion of depth
    would put almost every task at depth 1 and measure nothing about propagation.
    """
    return [c for turn in ANSWERS()[entry["id"]] for c in turn]
