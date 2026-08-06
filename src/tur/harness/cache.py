"""Content-addressed cache so re-runs cost nothing.

Key is a hash of (model, mode, messages, tools). Values are the raw backend
responses. This makes the whole pipeline deterministic and free to replay.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class Cache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(model: str, mode: str, messages: list[dict], tools: Any) -> str:
        blob = json.dumps({"model": model, "mode": mode, "messages": messages,
                           "tools": tools}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def get(self, key: str) -> dict | None:
        f = self.path / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text())
        return None

    def set(self, key: str, value: dict) -> None:
        (self.path / f"{key}.json").write_text(json.dumps(value))
