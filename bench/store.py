"""Every model call goes into one SQLite log, which also serves as the response cache.

Cache rule: same model + prompt + system + settings = reuse the stored answer and
charge nothing. Errors are logged but never served from cache.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from bench.models import CallResult

DB_PATH = Path(__file__).resolve().parent.parent / "runs" / "bench.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    model TEXT NOT NULL,
    input_id TEXT,
    settings TEXT NOT NULL,
    output TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    latency_s REAL,
    finish_reason TEXT,
    error TEXT,
    cached INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS calls_cache_key ON calls (cache_key);
"""


def cache_key(model: str, prompt: str, system: str | None, settings: dict) -> str:
    blob = json.dumps([model, prompt, system, settings], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class Store:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def cached(self, key: str) -> CallResult | None:
        row = self.db.execute(
            "SELECT model, output, input_tokens, output_tokens, finish_reason FROM calls "
            "WHERE cache_key = ? AND error IS NULL AND cached = 0 ORDER BY id DESC LIMIT 1",
            (key,),
        ).fetchone()
        if not row:
            return None
        model, output, tin, tout, finish = row
        return CallResult(model, output, tin, tout, 0.0, 0.0, finish_reason=finish, cached=True)

    def log(self, run_id: str, key: str, settings: dict, result: CallResult, input_id: str | None = None):
        self.db.execute(
            "INSERT INTO calls (run_id, created_at, cache_key, model, input_id, settings, output, "
            "input_tokens, output_tokens, cost_usd, latency_s, finish_reason, error, cached) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                key,
                result.model,
                input_id,
                json.dumps(settings, sort_keys=True),
                result.text,
                result.input_tokens,
                result.output_tokens,
                result.cost_usd,
                result.latency_s,
                result.finish_reason,
                result.error,
                int(result.cached),
            ),
        )
        self.db.commit()
