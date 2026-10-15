"""Load and validate a task's test cases (tasks/<task>/cases.jsonl)."""

from __future__ import annotations

import json
from collections import Counter

from bench.metrics import TASKS_DIR

REQUIRED_TAGS = ("profile_length", "product_type", "difficulty")


def load(task: str) -> list[dict]:
    path = TASKS_DIR / task / "cases.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(cases: list[dict]) -> list[str]:
    problems = []
    ids = Counter(c.get("id") for c in cases)
    problems += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    for c in cases:
        missing = [t for t in REQUIRED_TAGS if t not in c.get("tags", {})]
        if missing:
            problems.append(f"{c.get('id')}: missing tags {missing}")
        if not c.get("inputs"):
            problems.append(f"{c.get('id')}: no inputs")
    return problems


def summary(cases: list[dict]) -> dict[str, Counter]:
    """Count cases per tag value, e.g. {'difficulty': Counter({'normal': 13, 'tricky': 7})}."""
    return {t: Counter(c["tags"][t] for c in cases) for t in REQUIRED_TAGS}
