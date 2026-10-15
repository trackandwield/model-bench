"""Load, validate and lock a task's success metrics.

Locking records a hash of metrics.yaml before any model runs. Every run stores
that hash, so anyone can check the metrics weren't changed after results came in.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from bench.checks import CHECKS

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"
METRIC_TYPES = {"check", "judge", "human", "operational"}


def metrics_path(task: str) -> Path:
    return TASKS_DIR / task / "metrics.yaml"


def lock_path(task: str) -> Path:
    return TASKS_DIR / task / "metrics.lock.json"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(spec: dict) -> list[str]:
    """Return a list of problems; empty means the metrics file is usable."""
    problems = []
    metrics = spec.get("metrics") or []
    if not metrics:
        problems.append("no metrics defined")
    names = set()
    for m in metrics:
        name = m.get("name", "<unnamed>")
        if name in names:
            problems.append(f"{name}: duplicate name")
        names.add(name)
        if m.get("type") not in METRIC_TYPES:
            problems.append(f"{name}: type must be one of {sorted(METRIC_TYPES)}")
        if not isinstance(m.get("weight"), (int, float)) or m["weight"] < 0:
            problems.append(f"{name}: weight must be a non-negative number")
        if not m.get("pass"):
            problems.append(f"{name}: missing pass threshold")
        if not m.get("description"):
            problems.append(f"{name}: missing description")
        if m.get("type") == "check" and m.get("check") not in CHECKS:
            problems.append(f"{name}: check must name one of {sorted(CHECKS)}")
        if m.get("type") == "judge" and sorted((m.get("rubric") or {}).keys()) != [1, 2, 3, 4, 5]:
            problems.append(f"{name}: judge metrics need a rubric describing scores 1 to 5")
    for d in spec.get("diagnostics") or []:
        if d.get("check", d.get("name")) not in CHECKS:
            problems.append(f"diagnostic {d.get('name')}: unknown check")
    total = sum(m.get("weight", 0) for m in metrics if isinstance(m.get("weight"), (int, float)))
    if metrics and abs(total - 1.0) > 1e-6:
        problems.append(f"weights add up to {total:.2f}, not 1.0")
    return problems


def load(task: str) -> dict:
    return yaml.safe_load(metrics_path(task).read_text())


def lock(task: str) -> dict:
    """Validate the metrics and write the lock file. Raises ValueError if invalid."""
    spec = load(task)
    problems = validate(spec)
    if problems:
        raise ValueError("; ".join(problems))
    record = {
        "task": task,
        "sha256": file_hash(metrics_path(task)),
        "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": spec.get("status"),
        "reviewed_by": spec.get("reviewed_by"),
    }
    lock_path(task).write_text(json.dumps(record, indent=2) + "\n")
    return record


def verify(task: str) -> bool:
    """True when metrics.yaml still matches its lock."""
    lp = lock_path(task)
    if not lp.exists():
        return False
    return json.loads(lp.read_text())["sha256"] == file_hash(metrics_path(task))
