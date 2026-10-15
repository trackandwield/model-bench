"""Plan a run before it starts: the stored config that makes it reproducible, and a cost estimate.

The config is written before any call, so a crashed run can be resumed from it:
every finished call is already in the cache, so a resume only pays for what's missing.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import litellm

from bench import metrics
from bench.compare import RUNS_DIR, call_key
from bench.store import Store
from bench.tasks import Task
from bench.workflow import repeat_tag

ROOT = Path(__file__).resolve().parent.parent

# Output tokens assumed per step when a model has no history yet. Generous on
# purpose: reasoning models spend hidden tokens, and an estimate should err high.
DEFAULT_OUTPUT_TOKENS = 1500


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _git() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": git("rev-parse", "HEAD") or None, "uncommitted_changes": bool(git("status", "--porcelain"))}


def _version(pkg: str) -> str | None:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def fingerprints(task: str) -> dict:
    """Hashes of everything that defines a task, so a later run can tell if anything changed."""
    folder = metrics.TASKS_DIR / task
    return {
        "task_sha256": _sha(folder / "task.yaml"),
        "dataset_sha256": _sha(folder / "cases.jsonl"),
        "metrics_sha256": _sha(folder / "metrics.yaml"),
        "metrics_locked": metrics.verify(task),
    }


def build_config(run_id: str, task: str, models: list[str], case_ids: list[str], *, repeats: int,
                 settings: dict, budget_usd: float | None, mock: bool) -> dict:
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": task,
        "models": models,
        "case_ids": case_ids,
        "repeats": repeats,
        "settings": settings,
        "budget_usd": budget_usd,
        "mock": mock,
        **fingerprints(task),
        "git": _git(),
        "versions": {"litellm": _version("litellm"), "model-bench": _version("model-bench")},
    }


def config_path(run_id: str, task: str) -> Path:
    return RUNS_DIR / f"{run_id}-{task}.config.json"


def save_config(config: dict) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    path = config_path(config["run_id"], config["task"])
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def changed_since(config: dict) -> list[str]:
    """What about the task differs from when this config was saved."""
    now = fingerprints(config["task"])
    names = {"task_sha256": "task.yaml", "dataset_sha256": "cases.jsonl", "metrics_sha256": "metrics.yaml"}
    return [f for k, f in names.items() if config.get(k) != now[k]]


@dataclass
class Estimate:
    calls: int = 0
    cached: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    unpriced: set[str] = field(default_factory=set)   # models with no known price
    guessed: set[str] = field(default_factory=set)    # models with no token history; used the default

    @property
    def to_make(self) -> int:
        return self.calls - self.cached


def _tokens(text: str) -> int:
    # Rough count (about 4 characters a token). Good enough for a budget and
    # needs no tokenizer download.
    return max(1, len(text) // 4)


def output_history(store: Store) -> dict[tuple[str, str], float]:
    """Average output tokens per (model, step) from real, successful calls."""
    totals: dict[tuple[str, str], list[int]] = defaultdict(list)
    for model, input_id, settings, out in store.db.execute(
        "SELECT model, input_id, settings, output_tokens FROM calls "
        "WHERE error IS NULL AND cached = 0 AND input_id LIKE '%:%'"
    ):
        if '"mock": true' in settings:
            continue
        totals[(model, input_id.split(":")[1])].append(out)
    return {k: sum(v) / len(v) for k, v in totals.items()}


def _price(model: str, tokens_in: int, tokens_out: int, prices: dict) -> float | None:
    p = prices.get(model)
    if p:
        return (tokens_in * p["input"] + tokens_out * p["output"]) / 1_000_000
    try:
        cin, cout = litellm.cost_per_token(model=model, prompt_tokens=tokens_in, completion_tokens=tokens_out)
        return cin + cout
    except Exception:
        return None


def estimate(task: Task, cases: list[dict], models: list[str], *, repeats: int, settings: dict, store: Store,
             prices: dict | None = None, mock: bool = False, use_cache: bool = True) -> Estimate:
    """Walk every chain through the cache. Steps already cached cost nothing; for the
    rest, input tokens come from the prompt and output tokens from past runs."""
    est = Estimate()
    history = output_history(store)
    system = task.system_prompt()
    for m in models:
        for c in cases:
            for r in range(1, repeats + 1):
                previous: dict[str, str] = {}
                hit = use_cache
                for step in task.steps:
                    est.calls += 1
                    prompt = step.render(c["inputs"], previous)
                    if hit:
                        key, _ = call_key(m, prompt, system, settings, mock=mock, cache_tag=repeat_tag(r))
                        cached = store.cached(key)
                        if cached is not None:
                            est.cached += 1
                            previous[step.name] = cached.text
                            continue
                        hit = False  # later steps depend on this one, so they can't be cached either
                    out = history.get((m, step.name))
                    if out is None:
                        out = DEFAULT_OUTPUT_TOKENS
                        est.guessed.add(m)
                    previous[step.name] = "x" * int(out * 4)  # stand-in of the right length for later steps
                    if mock:
                        continue
                    cost = _price(m, _tokens((system or "") + prompt), int(out), prices or {})
                    if cost is None:
                        est.unpriced.add(m)
                    else:
                        est.cost_usd += cost
                        est.by_model[m] += cost
    return est
