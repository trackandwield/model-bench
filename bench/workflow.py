"""Run a task (prompt or multi-step workflow) on every model × every test case."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bench.checks import CheckResult, run_checks
from bench.compare import RUNS_DIR, cached_call, new_run_id
from bench.models import gather_limited, provider
from bench.store import Store
from bench.tasks import Task


@dataclass
class StepResult:
    name: str
    prompt: str
    output: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: float
    cached: bool
    error: str | None = None


@dataclass
class CaseResult:
    model: str
    case_id: str
    steps: list[StepResult] = field(default_factory=list)
    checks: dict[str, CheckResult] = field(default_factory=dict)

    @property
    def failed_step(self) -> str | None:
        """Where the chain broke, if it did."""
        return next((s.name for s in self.steps if s.error), None)

    @property
    def output(self) -> str | None:
        return None if self.failed_step else self.steps[-1].output

    @property
    def cost_usd(self) -> float | None:
        costs = [s.cost_usd for s in self.steps]
        return None if any(c is None for c in costs) else sum(costs)

    @property
    def latency_s(self) -> float:
        return sum(s.latency_s for s in self.steps)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "case_id": self.case_id,
            "output": self.output,
            "failed_step": self.failed_step,
            "cost_usd": self.cost_usd,
            "latency_s": self.latency_s,
            "checks": {k: v.to_dict() for k, v in self.checks.items()},
            "steps": [asdict(s) for s in self.steps],
        }


async def run_case(task: Task, case: dict, model: str, *, settings: dict, store: Store, run_id: str,
                   mock: bool = False, use_cache: bool = True, prices: dict | None = None) -> CaseResult:
    result = CaseResult(model, case["id"])
    previous: dict[str, str] = {}
    for step in task.steps:
        prompt = step.render(case["inputs"], previous)
        r = await cached_call(
            model, prompt, system=task.system_prompt(), settings=settings, store=store, run_id=run_id,
            mock_text=f"Mock {step.name} from {model} for {case['id']}." if mock else None,
            use_cache=use_cache, prices=prices, input_id=f"{case['id']}:{step.name}",
        )
        error = r.error
        if not error and r.finish_reason == "length":
            # A cut-off extract would feed half the facts into the draft step.
            error = f"Output cut off at max_tokens ({settings.get('max_tokens')}); raise it in models.yaml"
        result.steps.append(StepResult(step.name, prompt, r.text, r.input_tokens, r.output_tokens,
                                       r.cost_usd, r.latency_s, r.cached, error))
        if error:
            break  # later steps depend on this one
        previous[step.name] = r.text
    return result


async def run_matrix(task: Task, cases: list[dict], models: list[str], **kw) -> list[CaseResult]:
    by_provider: dict[str, list] = defaultdict(list)
    for m in models:
        for c in cases:
            by_provider[provider(m)].append(run_case(task, c, m, **kw))
    results = await gather_limited(by_provider)
    rank = {(m, c["id"]): (i, j) for i, m in enumerate(models) for j, c in enumerate(cases)}
    return sorted(results, key=lambda r: rank[(r.model, r.case_id)])


def write_results(run_id: str, task: Task, settings: dict, results: list[CaseResult]) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{run_id}-{task.name}.jsonl"
    with path.open("w") as f:
        for r in results:
            f.write(json.dumps({"run_id": run_id, "task": task.name, "settings": settings} | r.to_dict()) + "\n")
    return path


def apply_checks(results: list[CaseResult], cases: list[dict], specs: list[dict]) -> None:
    """Run the automatic checks on every finished output (failed chains have no output to check)."""
    inputs = {c["id"]: c["inputs"] for c in cases}
    for r in results:
        if r.output is not None:
            r.checks = run_checks(r.output, inputs[r.case_id], specs)


def run(task: Task, cases: list[dict], models: list[str], *, settings: dict, mock=False, use_cache=True,
        store=None, prices=None, check_specs: list[dict] | None = None):
    run_id = new_run_id()
    results = asyncio.run(run_matrix(task, cases, models, settings=settings, store=store or Store(),
                                     run_id=run_id, mock=mock, use_cache=use_cache, prices=prices))
    apply_checks(results, cases, check_specs or [])
    return run_id, results, write_results(run_id, task, settings, results)
