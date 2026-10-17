"""Runs and judging started from the browser, done in a background thread.

The web app is one local process, so jobs live in memory: a restart forgets the
progress bar, but the run file and the call cache in runs/ are kept, and an
interrupted run can still be resumed from its config with `bench run --resume`.
"""

from __future__ import annotations

import json
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bench import checks, compare, dataset, judge, metrics, plan, tasks, workflow
from bench.models import configured_prices, format_error, load_config
from bench.store import Store


@dataclass
class Job:
    id: str
    kind: str                  # "run" or "judge"
    run_name: str              # the run file's stem, for the results link
    total: int
    done: int = 0
    failed: int = 0
    status: str = "running"    # running, finished, error
    error: str | None = None
    spent: float = 0.0
    started: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def percent(self) -> int:
        return int(100 * self.done / self.total) if self.total else 100


JOBS: dict[str, Job] = {}
_lock = threading.Lock()


def _start(job: Job, work) -> Job:
    def target():
        try:
            work(job)
            job.status = "finished"
        except Exception as e:  # shown on the progress page rather than lost in a thread
            job.status, job.error = "error", format_error(e)
            traceback.print_exc()

    with _lock:
        JOBS[job.id] = job
    threading.Thread(target=target, daemon=True, name=f"bench-{job.kind}-{job.id}").start()
    return job


def plan_run(task_name: str, models: list[str], *, limit: int | None, repeats: int, mock: bool):
    """Everything a run needs, plus its cost estimate, without making any call."""
    config = load_config()
    settings = {k: config.get("settings", {}).get(k, d) for k, d in (("temperature", 0.2), ("max_tokens", 1024))}
    prices = configured_prices(config)
    task = tasks.load(task_name)
    cases = dataset.load(task_name)[: limit or None]
    est = plan.estimate(task, cases, models, repeats=repeats, settings=settings, store=Store(), prices=prices, mock=mock)
    return task, cases, settings, prices, est


def start_run(task_name: str, models: list[str], *, limit: int | None, repeats: int, mock: bool,
              budget: float | None) -> Job:
    task, cases, settings, prices, _ = plan_run(task_name, models, limit=limit, repeats=repeats, mock=mock)
    run_id = compare.new_run_id()
    config = plan.build_config(run_id, task.name, models, [c["id"] for c in cases], repeats=repeats,
                               settings=settings, budget_usd=budget, mock=mock)
    plan.save_config(config)
    job = Job(run_id, "run", f"{run_id}-{task.name}", total=len(models) * len(cases) * repeats)
    specs = checks.check_specs(metrics.load(task.name))

    def done(result):
        job.done += 1
        job.failed += bool(result.failed_step)

    def work(job):
        cap = workflow.Budget(budget)
        workflow.run(task, cases, models, settings=settings, mock=mock, prices=prices, check_specs=specs,
                     repeats=repeats, budget=cap, run_id=run_id, store=Store(), on_done=done)
        job.spent = cap.spent

    return _start(job, work)


def _judge_setup(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    task_name = rows[0]["task"]
    config = load_config()
    judges = config.get("judges") or []
    settings = {k: (config.get("judge_settings") or {}).get(k, d) for k, d in (("temperature", 0), ("max_tokens", 4096))}
    cases = {c["id"]: c for c in dataset.load(task_name)}
    return rows, metrics.load(task_name), judges, settings, cases, config


def estimate_judging(run_name: str, *, sample: int | None, mock: bool) -> plan.Estimate:
    rows, spec, judges, settings, cases, _ = _judge_setup(compare.RUNS_DIR / f"{run_name}.jsonl")
    return judge.estimate(rows, judge.judge_metrics(spec), judges, cases, settings=settings, store=Store(),
                          mock=mock, sample=sample)


def start_judging(run_name: str, *, sample: int | None, mock: bool, budget: float | None) -> Job:
    path = compare.RUNS_DIR / f"{run_name}.jsonl"
    rows, spec, judges, settings, cases, _ = _judge_setup(path)
    est = judge.estimate(rows, judge.judge_metrics(spec), judges, cases, settings=settings, store=Store(),
                         mock=mock, sample=sample)
    job = Job(f"judge-{compare.new_run_id()}", "judge", run_name, total=est.calls)

    def work(job):
        cap = workflow.Budget(budget)
        _, first, _ = judge.run(rows, spec, judges, cases, settings=settings, store=Store(), mock=mock,
                                sample=sample, budget=cap)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        job.done, job.failed, job.spent = len(first), sum(1 for _, j in first if j.error), cap.spent

    return _start(job, work)
