import json

import pytest

from bench import cli, compare as compare_mod, dataset, plan, tasks, workflow
from bench.models import CallResult
from bench.store import Store

MODELS = ["anthropic/claude-haiku-4-5", "openai/gpt-5-mini"]
SETTINGS = {"temperature": 0.2, "max_tokens": 256}


@pytest.fixture
def store(tmp_path, monkeypatch):
    for mod in (compare_mod, workflow, plan):
        monkeypatch.setattr(mod, "RUNS_DIR", tmp_path / "runs")
    return Store(tmp_path / "bench.db")


def paid_calls(monkeypatch, cost=0.01):
    calls = []

    async def fake(model, prompt, *, system, temperature, max_tokens, mock_response):
        calls.append((model, prompt))
        return CallResult(model, f"answer {len(calls)}", 100, 200, cost, 0.1)
    monkeypatch.setattr(compare_mod, "acall_model", fake)
    return calls


def test_repeats_are_separate_calls(store, monkeypatch):
    calls = paid_calls(monkeypatch)
    task, cases = tasks.load("battlecard"), dataset.load("battlecard")[:1]
    _, results, _ = workflow.run(task, cases, MODELS, settings=SETTINGS, store=store, repeats=3)
    assert [(r.model, r.repeat) for r in results] == [(m, r) for m in MODELS for r in (1, 2, 3)]
    assert len(calls) == 2 * 3 * 2  # models × repeats × steps
    assert len({r.output for r in results}) == 6


def test_budget_stops_new_calls(store, monkeypatch):
    calls = paid_calls(monkeypatch, cost=1.0)
    task, cases = tasks.load("battlecard"), dataset.load("battlecard")[:3]
    cap = workflow.Budget(1.5)
    _, results, _ = workflow.run(task, cases, MODELS[:1], settings=SETTINGS, store=store, budget=cap)
    assert len(calls) == 2
    assert all("Budget" in next(s.error for s in r.steps if s.error) for r in results[1:])


def test_estimate_counts_cached_steps_as_free(store, monkeypatch):
    paid_calls(monkeypatch)
    task, cases = tasks.load("battlecard"), dataset.load("battlecard")[:2]
    before = plan.estimate(task, cases, MODELS, repeats=2, settings=SETTINGS, store=store)
    assert (before.calls, before.cached) == (16, 0) and before.cost_usd > 0
    workflow.run(task, cases[:1], MODELS, settings=SETTINGS, store=store, repeats=2)
    after = plan.estimate(task, cases, MODELS, repeats=2, settings=SETTINGS, store=store)
    assert (after.calls, after.cached) == (16, 8)
    assert after.cost_usd < before.cost_usd


def test_estimate_uses_past_output_lengths(store, monkeypatch):
    paid_calls(monkeypatch)
    task, cases = tasks.load("battlecard"), dataset.load("battlecard")[:2]
    workflow.run(task, cases[:1], MODELS[:1], settings=SETTINGS, store=store)
    est = plan.estimate(task, cases[1:], MODELS[:1], repeats=1, settings=SETTINGS, store=store)
    assert MODELS[0] not in est.guessed  # 200 output tokens per step from history


def test_unpriced_model_is_reported_not_guessed(store):
    task, cases = tasks.load("battlecard"), dataset.load("battlecard")[:1]
    est = plan.estimate(task, cases, ["openrouter/some/unknown-model"], repeats=1, settings=SETTINGS, store=store)
    assert est.unpriced == {"openrouter/some/unknown-model"} and est.cost_usd == 0


def test_config_records_what_defines_the_run(store):
    config = plan.build_config("r1", "battlecard", MODELS, ["bc-001"], repeats=3, settings=SETTINGS,
                               budget_usd=2.0, mock=False)
    for key in ("task_sha256", "dataset_sha256", "metrics_sha256", "metrics_locked", "git", "versions", "created_at"):
        assert key in config
    assert plan.changed_since(config) == []
    assert plan.changed_since(config | {"dataset_sha256": "old"}) == ["cases.jsonl"]


def test_over_budget_run_is_refused(store, monkeypatch, capsys):
    calls = paid_calls(monkeypatch)
    monkeypatch.setattr(cli, "Store", lambda: store)
    code = cli.main(["run", "battlecard", "--limit", "20", "--repeats", "3", "--budget", "0.0001"])
    assert code == 1 and calls == []
    assert "over the $0.00 budget" in capsys.readouterr().out


def test_resume_finishes_only_what_failed(store, monkeypatch):
    calls = []

    async def flaky(model, prompt, *, system, temperature, max_tokens, mock_response):
        calls.append(model)
        if model.startswith("openai") and calls.count(model) == 1:  # only the first OpenAI call fails
            return CallResult(model, "", 0, 0, None, 0.1, error="RateLimitError")
        return CallResult(model, "ok", 1, 1, 0.001, 0.1)
    monkeypatch.setattr(compare_mod, "acall_model", flaky)
    monkeypatch.setattr(cli, "Store", lambda: store)
    args = ["run", "battlecard", "--limit", "1", "--model", MODELS[0], "--model", MODELS[1]]
    cli.main(args)
    config_file = next(plan.RUNS_DIR.glob("*.config.json"))
    first = len(calls)
    cli.main(["run", "--resume", str(config_file)])
    retried = calls[first:]
    assert retried and all(m.startswith("openai") for m in retried)  # Claude's chain came from the cache
    run_file = config_file.with_name(config_file.name.replace(".config.json", ".jsonl"))
    rows = [json.loads(line) for line in run_file.read_text().splitlines()]
    assert all(r["failed_step"] is None for r in rows)
