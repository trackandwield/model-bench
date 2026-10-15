import pytest

from bench import compare as compare_mod
from bench import dataset, tasks, workflow
from bench.models import CallResult
from bench.store import Store

MODELS = ["anthropic/claude-haiku-4-5", "openai/gpt-5-mini"]
SETTINGS = {"temperature": 0.2, "max_tokens": 256}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(workflow, "RUNS_DIR", tmp_path / "runs")
    return Store(tmp_path / "bench.db")


def test_battlecard_is_a_two_step_workflow():
    task = tasks.load("battlecard")
    assert task.kind == "workflow"
    assert [s.name for s in task.steps] == ["extract", "draft"]


def test_every_case_renders_without_missing_variables():
    task = tasks.load("battlecard")
    for case in dataset.load("battlecard"):
        task.steps[0].render(case["inputs"], {})
        task.steps[1].render(case["inputs"], {"extract": "facts"})


def test_missing_variable_is_an_error_not_a_blank():
    step = tasks.Step("s", "Hello {{ nme }}")
    with pytest.raises(Exception):
        step.render({"name": "x"}, {})


def test_step_cannot_use_a_later_step():
    spec = {"steps": [{"name": "a", "prompt": "{{ steps.b }}"}, {"name": "b", "prompt": "x"}]}
    assert any("hasn't run yet" in p for p in tasks.validate(spec))


def test_second_step_receives_first_steps_output(store):
    task = tasks.load("battlecard")
    case = dataset.load("battlecard")[0]
    _, results, path = workflow.run(task, [case], MODELS, settings=SETTINGS, mock=True, store=store)
    assert [r.model for r in results] == MODELS
    for r in results:
        assert r.failed_step is None
        assert f"Mock extract from {r.model}" in r.steps[1].prompt
        assert r.output == f"Mock draft from {r.model} for {case['id']}."
    assert path.exists()


def test_chain_stops_at_the_failing_step(store, monkeypatch):
    async def flaky(model, prompt, **kw):
        if "battlecard" in prompt:  # the draft step
            return CallResult(model, "", 0, 0, None, 0.0, error="RateLimitError")
        return CallResult(model, "facts", 1, 1, 0.001, 0.1)

    monkeypatch.setattr(compare_mod, "acall_model", flaky)
    task = tasks.load("battlecard")
    case = dataset.load("battlecard")[0]
    _, results, _ = workflow.run(task, [case], MODELS[:1], settings=SETTINGS, store=store)
    assert results[0].failed_step == "draft"
    assert results[0].output is None
    assert len(results[0].steps) == 2


def test_results_keep_model_then_case_order(store):
    task = tasks.load("battlecard")
    cases = dataset.load("battlecard")[:3]
    _, results, _ = workflow.run(task, cases, MODELS, settings=SETTINGS, mock=True, store=store)
    assert [(r.model, r.case_id) for r in results] == [(m, c["id"]) for m in MODELS for c in cases]


def test_truncated_output_counts_as_a_failed_step(store, monkeypatch):
    async def truncated(model, prompt, **kw):
        return CallResult(model, "half the fa", 1, 1, 0.001, 0.1, finish_reason="length")

    monkeypatch.setattr(compare_mod, "acall_model", truncated)
    task = tasks.load("battlecard")
    case = dataset.load("battlecard")[0]
    _, results, _ = workflow.run(task, [case], MODELS[:1], settings=SETTINGS, store=store)
    assert results[0].failed_step == "extract"
    assert "max_tokens" in results[0].steps[0].error


def test_checks_run_on_finished_outputs_only(store, monkeypatch):
    from bench import checks, metrics

    card = "## Overview\nx\n## Differentiators\nx\n## Pricing\nAt $49 per seat.\n## Objection handlers\nx\n## Landmines\nx"

    async def fake(model, prompt, **kw):
        if model.startswith("openai") and "battlecard" in prompt:
            return CallResult(model, "", 0, 0, None, 0.0, error="RateLimitError")
        return CallResult(model, card, 1, 1, 0.001, 0.1)

    monkeypatch.setattr(compare_mod, "acall_model", fake)
    task = tasks.load("battlecard")
    case = dataset.load("battlecard")[0]
    specs = checks.check_specs(metrics.load("battlecard"))
    _, results, _ = workflow.run(task, [case], MODELS, settings=SETTINGS, store=store, check_specs=specs)
    claude, gpt = results
    assert claude.checks["required_sections"].passed
    assert not claude.checks["pricing_matches_source"].passed  # $49 isn't in the profiles
    assert claude.checks["pricing_matches_source"].detail == ["$49"]
    assert gpt.checks == {}
