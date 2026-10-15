import asyncio
import json

import pytest

from bench import compare as compare_mod
from bench.models import CallResult
from bench.store import Store

ORIGINAL_ACALL = compare_mod.acall_model
MODELS = ["anthropic/claude-haiku-4-5", "openai/gpt-5-mini", "gemini/gemini-3.8-flash"]
SETTINGS = {"temperature": 0.2, "max_tokens": 256}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "bench.db")


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_mod, "RUNS_DIR", tmp_path / "runs")


def test_mock_run_returns_one_result_per_model_in_order(store):
    run_id, results, path = compare_mod.run(MODELS, "hi", settings=SETTINGS, mock=True, store=store)
    assert [r.model for r in results] == MODELS
    assert all(r.text == f"Mock answer from {r.model}." for r in results)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["model"] for row in rows] == MODELS
    assert all(row["run_id"] == run_id for row in rows)


def test_second_identical_run_is_served_from_cache_at_no_cost(store):
    compare_mod.run(MODELS, "hi", settings=SETTINGS, mock=True, store=store)
    _, results, _ = compare_mod.run(MODELS, "hi", settings=SETTINGS, mock=True, store=store)
    assert all(r.cached and r.cost_usd == 0 for r in results)


def test_changing_settings_misses_the_cache(store):
    compare_mod.run(MODELS, "hi", settings=SETTINGS, mock=True, store=store)
    _, results, _ = compare_mod.run(MODELS, "hi", settings=SETTINGS | {"temperature": 0.7}, mock=True, store=store)
    assert not any(r.cached for r in results)


def test_one_failing_model_does_not_stop_the_others(store, monkeypatch):
    async def fake(model, prompt, **kw):
        if model.startswith("openai"):
            return CallResult(model, "", 0, 0, None, 0.0, error="AuthenticationError: no key")
        return CallResult(model, "ok", 1, 1, 0.001, 0.1)

    monkeypatch.setattr(compare_mod, "acall_model", fake)
    _, results, _ = compare_mod.run(MODELS, "hi", settings=SETTINGS, store=store)
    assert [bool(r.error) for r in results] == [False, True, False]


def test_errors_are_never_served_from_cache(store, monkeypatch):
    async def failing(model, prompt, **kw):
        return CallResult(model, "", 0, 0, None, 0.0, error="RateLimitError")

    monkeypatch.setattr(compare_mod, "acall_model", failing)
    compare_mod.run(MODELS[:1], "hi", settings=SETTINGS, store=store)
    monkeypatch.setattr(compare_mod, "acall_model", ORIGINAL_ACALL)
    _, results, _ = compare_mod.run(MODELS[:1], "hi", settings=SETTINGS, mock=True, store=store)
    assert not results[0].cached and not results[0].error


def test_mock_answers_are_never_served_to_a_real_run(store, monkeypatch):
    compare_mod.run(MODELS[:1], "hi", settings=SETTINGS, mock=True, store=store)

    async def real(model, prompt, **kw):
        return CallResult(model, "real answer", 1, 1, 0.001, 0.1)

    monkeypatch.setattr(compare_mod, "acall_model", real)
    _, results, _ = compare_mod.run(MODELS[:1], "hi", settings=SETTINGS, store=store)
    assert results[0].text == "real answer" and not results[0].cached


def test_configured_price_fills_in_unknown_cost(store, monkeypatch):
    async def unpriced(model, prompt, **kw):
        return CallResult(model, "ok", 1000, 2000, None, 0.1)

    monkeypatch.setattr(compare_mod, "acall_model", unpriced)
    prices = {MODELS[0]: {"input": 1.0, "output": 2.0}}
    _, results, _ = compare_mod.run(MODELS[:1], "hi", settings=SETTINGS, store=store, prices=prices)
    assert results[0].cost_usd == pytest.approx(0.005)
