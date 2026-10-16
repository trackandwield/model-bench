import asyncio
import json

import pytest

from bench import dataset, judge, metrics
from bench.models import CallResult
from bench.store import Store

JUDGES = ["anthropic/claude-sonnet-5-5", "openai/gpt-5.6"]
SETTINGS = {"temperature": 0, "max_tokens": 256}
CARD = "## Overview\nAcme beats Northwind on setup.\n## Pricing\nAcme is $18 per host."


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "bench.db")


def rows_for(models):
    return [{"model": m, "case_id": "bc-001", "output": CARD, "checks": {}} for m in models]


def cases():
    return {c["id"]: c for c in dataset.load("battlecard")}


def test_battlecard_judge_metrics_have_rubrics():
    specs = judge.judge_metrics(metrics.load("battlecard"))
    assert [m["name"] for m in specs] == ["no_invented_facts", "specific_differentiators", "objection_handling"]
    assert all(sorted(m["rubric"]) == [1, 2, 3, 4, 5] for m in specs)


def test_judge_is_always_from_another_family():
    assert judge.pick_judge("anthropic/claude-haiku-4-5", JUDGES) == "openai/gpt-5.6"
    assert judge.pick_judge("openai/gpt-5-mini", JUDGES) == "anthropic/claude-sonnet-5-5"
    assert judge.pick_judge("gemini/gemini-3.8-flash", JUDGES) == "anthropic/claude-sonnet-5-5"
    assert judge.pick_judge("anthropic/x", ["anthropic/y"]) is None


def test_prompt_is_blind_to_the_model():
    metric = judge.judge_metrics(metrics.load("battlecard"))[0]
    prompt = judge.build_prompt(metric, {"our_product": "Acme"}, "As Claude, made by Anthropic, I think GPT-5 is worse.")
    assert "Claude" not in prompt and "Anthropic" not in prompt and "GPT-5" not in prompt
    assert "[model]" in prompt


def test_flags_from_automatic_checks_reach_the_judge():
    metric = judge.judge_metrics(metrics.load("battlecard"))[0]
    flagged = {"unsourced_numbers": {"passed": False, "detail": ["$28,800"]}}
    assert "$28,800" in judge.build_prompt(metric, {"x": "y"}, CARD, flagged)
    assert "automatic_flag" not in judge.build_prompt(metric, {"x": "y"}, CARD, {})


def test_parse_reads_json_even_inside_a_code_fence():
    score, reasoning, evidence = judge.parse('```json\n{"evidence": ["a"], "reasoning": "ok", "score": 4}\n```')
    assert (score, reasoning, evidence) == (4, "ok", ["a"])
    with pytest.raises(ValueError):
        judge.parse('{"score": 7}')


def test_pass_bars():
    assert judge.passes(4, ">= 4") and not judge.passes(3, ">= 4")
    assert judge.passes(None, ">= 4") is None


def test_mock_run_scores_every_output_on_every_metric(store):
    rows = rows_for(["anthropic/claude-haiku-4-5", "openai/gpt-5-mini"])
    rows.append({"model": "x/failed", "case_id": "bc-001", "output": None})
    _, first, _ = judge.run(rows, metrics.load("battlecard"), JUDGES, cases(), settings=SETTINGS,
                            store=store, mock=True)
    assert len(first) == 6
    for row in rows[:2]:
        assert set(row["judge"]) == {"no_invented_facts", "specific_differentiators", "objection_handling"}
        assert all(1 <= j["score"] <= 5 and j["passed"] is not None for j in row["judge"].values())
    assert rows[0]["judge"]["no_invented_facts"]["judge_model"] == "openai/gpt-5.6"
    assert "judge" not in rows[2]


def test_invented_quotes_are_flagged(store, monkeypatch):
    async def fake(model, prompt, **kw):
        reply = {"evidence": ["Acme is $18 per host.", "Acme has 900 customers."], "reasoning": "r", "score": 3}
        return CallResult(model, json.dumps(reply), 1, 1, 0.001, 0.1)
    monkeypatch.setattr("bench.compare.acall_model", fake)
    rows = rows_for(["openai/gpt-5-mini"])
    _, first, _ = judge.run(rows, metrics.load("battlecard"), JUDGES, cases(), settings=SETTINGS, store=store)
    assert first[0][1].missing_quotes == ["Acme has 900 customers."]


def test_unreadable_reply_is_an_error_not_a_score(store, monkeypatch):
    async def fake(model, prompt, **kw):
        return CallResult(model, "I'd give it a solid four.", 1, 1, 0.001, 0.1)
    monkeypatch.setattr("bench.compare.acall_model", fake)
    rows = rows_for(["openai/gpt-5-mini"])
    _, first, _ = judge.run(rows, metrics.load("battlecard"), JUDGES, cases(), settings=SETTINGS, store=store)
    assert all(j.score is None and "Unreadable" in j.error for _, j in first)


def test_repeats_make_new_calls_and_measure_agreement(store, monkeypatch):
    calls = []

    async def fake(model, prompt, *, system, temperature, max_tokens, mock_response):  # the real signature
        calls.append(prompt)
        score = 4 if len(calls) % 2 else 3
        return CallResult(model, json.dumps({"evidence": [], "reasoning": "r", "score": score}), 1, 1, 0.001, 0.1)
    monkeypatch.setattr("bench.compare.acall_model", fake)
    rows = rows_for(["openai/gpt-5-mini"])
    _, first, repeats = judge.run(rows, metrics.load("battlecard"), JUDGES, cases(), settings=SETTINGS,
                                  store=store, repeat_sample=3)
    assert len(calls) == 6  # repeats are not served from the cache
    c = judge.consistency(first, repeats)
    assert c["pairs"] == 3 and c["within_1"] == 1.0
    assert all("repeat_score" in j for j in rows[0]["judge"].values())


def test_no_cross_family_judge_is_reported(store):
    rows = rows_for(["anthropic/claude-haiku-4-5"])
    _, first, _ = judge.run(rows, metrics.load("battlecard"), ["anthropic/claude-sonnet-5-5"], cases(),
                            settings=SETTINGS, store=store, mock=True)
    assert all("different model family" in j.error for _, j in first)


def test_estimate_counts_cached_judgments_as_free(store):
    rows = rows_for(["openai/gpt-5-mini", "anthropic/claude-haiku-4-5"])
    specs = judge.judge_metrics(metrics.load("battlecard"))
    before = judge.estimate(rows, specs, JUDGES, cases(), settings=SETTINGS, store=store, repeat_sample=2)
    assert (before.calls, before.cached) == (8, 0) and before.cost_usd > 0
    judge.run(rows, metrics.load("battlecard"), JUDGES, cases(), settings=SETTINGS, store=store, mock=True)
    mocked = judge.estimate(rows, specs, JUDGES, cases(), settings=SETTINGS, store=store, mock=True)
    assert mocked.cached == mocked.calls == 6


def test_sample_judges_only_some_outputs(store):
    rows = rows_for(["openai/gpt-5-mini"] * 5)
    _, first, _ = judge.run(rows, metrics.load("battlecard"), JUDGES, cases(), settings=SETTINGS,
                            store=store, mock=True, sample=2)
    assert len({i for i, _ in first}) == 2
    assert sum("judge" in r for r in rows) == 2


def test_budget_stops_new_judge_calls(store, monkeypatch):
    calls = []

    async def fake(model, prompt, *, system, temperature, max_tokens, mock_response):
        calls.append(model)
        await asyncio.sleep(0.01)  # like a network call, so several are in flight at once
        return CallResult(model, json.dumps({"evidence": [], "reasoning": "r", "score": 4}), 1, 1, 1.0, 0.1)
    monkeypatch.setattr("bench.compare.acall_model", fake)
    from bench.workflow import Budget
    rows = rows_for(["openai/gpt-5-mini"] * 3)  # 9 judgments, all by one judge
    for n, row in enumerate(rows):
        row["output"] += f"\nCard {n}."  # distinct outputs, so nothing comes from the cache
    cap = Budget(0.5)
    _, first, _ = judge.run(rows, metrics.load("battlecard"), JUDGES, cases(), settings=SETTINGS,
                            store=store, budget=cap)
    # Up to 4 calls per provider run at once, so those finish; nothing starts after the cap.
    assert len(calls) == 4
    assert sum(1 for _, j in first if j.error and j.error.startswith("Budget")) == 5
