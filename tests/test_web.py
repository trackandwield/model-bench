import json

import pytest
from fastapi.testclient import TestClient

from bench import compare
from bench.web import app as web

RUN = "20261017-090000-battlecard"
CARD = "## Pricing\nAcme is $18 per host. Customers save 40% on average.\n"


def row(model, case, repeat=1, output=CARD, judge=None, unsourced=("40%",)):
    r = {"run_id": "20261017-090000", "task": "battlecard", "model": model, "case_id": case, "repeat": repeat,
         "output": output, "failed_step": None if output else "extract", "cost_usd": 0.005, "latency_s": 10.0,
         "checks": {"pricing_matches_source": {"passed": True, "detail": []},
                    "unsourced_numbers": {"passed": not unsourced, "detail": list(unsourced)}},
         "steps": [{"name": "extract", "cached": False, "error": None if output else "503 high demand",
                    "input_tokens": 10, "output_tokens": 10}]}
    if judge is not None:
        r["judge"] = {"no_invented_facts": {"score": judge, "passed": judge >= 4, "reasoning": "The savings claim isn't in the profile.",
                                            "evidence": ["Customers save 40% on average."]}}
    return r


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "RUNS_DIR", tmp_path)
    rows = [row(m, f"bc-00{i}", rep, judge=2 if i == 1 and rep == 1 else None)
            for m in ("anthropic/claude-haiku-4-5", "openai/gpt-5-mini") for i in range(1, 7) for rep in (1, 2)]
    rows.append(row("gemini/gemini-3.8-flash", "bc-001", output=None))
    (tmp_path / f"{RUN}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / f"{RUN}.config.json").write_text(json.dumps({"mock": False, "metrics_locked": True}))
    return TestClient(web.app)


def test_saved_runs_lists_task_runs(client):
    page = client.get("/runs")
    assert page.status_code == 200
    assert "20261017-090000" in page.text and "claude-haiku-4-5" in page.text


def test_results_page_shows_the_analysis(client):
    page = client.get(f"/runs/{RUN}")
    assert page.status_code == 200
    for text in ("Recommendation", "Scorecard", "Where each model", "Cost vs quality", "Why they fail", "not ranked"):
        assert text in page.text


def test_side_by_side_highlights_unsourced_numbers_and_judge_quotes(client):
    page = client.get(f"/runs/{RUN}/cases/bc-001?repeat=1")
    assert page.status_code == 200
    assert '<mark class="bad" title="Number not in the profile">40%</mark>' in page.text
    assert "No output: 503 high demand" in page.text  # Gemini's failed chain


def test_failed_judge_quote_is_highlighted_when_no_check_flagged_it(client, tmp_path):
    rows = [row("openai/gpt-5-mini", "bc-001", judge=1, unsourced=())]
    (tmp_path / f"{RUN}.jsonl").write_text(json.dumps(rows[0]) + "\n")
    page = client.get(f"/runs/{RUN}/cases/bc-001")
    assert 'title="Judge: no_invented_facts">Customers save 40% on average.</mark>' in page.text


def test_model_output_cannot_inject_html(client, tmp_path):
    r = row("openai/gpt-5-mini", "bc-001", output="<script>alert(1)</script> hi", unsourced=())
    (tmp_path / f"{RUN}.jsonl").write_text(json.dumps(r) + "\n")
    page = client.get(f"/runs/{RUN}/cases/bc-001")
    assert "<script>alert(1)</script>" not in page.text
    assert "&lt;script&gt;" in page.text


def test_unknown_runs_and_paths_are_not_found(client):
    assert client.get("/runs/nope").status_code == 404
    assert client.get("/runs/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get(f"/runs/{RUN}/cases/bc-999").status_code == 404


def test_missing_repeat_redirects_to_one_that_exists(client):
    page = client.get(f"/runs/{RUN}/cases/bc-001?repeat=9", follow_redirects=False)
    assert page.status_code in (302, 307) and page.headers["location"].endswith("repeat=1")
