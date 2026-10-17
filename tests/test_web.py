import json
import time

import pytest
from fastapi.testclient import TestClient

from bench import compare
from bench.store import Store
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


@pytest.fixture
def runs_here(tmp_path, monkeypatch):
    from bench import plan, workflow
    for mod in (compare, workflow, plan):
        monkeypatch.setattr(mod, "RUNS_DIR", tmp_path)
    monkeypatch.setattr("bench.web.jobs.Store", lambda: Store(tmp_path / "bench.db"))
    return tmp_path


def wait(job, seconds=20):
    end = time.time() + seconds
    while job.status == "running" and time.time() < end:
        time.sleep(0.05)
    return job


def test_run_builder_estimates_without_running(runs_here):
    client = TestClient(web.app)
    page = client.post("/new", data={"task": "battlecard", "action": "estimate", "models": ["openai/gpt-5-mini"],
                                     "limit": "2", "repeats": "1", "budget": "3", "mock": "true"})
    assert page.status_code == 200 and "model calls" in page.text
    assert not list(runs_here.glob("*.jsonl"))


def test_practice_run_from_the_browser_then_judge(runs_here):
    from bench.web import jobs
    client = TestClient(web.app)
    page = client.post("/new", data={"task": "battlecard", "action": "run", "models": ["openai/gpt-5-mini", "anthropic/claude-haiku-4-5"],
                                     "limit": "2", "repeats": "2", "budget": "3", "mock": "true"}, follow_redirects=False)
    assert page.status_code == 303
    job = wait(jobs.JOBS[page.headers["location"].rsplit("/", 1)[1]])
    assert (job.status, job.done, job.total) == ("finished", 8, 8)
    assert client.get(page.headers["location"]).text.count("See results") == 1
    results = client.get(f"/runs/{job.run_name}?judge_sample=1")
    assert "judge calls" in results.text
    judged = client.post(f"/runs/{job.run_name}/judge", data={"sample": "1", "mock": "true"}, follow_redirects=False)
    jjob = wait(jobs.JOBS[judged.headers["location"].rsplit("/", 1)[1]])
    assert jjob.status == "finished"
    rows = [json.loads(l) for l in (runs_here / f"{job.run_name}.jsonl").read_text().splitlines()]
    assert sum("judge" in r for r in rows) == 2  # one card per model


def test_real_run_needs_api_keys(runs_here, monkeypatch):
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    client = TestClient(web.app)
    page = client.post("/new", data={"task": "battlecard", "action": "run", "models": ["openai/gpt-5-mini"],
                                     "limit": "1", "repeats": "1", "budget": "3"})
    assert page.status_code == 400 and "No API key" in page.text


def test_run_over_budget_is_refused(runs_here, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    client = TestClient(web.app)
    page = client.post("/new", data={"task": "battlecard", "action": "run", "models": ["openai/gpt-5-mini"],
                                     "repeats": "3", "budget": "0.0001"})
    assert page.status_code == 400 and "over the $0.00 budget" in page.text


def review_rows(tmp_path, n=3):
    rows = [row("openai/gpt-5-mini", f"bc-00{i}", judge=2, unsourced=()) for i in range(1, n + 1)]
    (tmp_path / f"{RUN}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


SCORES = {"score_no_invented_facts": "2", "score_specific_differentiators": "4",
          "score_objection_handling": "3", "score_rep_ready": "5"}


def test_review_card_is_blind(client, tmp_path):
    review_rows(tmp_path)
    page = client.get(f"/review/{RUN}")
    assert page.status_code == 200 and "3 to go" in page.text
    assert "gpt-5-mini" not in page.text  # no model name
    assert "The savings claim" not in page.text and "<mark" not in page.text  # no judge reasoning or highlights
    assert "Every claim traces to the profiles" in page.text  # the judge's own rubric


def test_review_saves_scores_and_moves_on(client, tmp_path):
    review_rows(tmp_path)
    pos = client.get(f"/review/{RUN}").text.split('name="pos" value="')[1].split('"')[0]
    page = client.post(f"/review/{RUN}", data={"pos": pos, "action": "save", "reviewer": "Sam", **SCORES},
                       follow_redirects=False)
    assert page.status_code == 303 and page.cookies.get("reviewer") == "Sam"
    saved = [json.loads(l) for l in (tmp_path / f"{RUN}.reviews.jsonl").read_text().splitlines()]
    assert saved[0]["scores"]["rep_ready"] == 5 and saved[0]["reviewer"] == "Sam"
    assert "2 to go" in client.get(f"/review/{RUN}").text


def test_review_needs_every_score_unless_skipped(client, tmp_path):
    review_rows(tmp_path)
    page = client.post(f"/review/{RUN}", data={"pos": "0", "action": "save", "score_rep_ready": "4"})
    assert page.status_code == 400 and "Give a score for no invented facts" in page.text
    assert 'value="4" checked' in page.text  # keeps what was picked
    skip = client.post(f"/review/{RUN}", data={"pos": "0", "action": "skip"}, follow_redirects=False)
    assert skip.status_code == 303
    assert json.loads((tmp_path / f"{RUN}.reviews.jsonl").read_text())["skipped"] is True


def test_finished_queue_shows_agreement_and_results_show_trust(client, tmp_path):
    review_rows(tmp_path)
    assert "Judge not checked against a person yet (0 of 20 reviews)" in client.get(f"/runs/{RUN}").text
    for pos in range(3):
        client.post(f"/review/{RUN}", data={"pos": str(pos), "action": "save", **SCORES})
    done = client.get(f"/review/{RUN}")
    assert "All judged cards reviewed" in done.text
    assert "100%" in done.text  # judge said 2, person said 2
    results = client.get(f"/runs/{RUN}")
    assert "(3 of 20 reviews)" in results.text
    assert "rep_ready" in results.text  # human metric now measured
