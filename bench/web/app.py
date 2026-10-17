"""FastAPI app: pages for saved runs, a run's results, and one case side by side.

Every page is computed from the run files in runs/, the same way `bench analyze`
does it, so the browser and the terminal never disagree.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import litellm
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt

from bench import analyze, compare, dataset, metrics, review, tasks
from bench.models import configured_prices, load_config
from bench.web import jobs

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=HERE / "templates")
app = FastAPI(title="Model Bench")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

# Model output is shown as Markdown, never as raw HTML.
_md = MarkdownIt("commonmark", {"html": False}).enable("table")
_RUN_NAME = re.compile(r"^\d{8}-\d{6}-[\w-]+$")


def short(model: str) -> str:
    return model.split("/")[-1]


templates.env.filters["short"] = short


def runs_dir() -> Path:
    return compare.RUNS_DIR


def run_path(name: str) -> Path:
    if not _RUN_NAME.match(name):
        raise HTTPException(404, "No such run")
    path = runs_dir() / f"{name}.jsonl"
    if not path.exists():
        raise HTTPException(404, "No such run")
    return path


def load_rows(name: str) -> list[dict]:
    path = run_path(name)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_config_file(name: str) -> dict:
    path = runs_dir() / f"{name}.config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def saved_patterns(name: str) -> dict:
    """Failure patterns from an earlier `bench analyze --patterns`, if it was run."""
    path = runs_dir() / f"{name}.analysis.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("patterns") or {}


def list_runs() -> list[dict]:
    """Every task run in runs/, newest first. Single-prompt `compare` runs have no task and are skipped."""
    out = []
    for path in sorted(runs_dir().glob("*.jsonl"), reverse=True):
        if not _RUN_NAME.match(path.stem):
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows or "task" not in rows[0] or "case_id" not in rows[0]:
            continue
        config = load_config_file(path.stem)
        out.append({
            "name": path.stem,
            "run_id": rows[0]["run_id"],
            "task": rows[0]["task"],
            "models": list(dict.fromkeys(r["model"] for r in rows)),
            "cases": len({r["case_id"] for r in rows}),
            "repeats": max(r.get("repeat", 1) for r in rows),
            "finished": sum(1 for r in rows if r.get("output") is not None),
            "total": len(rows),
            "judged": sum(1 for r in rows if r.get("judge")),
            "mock": config.get("mock", False),
            "locked": config.get("metrics_locked"),
        })
    return out


def analyze_run(name: str) -> tuple[list[dict], dict, dict, dict]:
    rows = load_rows(name)
    task = rows[0]["task"]
    spec = metrics.load(task)
    cases = {c["id"]: c for c in dataset.load(task)}
    analyze.fill_prices(rows, configured_prices(load_config()))
    reviews = review.load(run_path(name))
    review.merge(rows, reviews)
    result = analyze.analyze(rows, spec, cases)
    result["review"] = review.summary(rows, reviews, spec)
    return rows, spec, cases, result


def heat_style(score: float | None) -> str:
    """Background and text color for a 0-1 score: orange is weak, deep blue is strong."""
    if score is None:
        return "background: var(--empty); color: var(--muted);"
    for bar, bg, fg in ((0.8, "#2449B8", "#FFFFFF"), (0.65, "#8FA8E8", "#16181D"), (0.5, "#DCE4F8", "#16181D")):
        if score >= bar:
            return f"background: {bg}; color: {fg};"
    return "background: #FBE3CF; color: #7A3608;"


def scatter(card: dict, frontier: list[str]) -> dict:
    """Points for the cost vs quality chart. Models with no price sit in their own lane on the right."""
    ranked = {m: v for m, v in card.items() if v["ranked"] and v["score"] is not None}
    costs = [v["cost_per_output"] for v in ranked.values() if v["cost_per_output"] is not None]
    scores = [v["score"] for v in ranked.values()]
    if not ranked:
        return {"points": [], "line": "", "lo": 0, "hi": 1, "max_cost": 0}
    lo, hi = max(0.0, min(scores) - 0.1), min(1.0, max(scores) + 0.1)
    max_cost = max(costs) * 1.15 if costs else 1.0
    x0, x1, y0, y1 = 56, 330, 200, 24

    def y(s):
        return y1 + (hi - s) / ((hi - lo) or 1) * (y0 - y1)

    points = []
    for m, v in ranked.items():
        priced = v["cost_per_output"] is not None
        x = x0 + v["cost_per_output"] / max_cost * (x1 - x0) if priced else 430
        points.append({"model": m, "x": round(x, 1), "y": round(y(v["score"]), 1), "priced": priced,
                       "best": m in frontier, "score": v["score"], "cost": v["cost_per_output"]})
    line = " ".join(f"{p['x']},{p['y']}" for p in sorted(points, key=lambda p: p["x"]) if p["best"] and p["priced"])
    return {"points": points, "line": line, "lo": lo, "hi": hi, "max_cost": max_cost}


def _marks(text: str, needles: list[tuple[str, str]]) -> str:
    """Rendered Markdown with each problem phrase highlighted. Needles are (phrase, why)."""
    out = _md.render(text)
    for phrase, why in needles:
        phrase = phrase.replace("**", "").strip().strip('"“”').strip()
        if len(phrase) < 2:
            continue
        for part in (p.strip() for p in re.split(r"\s*(?:\.\.\.|…)\s*", phrase)):
            esc = html.escape(part, quote=False)
            if len(part) >= 2 and esc in out:
                out = out.replace(esc, f'<mark class="bad" title="{html.escape(why)}">{esc}</mark>', 1)
    return out


def card_view(row: dict, spec: dict) -> dict:
    """One model's output for the side-by-side page: rendered card, checks, judge notes."""
    judged = [m for m in spec["metrics"] if m["type"] == "judge"]
    checked = [m for m in spec["metrics"] if m["type"] == "check"]
    needles: list[tuple[str, str]] = []
    checks = []
    for m in checked:
        c = (row.get("checks") or {}).get(m["name"])
        if c is None:
            continue
        checks.append({"name": m["name"], "passed": c["passed"], "detail": c.get("detail") or []})
        if not c["passed"] and m["name"] == "pricing_matches_source":
            needles += [(str(d), "Price not in the profile") for d in c.get("detail") or []]
    extra = (row.get("checks") or {}).get("unsourced_numbers")
    if extra and not extra["passed"]:
        needles += [(str(d), "Number not in the profile") for d in extra.get("detail") or []]
    judge = []
    for m in judged:
        j = (row.get("judge") or {}).get(m["name"])
        if not j:
            continue
        judge.append({"name": m["name"], "score": j.get("score"), "passed": j.get("passed"),
                      "reasoning": j.get("reasoning") or j.get("error") or "", "evidence": j.get("evidence") or []})
        if j.get("passed") is False:
            needles += [(q, f"Judge: {m['name']}") for q in j.get("evidence") or []]
    output = row.get("output")
    score = analyze.output_score(row, [m for m in spec["metrics"] if m["type"] in ("check", "judge", "human")])
    return {
        "model": row["model"],
        "html": _marks(output, needles) if output is not None else None,
        "error": None if output is not None else next((s.get("error") for s in row.get("steps", []) if s.get("error")), "Failed"),
        "checks": checks,
        "judge": judge,
        "score": score,
        "cost": row.get("cost_usd"),
        "latency": row.get("latency_s"),
    }


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/runs")


@app.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request):
    return templates.TemplateResponse(request, "runs.html", {"runs": list_runs(), "nav": "runs"})


@app.get("/runs/{name}", response_class=HTMLResponse)
def results_page(request: Request, name: str, judge_sample: str | None = None, over: int = 0):
    rows, spec, cases, result = analyze_run(name)
    config = load_config_file(name)
    n = int(judge_sample) if judge_sample and judge_sample.isdigit() and int(judge_sample) > 0 else None
    judge_est = (jobs.estimate_judging(name, sample=n, mock=bool(config.get("mock")))
                 if judge_sample is not None else None)
    card = result["scorecard"]
    models = list(card)
    ranked = sorted((m for m in models if card[m]["ranked"] and card[m]["score"] is not None),
                    key=lambda m: -card[m]["score"])
    order = ranked + [m for m in models if m not in ranked]
    total_weight = sum(m["weight"] for m in spec["metrics"])
    return templates.TemplateResponse(request, "results.html", {
        "nav": "runs",
        "name": name,
        "run": next((r for r in list_runs() if r["name"] == name), None),
        "config": config,
        "result": result,
        "card": card,
        "models": order,
        "metric_specs": [m for m in spec["metrics"] if m["type"] in ("check", "judge", "human")],
        "basis_share": (next(iter(card.values()))["weight_measured"] / total_weight) if card and total_weight else 0,
        "heat_style": heat_style,
        "scatter": scatter(card, result["frontier"]),
        "patterns": saved_patterns(name),
        "first_case": rows[0]["case_id"],
        "judge": {"sample": judge_sample if judge_sample is not None else "6", "estimate": judge_est, "over": over,
                  "judged": sum(1 for r in rows if r.get("judge")), "finished": sum(1 for r in rows if r.get("output")),
                  "budget": load_config().get("budget_usd"), "judges": load_config().get("judges") or []},
    })


@app.get("/runs/{name}/cases/{case_id}", response_class=HTMLResponse)
def compare_page(request: Request, name: str, case_id: str, repeat: int = 1):
    rows = load_rows(name)
    task = rows[0]["task"]
    spec = metrics.load(task)
    cases = {c["id"]: c for c in dataset.load(task)}
    case_ids = list(dict.fromkeys(r["case_id"] for r in rows))
    if case_id not in case_ids:
        raise HTTPException(404, "No such case in this run")
    repeats = sorted({r.get("repeat", 1) for r in rows if r["case_id"] == case_id})
    if repeat not in repeats:
        return RedirectResponse(f"/runs/{name}/cases/{case_id}?repeat={repeats[0]}")
    here = [r for r in rows if r["case_id"] == case_id and r.get("repeat", 1) == repeat]
    i = case_ids.index(case_id)
    return templates.TemplateResponse(request, "compare.html", {
        "nav": "runs",
        "name": name,
        "case": cases.get(case_id, {"id": case_id, "inputs": {}, "tags": {}}),
        "case_ids": case_ids,
        "case_tags": {c: cases.get(c, {}).get("tags", {}) for c in case_ids},
        "position": i + 1,
        "prev": case_ids[i - 1] if i > 0 else None,
        "next": case_ids[i + 1] if i + 1 < len(case_ids) else None,
        "repeat": repeat,
        "repeats": repeats,
        "cards": [card_view(r, spec) for r in here],
    })


def task_names() -> list[str]:
    return sorted(p.parent.name for p in metrics.TASKS_DIR.glob("*/task.yaml"))


def model_options(config: dict) -> list[dict]:
    """Models from models.yaml, with whether this computer has the API key each one needs."""
    out = []
    for m in config.get("models", []):
        try:
            missing = litellm.validate_environment(m["id"]).get("missing_keys") or []
        except Exception:
            missing = []
        out.append({"id": m["id"], "label": m.get("label") or short(m["id"]), "missing": missing,
                    "priced": m["id"] in configured_prices(config) or _litellm_priced(m["id"])})
    return out


def _litellm_priced(model: str) -> bool:
    try:
        litellm.cost_per_token(model=model, prompt_tokens=1, completion_tokens=1)
        return True
    except Exception:
        return False


def new_run_context(task: str, form: dict | None = None) -> dict:
    config = load_config()
    if task not in task_names():
        raise HTTPException(404, "No such task")
    cases = dataset.load(task)
    t = tasks.load(task)
    defaults = {"models": [m["id"] for m in config.get("models", [])], "limit": "", "repeats": 1,
                "budget": config.get("budget_usd"), "mock": False}
    return {
        "nav": "new",
        "tasks": task_names(),
        "task": task,
        "steps": t.steps,
        "case_count": len(cases),
        "tag_counts": {k: dict(v) for k, v in dataset.summary(cases).items()},
        "models": model_options(config),
        "spec": metrics.load(task),
        "locked": metrics.verify(task),
        "form": defaults | (form or {}),
        "judges": config.get("judges") or [],
    }


@app.get("/new", response_class=HTMLResponse)
def new_run_page(request: Request, task: str | None = None):
    task = task or (task_names() or [None])[0]
    if task is None:
        return templates.TemplateResponse(request, "new.html", {"nav": "new", "tasks": []})
    return templates.TemplateResponse(request, "new.html", new_run_context(task))


@app.post("/new", response_class=HTMLResponse)
def new_run_submit(request: Request, task: str = Form(...), action: str = Form("estimate"),
                   models: list[str] = Form([]), limit: str = Form(""), repeats: int = Form(1),
                   budget: str = Form(""), mock: bool = Form(False)):
    form = {"models": models, "limit": limit, "repeats": repeats, "budget": budget, "mock": mock}
    ctx = new_run_context(task, form)
    known = {m["id"] for m in ctx["models"]}
    problems = []
    if not models:
        problems.append("Pick at least one model.")
    if set(models) - known:
        problems.append("Unknown model; models come from models.yaml.")
    n = int(limit) if limit.strip().isdigit() and int(limit) > 0 else None
    if not 1 <= repeats <= 10:
        problems.append("Repeats must be between 1 and 10.")
    try:
        cap = float(budget) if budget.strip() else None
    except ValueError:
        cap, problems = None, problems + ["Budget must be a number of dollars."]
    if action == "run" and not mock:
        missing = [m["label"] for m in ctx["models"] if m["id"] in models and m["missing"]]
        if missing:
            problems.append(f"No API key on this computer for {', '.join(missing)}. Add it to .env and restart "
                            "the app, untick those models, or tick practice run.")
    if problems:
        return templates.TemplateResponse(request, "new.html", ctx | {"problems": problems}, status_code=400)
    *_, est = jobs.plan_run(task, models, limit=n, repeats=repeats, mock=mock)
    ctx |= {"estimate": est, "cap": cap}
    if action != "run":
        return templates.TemplateResponse(request, "new.html", ctx)
    if cap is not None and est.cost_usd > cap:
        ctx["problems"] = [f"The estimate (${est.cost_usd:.2f}) is over the ${cap:.2f} budget. "
                           "Raise the budget, or use fewer cases, models or repeats."]
        return templates.TemplateResponse(request, "new.html", ctx, status_code=400)
    job = jobs.start_run(task, models, limit=n, repeats=repeats, mock=mock, budget=cap)
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


@app.post("/runs/{name}/judge", response_class=HTMLResponse)
def judge_submit(name: str, sample: str = Form(""), budget: str = Form(""), mock: bool = Form(False)):
    load_rows(name)  # 404 for an unknown run
    n = int(sample) if sample.strip().isdigit() and int(sample) > 0 else None
    try:
        cap = float(budget) if budget.strip() else load_config().get("budget_usd")
    except ValueError:
        raise HTTPException(400, "Budget must be a number of dollars.")
    est = jobs.estimate_judging(name, sample=n, mock=mock)
    if cap is not None and est.cost_usd > cap:
        return RedirectResponse(f"/runs/{name}?judge_sample={sample}&over=1#judge", status_code=303)
    job = jobs.start_judging(name, sample=n, mock=mock, budget=cap)
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str):
    job = jobs.JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job. Jobs are forgotten when the app restarts; the run file is in Saved runs.")
    return templates.TemplateResponse(request, "job.html", {"nav": "runs", "job": job})


# Review queue: a person scores judged cards blind, against the judge's own rubrics.

def reviewable_runs() -> list[dict]:
    out = []
    for r in list_runs():
        if not r["judged"]:
            continue
        reviews = review.latest(review.load(run_path(r["name"])))
        r["reviewed"] = sum(1 for v in reviews.values() if not v.get("skipped"))
        r["skipped"] = len(reviews) - r["reviewed"]
        out.append(r)
    return out


def review_metrics(spec: dict) -> list[dict]:
    """The metrics a person scores 1-5: every judge metric (to check the judge) and every human one."""
    return [m for m in spec["metrics"] if m["type"] in ("judge", "human")]


@app.get("/review", response_class=HTMLResponse)
def review_index(request: Request):
    return templates.TemplateResponse(request, "review_runs.html", {"nav": "review", "runs": reviewable_runs()})


def review_context(name: str, reviewer: str = "") -> dict:
    rows, spec, cases, result = analyze_run(name)
    path = run_path(name)
    todo = review.queue(rows, review.load(path))
    judged = sum(1 for r in rows if r.get("output") is not None and r.get("judge"))
    ctx = {"nav": "review", "name": name, "reviewer": reviewer, "todo": len(todo), "judged": judged,
           "summary": result["review"], "metrics": review_metrics(spec), "problems": [], "picked": {}, "note": ""}
    if todo:
        row = todo[0]
        ctx |= {"pos": rows.index(row), "html": _md.render(row["output"]),
                "case": cases.get(row["case_id"], {"id": row["case_id"], "inputs": {}, "tags": {}})}
    return ctx


@app.get("/review/{name}", response_class=HTMLResponse)
def review_page(request: Request, name: str):
    return templates.TemplateResponse(request, "review.html",
                                      review_context(name, request.cookies.get("reviewer", "")))


@app.post("/review/{name}", response_class=HTMLResponse)
async def review_submit(request: Request, name: str):
    form = await request.form()
    rows = load_rows(name)
    try:
        row = rows[int(form.get("pos", ""))]
    except (ValueError, IndexError):
        raise HTTPException(400, "That card isn't in this run.")
    spec = metrics.load(row["task"])
    reviewer = str(form.get("reviewer", "")).strip()[:80]
    note = str(form.get("note", "")).strip()[:2000]
    skipped = form.get("action") == "skip"
    scores, missing = {}, []
    for m in review_metrics(spec):
        v = str(form.get(f"score_{m['name']}", ""))
        if v in {"1", "2", "3", "4", "5"}:
            scores[m["name"]] = int(v)
        else:
            missing.append(m["name"].replace("_", " "))
    if missing and not skipped:
        ctx = review_context(name, reviewer)
        if ctx.get("pos") != rows.index(row):  # someone else reviewed it meanwhile; just move on
            return RedirectResponse(f"/review/{name}", status_code=303)
        ctx |= {"problems": [f"Give a score for {', '.join(missing)}, or skip this card."],
                "picked": scores, "note": note}
        return templates.TemplateResponse(request, "review.html", ctx, status_code=400)
    review.save(run_path(name), row, {} if skipped else scores, note=note, reviewer=reviewer, skipped=skipped)
    response = RedirectResponse(f"/review/{name}", status_code=303)
    if reviewer:
        response.set_cookie("reviewer", reviewer, max_age=60 * 60 * 24 * 365, samesite="lax")
    return response
