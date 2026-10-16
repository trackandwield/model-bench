"""Turn a judged run into conclusions: which model, where it fails, how reliably, at what cost.

Everything here reads an existing run file; nothing calls a model except the
optional failure-pattern summary.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict

from bench.judge import passes

# A model that finished fewer of its cases than this isn't ranked: too little data.
MIN_COMPLETION = 0.5


def fill_prices(rows: list[dict], prices: dict[str, dict]) -> int:
    """Cost outputs that were run before their model had a price, from the tokens each
    step used. Returns how many outputs got a cost."""
    filled = 0
    for r in rows:
        p = prices.get(r["model"])
        if not p or r.get("cost_usd") is not None or r.get("output") is None:
            continue
        r["cost_usd"] = sum(s["input_tokens"] * p["input"] + s["output_tokens"] * p["output"]
                            for s in r.get("steps", [])) / 1_000_000
        filled += 1
    return filled


def metric_value(row: dict, metric: dict) -> float | None:
    """One output's result on one metric, from 0 (worst) to 1 (best); None if not measured."""
    if metric["type"] == "check":
        c = (row.get("checks") or {}).get(metric["name"])
        return None if c is None else float(c["passed"])
    if metric["type"] == "judge":
        j = (row.get("judge") or {}).get(metric["name"])
        return None if not j or j.get("score") is None else (j["score"] - 1) / 4
    return None  # human metrics come from the review screen; operational ones aren't quality


def metric_passed(row: dict, metric: dict) -> bool | None:
    if metric["type"] == "check":
        c = (row.get("checks") or {}).get(metric["name"])
        return None if c is None else bool(c["passed"])
    if metric["type"] == "judge":
        j = (row.get("judge") or {}).get(metric["name"])
        return None if not j else passes(j.get("score"), metric["pass"])
    return None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def scorecard(rows: list[dict], spec: dict, min_n: int = 5) -> dict:
    """Per model: each metric's mean and pass rate, and a weighted score.

    Models are only comparable on the same metrics, so the weighted score uses the
    metrics every ranked model has at least `min_n` measured outputs for (its
    "basis"), with the locked weights re-normalized over them. A metric judged on
    only a small sample is still shown, just not used to rank.
    """
    scored = [m for m in spec["metrics"] if m["type"] in ("check", "judge")]
    out = {}
    for model in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == model]
        done = [r for r in mine if r.get("output") is not None]
        per_metric = {}
        for m in scored:
            measured = [v for v in (metric_value(r, m) for r in done) if v is not None]
            passed = [p for p in (metric_passed(r, m) for r in done) if p is not None]
            per_metric[m["name"]] = {
                "mean": _mean(measured),
                "n": len(measured),
                "passed": sum(passed),
                "pass_rate": _mean([float(p) for p in passed]),
            }
        # A cached output was paid for in an earlier run and logs $0 and 0 s here, so cost
        # and speed come from outputs this run actually generated.
        fresh = [r for r in done if not any(s.get("cached") for s in r.get("steps", []))]
        costs = [r.get("cost_usd") for r in fresh]
        out[model] = {
            "completed": len(done),
            "total": len(mine),
            "ranked": bool(mine) and len(done) / len(mine) >= MIN_COMPLETION,
            "metrics": per_metric,
            "cost_per_output": None if not costs or None in costs else sum(costs) / len(costs),
            "latency_s": _mean([r.get("latency_s") for r in fresh]),
        }
    ranked = [v for v in out.values() if v["ranked"]]
    basis = [m for m in scored if ranked and all(v["metrics"][m["name"]]["n"] >= min_n for v in ranked)]
    weight = sum(m["weight"] for m in basis)
    for v in out.values():
        have = [m for m in basis if v["metrics"][m["name"]]["mean"] is not None]
        v["score"] = (sum(v["metrics"][m["name"]]["mean"] * m["weight"] for m in have) / weight
                      if weight and len(have) == len(basis) else None)
        v["basis"] = [m["name"] for m in basis]
        v["weight_measured"] = weight
    return out


def output_score(row: dict, metrics: list[dict]) -> float | None:
    vals = [(metric_value(row, m), m["weight"]) for m in metrics]
    vals = [(v, w) for v, w in vals if v is not None]
    total = sum(w for _, w in vals)
    return sum(v * w for v, w in vals) / total if total else None


def where(rows: list[dict], spec: dict, cases: dict[str, dict]) -> dict:
    """Mean weighted score per model per case tag (e.g. difficulty=tricky)."""
    scored = [m for m in spec["metrics"] if m["type"] in ("check", "judge")]
    cells: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("output") is None:
            continue
        s = output_score(r, scored)
        if s is None:
            continue
        for tag, value in (cases[r["case_id"]].get("tags") or {}).items():
            if value is not None:
                cells[(f"{tag}={value}", r["model"])].append(s)
    table: dict[str, dict] = defaultdict(dict)
    for (tag, model), vals in sorted(cells.items()):
        table[tag][model] = {"score": sum(vals) / len(vals), "n": len(vals)}
    return dict(table)


def consistency(rows: list[dict], spec: dict) -> dict:
    """How much a model's results change between repeats of the same case."""
    scored = [m for m in spec["metrics"] if m["type"] in ("check", "judge")]
    by_case: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("output") is not None:
            by_case[(r["model"], r["case_id"])].append(r)
    out = {}
    for model in dict.fromkeys(r["model"] for r in rows):
        groups = [g for (m, _), g in by_case.items() if m == model and len(g) > 1]
        if not groups:
            out[model] = None
            continue
        spreads, flips, compared = [], 0, 0
        for g in groups:
            scores = [s for s in (output_score(r, scored) for r in g) if s is not None]
            if len(scores) > 1:
                spreads.append(statistics.pstdev(scores))
            for m in scored:
                results = {metric_passed(r, m) for r in g} - {None}
                if results:
                    compared += 1
                    flips += len(results) > 1  # passed on one repeat, failed on another
        out[model] = {
            "cases": len(groups),
            "score_spread": _mean(spreads),
            "flip_rate": flips / compared if compared else None,
        }
    return out


def frontier(card: dict) -> list[str]:
    """Ranked models that no other model beats on both quality and cost per card.
    A model with unknown cost can't be compared on cost, so it's never ruled out."""
    ranked = {m: v for m, v in card.items() if v["ranked"] and v["score"] is not None}

    def beats(o, v):
        if o["cost_per_output"] is None or v["cost_per_output"] is None:
            return False
        return (o["score"] >= v["score"] and o["cost_per_output"] <= v["cost_per_output"]
                and (o["score"] > v["score"] or o["cost_per_output"] < v["cost_per_output"]))

    keep = [m for m, v in ranked.items() if not any(beats(o, v) for n, o in ranked.items() if n != m)]
    return sorted(keep, key=lambda m: -ranked[m]["score"])


def failures(rows: list[dict], spec: dict, limit: int = 3) -> dict:
    """Per model: how often each metric failed, with example cases and evidence."""
    scored = [m for m in spec["metrics"] if m["type"] in ("check", "judge")]
    out = {}
    for model in dict.fromkeys(r["model"] for r in rows):
        counts = {}
        for m in scored:
            failed = [r for r in rows if r["model"] == model and metric_passed(r, m) is False]
            if not failed:
                continue
            examples = []
            for r in failed[:limit]:
                if m["type"] == "check":
                    detail = r["checks"][m["name"]].get("detail")
                    examples.append({"case": r["case_id"], "evidence": ", ".join(map(str, detail or []))})
                else:
                    examples.append({"case": r["case_id"], "evidence": r["judge"][m["name"]].get("reasoning", "")})
            measured = sum(1 for r in rows if r["model"] == model and metric_passed(r, m) is not None)
            counts[m["name"]] = {"failed": len(failed), "measured": measured, "examples": examples}
        out[model] = dict(sorted(counts.items(), key=lambda kv: -kv[1]["failed"]))
    return out


def recommend(card: dict, spec: dict, good_enough: float = 0.05) -> list[str]:
    """Plain-language conclusions, each with the caveat that limits it."""
    lines = []
    ranked = {m: v for m, v in card.items() if v["ranked"] and v["score"] is not None}
    skipped = [m for m, v in card.items() if not v["ranked"]]
    if not ranked:
        if len(skipped) < len(card):
            return ["Too few outputs to rank: every model needs at least 5 measured on the same metrics."]
        return ["No model finished enough cases to rank."]
    best = max(ranked, key=lambda m: ranked[m]["score"])
    tied = [m for m, v in ranked.items() if ranked[best]["score"] - v["score"] < 0.01]
    if len(tied) > 1:
        lines.append(f"No clear winner on quality: {', '.join(tied)} score within 0.01 of each other "
                     f"({ranked[best]['score']:.2f}).")
    else:
        lines.append(f"Best quality: {best} (weighted score {ranked[best]['score']:.2f}).")
    priced = {m: v for m, v in ranked.items() if v["cost_per_output"] is not None}
    near = [m for m, v in priced.items() if v["score"] >= ranked[best]["score"] - good_enough]
    if near:
        cheapest = min(near, key=lambda m: priced[m]["cost_per_output"])
        if cheapest != best and len(tied) == 1:
            per_card = f"${priced[cheapest]['cost_per_output']:.4f} per card"
            if best in priced:
                lines.append(f"Good enough for less: {cheapest} is within {good_enough:.2f} of the best at {per_card}.")
            else:
                lines.append(f"Cheapest close to the best: {cheapest}, within {good_enough:.2f} at {per_card}. "
                             f"{best} has no price set, so it can't be compared on cost; add its price_per_mtok in models.yaml.")
    for m, v in ranked.items():
        weak = [f"{n} ({s['passed']}/{s['n']})" for n, s in v["metrics"].items()
                if s["n"] >= 5 and s["pass_rate"] < 0.7]
        if weak:
            lines.append(f"{m} passes under 70% of the time on: {', '.join(weak)}.")
    total = sum(m["weight"] for m in spec["metrics"])
    any_v = next(iter(ranked.values()))
    left_out = [m["name"] for m in spec["metrics"] if m["type"] in ("check", "judge") and m["name"] not in any_v["basis"]]
    lines.append(f"Caveat: the score is based on {', '.join(any_v['basis'])}: {any_v['weight_measured'] / total:.0%} "
                 "of the locked weight. Human-reviewed metrics aren't in it yet"
                 + (f", and {', '.join(left_out)} had too few judged outputs for every model to rank on."
                    if left_out else "."))
    if skipped:
        lines.append(f"Not ranked (finished under {MIN_COMPLETION:.0%} of cases): {', '.join(skipped)}.")
    return lines


def analyze(rows: list[dict], spec: dict, cases: dict[str, dict]) -> dict:
    card = scorecard(rows, spec)
    return {
        "scorecard": card,
        "where": where(rows, spec, cases),
        "consistency": consistency(rows, spec),
        "frontier": frontier(card),
        "failures": failures(rows, spec),
        "recommendation": recommend(card, spec),
    }


PATTERNS_PROMPT = """Below are reasons a reviewer gave for low scores on sales battlecards written by one AI model.
Group them into the 3 to 5 most common recurring problems. For each, give a short name, how many of the
reasons show it, and up to two case ids as examples.

{items}

Reply with only JSON: {{"patterns": [{{"name": "...", "count": 0, "examples": ["bc-001"]}}]}}"""


def pattern_prompt(model: str, rows: list[dict], spec: dict) -> str | None:
    """The judge's reasoning behind this model's failed judgments, ready to cluster."""
    judged = [m for m in spec["metrics"] if m["type"] == "judge"]
    items = [f"- {r['case_id']} ({m['name']}): {r['judge'][m['name']]['reasoning']}"
             for r in rows if r["model"] == model for m in judged if metric_passed(r, m) is False]
    return PATTERNS_PROMPT.format(items="\n".join(items)) if len(items) >= 2 else None


def parse_patterns(text: str) -> list[dict]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0:
        return []
    try:
        return json.loads(text[start:end + 1]).get("patterns", [])
    except json.JSONDecodeError:
        return []


def report_markdown(result: dict, run_name: str) -> str:
    """A readable report of the analysis, for the repo or the write-up."""
    card = result["scorecard"]
    models = list(card)
    short = {m: m.split("/")[-1] for m in models}
    lines = [f"# Results: {run_name}", "", "## Recommendation", ""]
    lines += [f"- {line}" for line in result["recommendation"]]
    lines += ["", "## Scorecard", "", "| Model | Weighted score | Finished | Cost per card | Avg latency |",
              "|---|---|---|---|---|"]
    for m in models:
        v = card[m]
        score = "not ranked" if not v["ranked"] else "-" if v["score"] is None else f"{v['score']:.2f}"
        cost = "unknown" if v["cost_per_output"] is None else f"${v['cost_per_output']:.4f}"
        lat = "-" if v["latency_s"] is None else f"{v['latency_s']:.1f} s"
        lines.append(f"| {short[m]} | {score} | {v['completed']}/{v['total']} | {cost} | {lat} |")
    metric_names = list(next(iter(card.values()))["metrics"]) if card else []
    lines += ["", "## Pass rate by metric", "", "| Metric | " + " | ".join(short[m] for m in models) + " |",
              "|---|" + "---|" * len(models)]
    for n in metric_names:
        cells = []
        for m in models:
            s = card[m]["metrics"][n]
            cells.append("-" if s["pass_rate"] is None else f"{s['pass_rate']:.0%} (n={s['n']})")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "## Where each model does well or badly", "",
              "| Case type | " + " | ".join(short[m] for m in models) + " |", "|---|" + "---|" * len(models)]
    for tag, by_model in result["where"].items():
        cells = [f"{by_model[m]['score']:.2f}" if m in by_model else "-" for m in models]
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    lines += ["", "## Consistency across repeats", "", "| Model | Cases repeated | Score spread | Pass/fail flips |",
              "|---|---|---|---|"]
    for m, c in result["consistency"].items():
        if c:
            spread = "-" if c["score_spread"] is None else f"{c['score_spread']:.3f}"
            flips = "-" if c["flip_rate"] is None else f"{c['flip_rate']:.0%}"
            lines.append(f"| {short[m]} | {c['cases']} | {spread} | {flips} |")
    lines += ["", "## Most common failures", ""]
    for m, fails in result["failures"].items():
        if not fails:
            continue
        lines.append(f"**{short[m]}**")
        for name, f in list(fails.items())[:3]:
            ex = f["examples"][0]
            lines.append(f"- {name}: failed {f['failed']} of {f['measured']}. e.g. {ex['case']}: {ex['evidence'][:200]}")
        for p in result.get("patterns", {}).get(m, []):
            lines.append(f"- Pattern: {p.get('name')} ({p.get('count')}), e.g. {', '.join(p.get('examples', []))}")
        lines.append("")
    if result["frontier"]:
        lines += ["## Cost vs quality", "",
                  f"No other model is both cheaper and better than: {', '.join(short[m] for m in result['frontier'])}.", ""]
    return "\n".join(lines) + "\n"
