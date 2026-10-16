from bench import analyze, metrics

SPEC = {"metrics": [
    {"name": "facts", "type": "judge", "weight": 0.5, "pass": ">= 4"},
    {"name": "prices", "type": "check", "weight": 0.3, "pass": "true"},
    {"name": "length", "type": "check", "weight": 0.2, "pass": "true"},
]}
CASES = {f"c{i}": {"tags": {"difficulty": "tricky" if i < 2 else "normal"}} for i in range(6)}


def row(model, case, repeat=1, prices=True, length=True, facts=None, cost=0.01):
    r = {"model": model, "case_id": case, "repeat": repeat, "output": "card", "cost_usd": cost, "latency_s": 1.0,
         "checks": {"prices": {"passed": prices, "detail": [] if prices else ["$49"]},
                    "length": {"passed": length, "detail": []}}}
    if facts is not None:
        r["judge"] = {"facts": {"score": facts, "reasoning": f"why {facts}"}}
    return r


def test_models_are_ranked_only_on_metrics_they_all_have():
    rows = [row("a", f"c{i}", facts=5) for i in range(6)] + [row("b", f"c{i}", prices=i > 2) for i in range(6)]
    card = analyze.scorecard(rows, SPEC)
    assert card["a"]["basis"] == ["prices", "length"]  # b was never judged, so facts can't rank
    assert card["a"]["score"] == 1.0
    assert card["b"]["score"] == (0.3 * 0.5 + 0.2) / 0.5


def test_judge_metric_counts_once_every_model_has_enough_judged():
    rows = [row(m, f"c{i}", facts=5 if m == "a" else 1) for m in "ab" for i in range(6)]
    card = analyze.scorecard(rows, SPEC)
    assert "facts" in card["a"]["basis"]
    assert card["a"]["score"] > card["b"]["score"]
    assert card["b"]["metrics"]["facts"]["pass_rate"] == 0


def test_models_that_mostly_failed_are_not_ranked():
    rows = [row("a", f"c{i}") for i in range(6)]
    rows += [row("g", "c0")] + [dict(row("g", f"c{i}"), output=None) for i in range(1, 6)]
    card = analyze.scorecard(rows, SPEC)
    assert not card["g"]["ranked"]
    assert any("Not ranked" in line and "g" in line for line in analyze.recommend(card, SPEC))


def test_where_splits_by_case_tag():
    rows = [row("a", f"c{i}", prices=i >= 2) for i in range(6)]
    table = analyze.where(rows, SPEC, CASES)
    assert table["difficulty=tricky"]["a"]["score"] < table["difficulty=normal"]["a"]["score"]


def test_consistency_counts_pass_fail_flips_between_repeats():
    rows = [row("a", "c0", 1, prices=True), row("a", "c0", 2, prices=False), row("a", "c0", 3, prices=True)]
    c = analyze.consistency(rows, SPEC)["a"]
    assert c["cases"] == 1 and c["flip_rate"] == 0.5  # prices flipped, length didn't
    assert c["score_spread"] > 0


def test_frontier_drops_models_that_are_worse_and_dearer():
    card = {
        "cheap": {"ranked": True, "score": 0.8, "cost_per_output": 0.001},
        "best": {"ranked": True, "score": 0.9, "cost_per_output": 0.01},
        "worse": {"ranked": True, "score": 0.7, "cost_per_output": 0.02},
        "unpriced": {"ranked": True, "score": 0.6, "cost_per_output": None},
    }
    assert analyze.frontier(card) == ["best", "cheap", "unpriced"]


def test_recommendation_names_a_cheaper_model_that_is_good_enough():
    rows = [row("big", f"c{i}", cost=0.05) for i in range(6)]
    rows += [row("small", f"c{i}", cost=0.001, length=i > 0) for i in range(6)]
    card = analyze.scorecard(rows, SPEC)
    lines = analyze.recommend(card, SPEC, good_enough=0.1)
    assert lines[0].startswith("Best quality: big")
    assert any("Good enough for less: small" in line for line in lines)


def test_report_renders_for_the_battlecard_metrics():
    spec = metrics.load("battlecard")
    rows = [row("a", "bc-001")]
    rows[0]["checks"] = {"pricing_matches_source": {"passed": True, "detail": []}}
    from bench import dataset
    cases = {c["id"]: c for c in dataset.load("battlecard")}
    md = analyze.report_markdown(analyze.analyze(rows, spec, cases), "test")
    assert "## Scorecard" in md and "## Recommendation" in md
