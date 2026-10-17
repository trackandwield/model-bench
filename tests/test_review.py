from bench import analyze, review

SPEC = {"metrics": [
    {"name": "facts", "type": "judge", "weight": 0.5, "pass": ">= 4"},
    {"name": "rep_ready", "type": "human", "weight": 0.5, "pass": ">= 4"},
]}


def row(case, judge, model="m"):
    return {"case_id": case, "model": model, "repeat": 1, "output": "x", "judge": {"facts": {"score": judge}}}


def test_queue_skips_reviewed_and_unjudged_cards():
    rows = [row("a", 4), row("b", 4), {"case_id": "c", "model": "m", "repeat": 1, "output": "x"}]
    reviews = [{"case_id": "a", "model": "m", "repeat": 1, "scores": {}, "skipped": True}]
    assert [r["case_id"] for r in review.queue(rows, reviews)] == ["b"]


def test_later_review_replaces_earlier_one():
    rows = [row("a", 4)]
    review.merge(rows, [{"case_id": "a", "model": "m", "repeat": 1, "scores": {"facts": 1, "rep_ready": 2}},
                        {"case_id": "a", "model": "m", "repeat": 1, "scores": {"facts": 4, "rep_ready": 5}}])
    assert rows[0]["human"]["rep_ready"] == {"score": 5}
    assert analyze.metric_passed(rows[0], SPEC["metrics"][1]) is True
    assert analyze.metric_value(rows[0], SPEC["metrics"][1]) == 1.0


def test_agreement_and_trust_label():
    rows = [row(c, j) for c, j in zip("abcd", (5, 4, 2, 4))]
    reviews = [{"case_id": c, "model": "m", "repeat": 1, "scores": {"facts": h, "rep_ready": 3}}
               for c, h in zip("abcd", (5, 2, 2, 3))]
    review.merge(rows, reviews)
    s = review.summary(rows, reviews, SPEC)
    a = s["agreement"]["facts"]
    assert (a["n"], a["exact"], a["within_1"], a["judge_minus_human"]) == (4, 0.5, 0.75, 0.75)
    assert s["trust"].startswith("Judge not checked against a person yet (4 of 20")
    assert review.trust_label(s["agreement"], 20) == \
        "Judge agrees with the reviewer within 1 point on 75% of 4 scores (20 cards reviewed)"
