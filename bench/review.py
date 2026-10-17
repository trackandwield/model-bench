"""Blind human review: a person scores cards against the same rubrics as the judge.

Reviews are saved next to the run as runs/<run>.reviews.jsonl, one line per
card. They do two jobs: scores on `human` metrics (like rep_ready) count in the
analysis, and scores on `judge` metrics show how far the judge can be trusted.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

# Fewer reviews than this and the judge counts as unchecked.
TRUSTED_AFTER = 20


def path_for(run_path: Path) -> Path:
    """runs/<run>.jsonl -> runs/<run>.reviews.jsonl"""
    return run_path.with_name(run_path.stem + ".reviews.jsonl")


def key(row: dict) -> tuple[str, str, int]:
    return row["case_id"], row["model"], row.get("repeat", 1)


def load(run_path: Path) -> list[dict]:
    p = path_for(run_path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(run_path: Path, row: dict, scores: dict[str, int], note: str = "", reviewer: str = "", skipped=False) -> dict:
    record = {"case_id": row["case_id"], "model": row["model"], "repeat": row.get("repeat", 1),
              "scores": scores, "note": note, "reviewer": reviewer, "skipped": skipped,
              "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with path_for(run_path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def latest(reviews: list[dict]) -> dict[tuple, dict]:
    """The last review of each card; a later review of the same card replaces an earlier one."""
    return {(r["case_id"], r["model"], r.get("repeat", 1)): r for r in reviews}


def queue(rows: list[dict], reviews: list[dict], seed: int = 0) -> list[dict]:
    """Judged cards nobody has reviewed or skipped yet, in a fixed shuffled order so
    cards from one model don't come in a row."""
    done = latest(reviews)
    todo = [r for r in rows if r.get("output") is not None and r.get("judge") and key(r) not in done]
    random.Random(seed).shuffle(todo)
    return todo


def merge(rows: list[dict], reviews: list[dict]) -> None:
    """Put each card's human scores on its row as row["human"][metric] = {"score": n}."""
    by_key = latest(reviews)
    for r in rows:
        rev = by_key.get(key(r))
        if rev and not rev.get("skipped") and rev.get("scores"):
            r["human"] = {m: {"score": s} for m, s in rev["scores"].items()}


def agreement(rows: list[dict], metrics: list[dict]) -> dict:
    """Per judge metric: how often the person and the judge gave the same score, or
    within one point, and whether the judge tends to score higher or lower."""
    out = {}
    for m in metrics:
        pairs = [(r["human"][m["name"]]["score"], r["judge"][m["name"]]["score"]) for r in rows
                 if (r.get("human") or {}).get(m["name"]) and (r.get("judge") or {}).get(m["name"])
                 and r["judge"][m["name"]].get("score") is not None]
        if pairs:
            out[m["name"]] = {
                "n": len(pairs),
                "exact": sum(h == j for h, j in pairs) / len(pairs),
                "within_1": sum(abs(h - j) <= 1 for h, j in pairs) / len(pairs),
                "judge_minus_human": sum(j - h for h, j in pairs) / len(pairs),
            }
    return out


def trust_label(agree: dict, reviewed: int) -> str:
    if reviewed < TRUSTED_AFTER or not agree:
        return f"Judge not checked against a person yet ({reviewed} of {TRUSTED_AFTER} reviews)"
    n = sum(a["n"] for a in agree.values())
    within = sum(a["within_1"] * a["n"] for a in agree.values()) / n
    return f"Judge agrees with the reviewer within 1 point on {within:.0%} of {n} scores ({reviewed} cards reviewed)"


def summary(rows: list[dict], reviews: list[dict], spec: dict) -> dict:
    """What the analysis and report show about human review. Call after merge()."""
    by_key = latest(reviews)
    reviewed = sum(1 for r in by_key.values() if not r.get("skipped"))
    agree = agreement(rows, [m for m in spec["metrics"] if m["type"] == "judge"])
    return {"reviewed": reviewed, "skipped": len(by_key) - reviewed, "agreement": agree,
            "trust": trust_label(agree, reviewed), "trusted": reviewed >= TRUSTED_AFTER and bool(agree)}
