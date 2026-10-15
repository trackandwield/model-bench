import shutil

import pytest

from bench import metrics


def test_battlecard_draft_is_valid():
    assert metrics.validate(metrics.load("battlecard")) == []


def test_validate_catches_bad_weights_and_types():
    spec = {"metrics": [
        {"name": "a", "description": "x", "type": "vibes", "weight": 0.5, "pass": ">= 3"},
        {"name": "a", "description": "x", "type": "judge", "weight": 0.2, "pass": ">= 3"},
    ]}
    problems = " ".join(metrics.validate(spec))
    assert "type must be" in problems
    assert "duplicate" in problems
    assert "weights add up to 0.70" in problems


@pytest.fixture
def tmp_tasks(tmp_path, monkeypatch):
    shutil.copytree(metrics.TASKS_DIR, tmp_path / "tasks")
    monkeypatch.setattr(metrics, "TASKS_DIR", tmp_path / "tasks")
    return tmp_path / "tasks"


def test_lock_then_edit_breaks_verify(tmp_tasks):
    metrics.lock("battlecard")
    assert metrics.verify("battlecard")
    path = tmp_tasks / "battlecard" / "metrics.yaml"
    path.write_text(path.read_text().replace("weight: 0.30", "weight: 0.31"))
    assert not metrics.verify("battlecard")
