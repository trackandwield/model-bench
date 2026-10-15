"""Task definitions: a single prompt, a multi-step workflow, or either with reference files (a "skill").

tasks/<task>/task.yaml:

    task: battlecard
    system: "You are ..."            # optional, the same for every model and step
    references: [brand-voice.md]     # optional files, appended to the system prompt
    steps:                           # one step = a plain prompt; more = a workflow
      - name: extract
        prompt: "... {{ competitor }} ..."
      - name: draft
        prompt: "... {{ steps.extract }} ..."

Prompts are Jinja2 templates. They see the case's inputs by name, and each
earlier step's output as {{ steps.<name> }}. A missing variable is an error,
not a blank, so a typo can't silently send a broken prompt to every model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jinja2
import yaml

from bench.metrics import TASKS_DIR

_env = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True)


@dataclass
class Step:
    name: str
    prompt: str

    def render(self, inputs: dict, previous: dict[str, str]) -> str:
        return _env.from_string(self.prompt).render(**inputs, steps=previous)


@dataclass
class Task:
    name: str
    steps: list[Step]
    system: str | None = None
    references: dict[str, str] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "workflow" if len(self.steps) > 1 else "prompt"

    def system_prompt(self) -> str | None:
        parts = [self.system] if self.system else []
        parts += [f"<reference name=\"{n}\">\n{text}\n</reference>" for n, text in self.references.items()]
        return "\n\n".join(parts) or None


def validate(spec: dict) -> list[str]:
    problems = []
    steps = spec.get("steps") or []
    if not steps:
        problems.append("no steps defined")
    names = [s.get("name") for s in steps]
    if len(set(names)) != len(names):
        problems.append("step names must be unique")
    for i, s in enumerate(steps):
        if not s.get("name") or not s.get("prompt"):
            problems.append(f"step {i + 1}: needs a name and a prompt")
            continue
        # A step may only use steps that ran before it.
        later = set(names[i:])
        for other in later:
            if f"steps.{other}" in s["prompt"]:
                problems.append(f"step {s['name']}: uses steps.{other}, which hasn't run yet")
    return problems


def load(task: str) -> Task:
    folder = TASKS_DIR / task
    spec = yaml.safe_load((folder / "task.yaml").read_text())
    problems = validate(spec)
    if problems:
        raise ValueError(f"{task}/task.yaml: " + "; ".join(problems))
    refs = {name: (folder / name).read_text() for name in spec.get("references") or []}
    steps = [Step(s["name"], s["prompt"]) for s in spec["steps"]]
    return Task(spec.get("task", task), steps, spec.get("system"), refs)
