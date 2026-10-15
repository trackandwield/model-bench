"""LLM-as-judge: score each output on each subjective metric, with reasoning and quoted evidence.

One judge call per output per metric, each with that metric's 1-5 rubric, so a
judge can't let one strong section carry a weak one. Controls for known judge
biases:

- Blind: the judge never sees which model wrote the output, and vendor or model
  names inside the output are masked.
- Cross-family: an output is judged by a model from a different provider, so a
  model never grades its own family (self-preference).
- Checked evidence: the judge must quote the output; quotes that aren't in it
  are flagged, because a judge can invent evidence too.
- Consistency: a sample is judged a second time to measure how stable scores are.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from bench.compare import cached_call, new_run_id
from bench.models import gather_limited, provider
from bench.store import Store

SYSTEM = (
    "You are a strict evaluator of sales enablement content. You score one criterion at a time "
    "against a rubric, using only the source material given. Reply with JSON only."
)

PROMPT = """Score this battlecard on one criterion.

<criterion name="{name}">
{description}
</criterion>

<rubric>
{rubric}
</rubric>

<source>
{source}
</source>
{flags}
<battlecard>
{output}
</battlecard>

Reply with only this JSON object:
{{"evidence": ["exact quotes from the battlecard that decide the score"], "reasoning": "two to four sentences", "score": <integer 1-5>}}
Quote the battlecard word for word. Write evidence and reasoning before deciding the score."""

# Vendor and model names masked before judging, so an output can't reveal who wrote it.
_NAMES = re.compile(
    r"\b(claude|anthropic|gpt[-\w.]*|chatgpt|openai|gemini|google|llama[-\w.]*|meta ai|mistral)\b", re.I
)


def blind(text: str) -> str:
    return _NAMES.sub("[model]", text)


def pick_judge(model: str, judges: list[str]) -> str | None:
    """First judge from a different provider than the model under test."""
    return next((j for j in judges if provider(j) != provider(model)), None)


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[*_`#>\"“”]", "", s).lower().split())


def parse(text: str) -> tuple[int, str, list[str]]:
    """Pull score, reasoning and evidence out of the judge's reply. Raises ValueError."""
    found = re.search(r"\{.*\}", text, re.S)
    if not found:
        raise ValueError("no JSON object in judge reply")
    data = json.loads(found.group(0))
    score = data.get("score")
    if isinstance(score, str) and score.strip().isdigit():
        score = int(score)
    if not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError(f"score must be an integer 1-5, got {score!r}")
    evidence = data.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    return score, str(data.get("reasoning", "")).strip(), [str(e) for e in evidence]


@dataclass
class Judgment:
    metric: str
    judge_model: str
    score: int | None = None
    reasoning: str = ""
    evidence: list[str] = field(default_factory=list)
    missing_quotes: list[str] = field(default_factory=list)  # evidence not found in the output
    cost_usd: float | None = None
    cached: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def passes(score: int | None, bar: str) -> bool | None:
    """Apply a pass bar like ">= 4" to a score."""
    if score is None:
        return None
    m = re.fullmatch(r"\s*(>=|<=|>|<|==)\s*(\d+(?:\.\d+)?)\s*", str(bar))
    if not m:
        return None
    op, n = m.group(1), float(m.group(2))
    return {">=": score >= n, "<=": score <= n, ">": score > n, "<": score < n, "==": score == n}[op]


def judge_metrics(metrics_spec: dict) -> list[dict]:
    return [m for m in metrics_spec.get("metrics", []) if m.get("type") == "judge"]


def build_prompt(metric: dict, inputs: dict, output: str, checks: dict | None = None) -> str:
    rubric = "\n".join(f"{k}: {' '.join(str(v).split())}" for k, v in sorted(metric["rubric"].items(), reverse=True))
    source = "\n".join(f"<{k}>\n{v}\n</{k}>" for k, v in inputs.items())
    flags = ""
    for name in metric.get("show_checks") or []:
        c = (checks or {}).get(name)
        if c and not c.get("passed") and c.get("detail"):
            flags += (f"\n<automatic_flag check=\"{name}\">\nNot found in the source: "
                      f"{', '.join(map(str, c['detail']))}. Decide for each whether it is invented.\n</automatic_flag>\n")
    return PROMPT.format(name=metric["name"], description=" ".join(metric["description"].split()),
                         rubric=rubric, source=source, flags=flags, output=blind(output))


def _mock_reply(output: str, metric: str) -> str:
    """Deterministic fake verdict for --mock: a stable score and a real quote."""
    score = 1 + sum(map(ord, output + metric)) % 5
    quote = blind(output).strip().splitlines()[0][:60] if output.strip() else ""
    return json.dumps({"evidence": [quote], "reasoning": "Mock judgment.", "score": score})


async def judge_one(row: dict, metric: dict, judge_model: str, *, inputs: dict, settings: dict, store: Store,
                    run_id: str, mock: bool = False, use_cache: bool = True, attempt: int = 1) -> Judgment:
    j = Judgment(metric["name"], judge_model)
    output = row["output"]
    prompt = build_prompt(metric, inputs, output, row.get("checks"))
    r = await cached_call(
        judge_model, prompt, system=SYSTEM, settings=settings, store=store, run_id=run_id,
        mock_text=_mock_reply(output, metric["name"]) if mock else None, use_cache=use_cache,
        input_id=f"{row['case_id']}:{metric['name']}:judge{attempt}",
        # A repeat needs its own cache entry, or it would just return the first answer.
        cache_tag=f"repeat{attempt}" if attempt > 1 else None,
    )
    j.cost_usd, j.cached = r.cost_usd, r.cached
    if r.error:
        j.error = r.error
        return j
    try:
        j.score, j.reasoning, j.evidence = parse(r.text)
    except (ValueError, json.JSONDecodeError) as e:
        j.error = f"Unreadable judge reply ({e}): {r.text[:200]}"
        return j
    texts = (_norm(output), _norm(blind(output)))
    j.missing_quotes = [q for q in j.evidence if _norm(q) and not any(_norm(q) in t for t in texts)]
    return j


async def judge_rows(rows: list[dict], metrics: list[dict], judges: list[str], cases: dict[str, dict], *,
                     settings: dict, store: Store, run_id: str, mock=False, use_cache=True,
                     repeat_sample: int = 0, seed: int = 0) -> tuple[list[tuple], list[tuple]]:
    """Judge every finished output on every judge metric.

    Returns (first, repeats): lists of (row index, Judgment). Repeats re-judge a
    random sample of the same (output, metric) pairs to measure consistency.
    """
    jobs = []
    for i, row in enumerate(rows):
        if row.get("output") is None:
            continue  # the chain failed; nothing to judge
        judge_model = pick_judge(row["model"], judges)
        for m in metrics:
            jobs.append((i, m, judge_model))
    sample = random.Random(seed).sample(jobs, min(repeat_sample, len(jobs)))

    async def run(job_list, attempt):
        by_provider: dict[str, list] = defaultdict(list)
        index: dict[str, list[int]] = defaultdict(list)
        for i, m, jm in job_list:
            if jm is None:
                continue
            by_provider[provider(jm)].append(judge_one(
                rows[i], m, jm, inputs=cases[rows[i]["case_id"]]["inputs"], settings=settings, store=store,
                run_id=run_id, mock=mock, use_cache=use_cache, attempt=attempt))
            index[provider(jm)].append(i)
        results = await gather_limited(by_provider)
        out = list(zip([i for p in by_provider for i in index[p]], results))
        # Outputs with no cross-family judge get an explicit error, not a silent gap.
        out += [(i, Judgment(m["name"], "", error="No judge from a different model family; add one to "
                             "`judges` in models.yaml")) for i, m, jm in job_list if jm is None]
        return out

    first = await run(jobs, 1)
    repeats = await run(sample, 2) if sample else []
    return first, repeats


def consistency(first: list[tuple], repeats: list[tuple]) -> dict | None:
    """How often a second judgment of the same output agrees with the first."""
    a = {(i, j.metric): j.score for i, j in first if j.score is not None}
    pairs = [(a[(i, j.metric)], j.score) for i, j in repeats if j.score is not None and (i, j.metric) in a]
    if not pairs:
        return None
    return {
        "pairs": len(pairs),
        "exact": sum(x == y for x, y in pairs) / len(pairs),
        "within_1": sum(abs(x - y) <= 1 for x, y in pairs) / len(pairs),
        "differences": [(x, y) for x, y in pairs if x != y],
    }


def run(rows: list[dict], metrics_spec: dict, judges: list[str], cases: dict[str, dict], *, settings: dict,
        store: Store | None = None, mock=False, use_cache=True, repeat_sample=0, seed=0):
    """Judge an earlier run's rows in place: each row gets row["judge"][metric] = judgment."""
    run_id = new_run_id()
    metrics = judge_metrics(metrics_spec)
    first, repeats = asyncio.run(judge_rows(
        rows, metrics, judges, cases, settings=settings, store=store or Store(), run_id=run_id, mock=mock,
        use_cache=use_cache, repeat_sample=repeat_sample, seed=seed))
    bars = {m["name"]: m["pass"] for m in metrics}
    for i, j in first:
        rows[i].setdefault("judge", {})[j.metric] = j.to_dict() | {"passed": passes(j.score, bars[j.metric])}
    for i, j in repeats:
        rows[i]["judge"][j.metric]["repeat_score"] = j.score
    return run_id, first, repeats
