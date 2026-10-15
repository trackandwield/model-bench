"""Run one prompt on every model in models.yaml and save the results."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from bench.models import CallResult, acall_model, gather_limited, provider
from bench.store import Store, cache_key

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


async def compare(
    models: list[str],
    prompt: str,
    *,
    system: str | None,
    settings: dict,
    store: Store,
    run_id: str,
    mock: bool = False,
    use_cache: bool = True,
    prices: dict[str, dict] | None = None,
) -> list[CallResult]:
    """`prices` fills in cost for models LiteLLM has no price for: {model: {"input": $/Mtok, "output": $/Mtok}}."""
    prices = prices or {}
    # Mocked answers get their own cache entries so they're never served to a real run.
    key_settings = settings | {"mock": True} if mock else settings
    keys = {m: cache_key(m, prompt, system, key_settings) for m in models}
    results: dict[str, CallResult] = {}
    to_call: dict[str, list] = defaultdict(list)
    order: dict[str, list[str]] = defaultdict(list)

    for m in models:
        hit = store.cached(keys[m]) if use_cache else None
        if hit:
            results[m] = hit
        else:
            mock_text = f"Mock answer from {m}." if mock else None
            to_call[provider(m)].append(acall_model(m, prompt, system=system, mock_response=mock_text, **settings))
            order[provider(m)].append(m)

    fresh = await gather_limited(to_call)
    called = [m for p in to_call for m in order[p]]
    results.update(zip(called, fresh))

    for m in models:
        r = results[m]
        if r.cost_usd is None and not r.error and m in prices:
            p = prices[m]
            r.cost_usd = (r.input_tokens * p["input"] + r.output_tokens * p["output"]) / 1_000_000
        store.log(run_id, keys[m], key_settings, r)
    return [results[m] for m in models]


def write_results(run_id: str, prompt: str, settings: dict, results: list[CallResult]) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{run_id}.jsonl"
    with path.open("w") as f:
        for r in results:
            f.write(json.dumps({"run_id": run_id, "prompt": prompt, "settings": settings} | r.to_dict()) + "\n")
    return path


def run(models, prompt, *, system=None, settings, mock=False, use_cache=True, store=None, prices=None):
    run_id = new_run_id()
    store = store or Store()
    results = asyncio.run(compare(models, prompt, system=system, settings=settings, store=store,
                                  run_id=run_id, mock=mock, use_cache=use_cache, prices=prices))
    return run_id, results, write_results(run_id, prompt, settings, results)
