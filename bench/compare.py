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


async def cached_call(
    model: str,
    prompt: str,
    *,
    system: str | None,
    settings: dict,
    store: Store,
    run_id: str,
    mock_text: str | None = None,
    use_cache: bool = True,
    prices: dict[str, dict] | None = None,
    input_id: str | None = None,
) -> CallResult:
    """One model call that checks the cache first, fills in configured prices, and logs the result.

    `mock_text` skips the network. Mocked answers get their own cache entries so
    they're never served to a real run.
    """
    key_settings = settings | {"mock": True} if mock_text is not None else settings
    key = cache_key(model, prompt, system, key_settings)
    result = store.cached(key) if use_cache else None
    if result is None:
        result = await acall_model(model, prompt, system=system, mock_response=mock_text, **settings)
        price = (prices or {}).get(model)
        if result.cost_usd is None and not result.error and price:
            result.cost_usd = (result.input_tokens * price["input"] + result.output_tokens * price["output"]) / 1_000_000
    store.log(run_id, key, key_settings, result, input_id=input_id)
    return result


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
    by_provider: dict[str, list] = defaultdict(list)
    order: dict[str, list[str]] = defaultdict(list)
    for m in models:
        by_provider[provider(m)].append(cached_call(
            m, prompt, system=system, settings=settings, store=store, run_id=run_id,
            mock_text=f"Mock answer from {m}." if mock else None, use_cache=use_cache, prices=prices,
        ))
        order[provider(m)].append(m)
    results = dict(zip([m for p in by_provider for m in order[p]], await gather_limited(by_provider)))
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
