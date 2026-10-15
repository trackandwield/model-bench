"""One way to call any model, with token counts, cost and latency for every call."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import litellm
import yaml

litellm.suppress_debug_info = True
# Some models reject settings others accept (GPT-5 reasoning models only allow
# temperature=1). Drop what a model doesn't support instead of failing the call;
# see docs/decision-log.md.
litellm.drop_params = True

CONFIG_PATH = Path(__file__).resolve().parent.parent / "models.yaml"

# Provider errors worth retrying (rate limits, timeouts, overloaded servers).
NUM_RETRIES = 3


@dataclass
class CallResult:
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: float
    finish_reason: str | None = None
    error: str | None = None
    cached: bool = False

    @property
    def refused(self) -> bool:
        return self.finish_reason == "content_filter"

    def to_dict(self) -> dict:
        return asdict(self) | {"refused": self.refused}


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def provider(model: str) -> str:
    return model.split("/", 1)[0]


def _messages(prompt: str, system: str | None) -> list[dict]:
    messages = [{"role": "system", "content": system}] if system else []
    return messages + [{"role": "user", "content": prompt}]


def _result(model: str, response, latency: float) -> CallResult:
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        # LiteLLM has no price for this model; record unknown rather than guess.
        cost = None
    choice = response.choices[0]
    return CallResult(
        model=model,
        text=choice.message.content or "",
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        cost_usd=cost,
        latency_s=latency,
        finish_reason=choice.finish_reason,
    )


def call_model(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    mock_response: str | None = None,
) -> CallResult:
    """Call one model and return its answer with usage and cost.

    `mock_response` skips the network and returns that text, so the pipeline
    can be tested without API keys or spend.
    """
    start = time.perf_counter()
    response = litellm.completion(
        model=model,
        messages=_messages(prompt, system),
        temperature=temperature,
        max_tokens=max_tokens,
        mock_response=mock_response,
        num_retries=NUM_RETRIES,
    )
    return _result(model, response, time.perf_counter() - start)


async def acall_model(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    mock_response: str | None = None,
) -> CallResult:
    """Async version of call_model. Errors are returned in the result, not raised,
    so one failing provider doesn't stop a run."""
    start = time.perf_counter()
    try:
        response = await litellm.acompletion(
            model=model,
            messages=_messages(prompt, system),
            temperature=temperature,
            max_tokens=max_tokens,
            mock_response=mock_response,
            num_retries=NUM_RETRIES,
        )
    except Exception as e:
        return CallResult(model, "", 0, 0, None, time.perf_counter() - start,
                          error=f"{type(e).__name__}: {e}".split("\n")[0][:300])
    return _result(model, response, time.perf_counter() - start)


async def gather_limited(coros_by_provider: dict[str, list], per_provider: int = 4) -> list:
    """Run coroutines with at most `per_provider` in flight per provider."""
    async def run_group(coros):
        sem = asyncio.Semaphore(per_provider)

        async def one(c):
            async with sem:
                return await c

        return await asyncio.gather(*(one(c) for c in coros))

    groups = await asyncio.gather(*(run_group(c) for c in coros_by_provider.values()))
    return [r for g in groups for r in g]
