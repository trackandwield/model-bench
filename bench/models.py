"""One way to call any model, with token counts, cost and latency for every call."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import litellm
import yaml

litellm.suppress_debug_info = True

CONFIG_PATH = Path(__file__).resolve().parent.parent / "models.yaml"


@dataclass
class CallResult:
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: float


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text())


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
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    start = time.perf_counter()
    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        mock_response=mock_response,
    )
    latency = time.perf_counter() - start

    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        # LiteLLM has no price for this model; record unknown rather than guess.
        cost = None

    usage = response.usage
    return CallResult(
        model=model,
        text=response.choices[0].message.content or "",
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cost_usd=cost,
        latency_s=latency,
    )
