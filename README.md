# Model Comparison Bench

Run the same marketing prompt, workflow or skill across several LLMs, score every output against success metrics locked before the run, and get a recommendation: which model to use for which task, why, and at what cost.

## The problem

Marketing teams pick an AI model by gut feel or by a public leaderboard that tests coding and math, not battlecards. Nobody on the team can say which model writes the best launch post for *their* product, where a cheaper model is good enough, or why the others fall short.

## How it works

1. **Lock the metrics first.** Each task has a `metrics.yaml` written with a working product marketer: what "good" means, how each criterion is scored (automatic check, LLM judge, human review or cost), its weight and its pass bar. `bench lock` records a hash before any model runs, so results can't be tuned after the fact.
2. **Run the matrix.** Every model × every test input × several repeats, with identical settings, logging tokens, cost and latency for every call.
3. **Score.** Rule-based checks first, then a blinded LLM judge from a different model family, calibrated against a human reviewer.
4. **Explain.** A weighted scorecard, where each model wins or loses by input type, the recurring failure patterns behind low scores, and the cost-quality trade-off.

## Status

| Milestone | State |
|---|---|
| 1. Foundations: setup, locked metrics, test set | Built; metrics awaiting PMM review |
| 2. Engine: model layer, tasks, checks, judge, runs | Model layer built |
| 3. Human review and analysis | Not started |
| 4. Web app and write-up | Not started |

## Run it

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env              # add at least one API key
uv run python -m bench hello --mock   # no key needed: checks the wiring
uv run python -m bench hello          # calls the first model in models.yaml
uv run python -m bench hello --model openai/gpt-5-mini
uv run python -m bench compare --prompt "Write a tagline for an observability product."
uv run python -m bench compare --mock --prompt "..."   # no keys, no cost
uv run python -m bench lock battlecard
uv run python -m bench verify battlecard
uv run pytest
```

Models live in `models.yaml` as [LiteLLM](https://docs.litellm.ai/) ids, so adding or swapping one is a config change. `compare` runs every model in parallel (a few calls at a time per provider, with retries), prints tokens, cost and latency per model, and saves the outputs to `runs/<run id>.jsonl`. Every call is also logged to `runs/bench.db`, which doubles as a cache: an identical call is served from it for $0.

## Repo layout

```
bench/          the engine (CLI, model layer, metrics)
tasks/<task>/   metrics.yaml, its lock, and cases.jsonl (tagged test inputs)
models.yaml     models under test and the settings held constant
docs/           decision log
tests/
```
