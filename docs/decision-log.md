# Decision log

One entry per design choice: date, decision, options considered, why. Mirror into Notion.

## 2026-10-15: Python with uv
- **Options:** Poetry, plain pip + venv, uv.
- **Why:** uv is fast, writes a lockfile, and is becoming the default in AI tooling teams.

## 2026-10-15: LiteLLM as the model layer
- **Options:** LiteLLM, OpenRouter for everything, one hand-written adapter per provider.
- **Why:** one function for every provider, and it ships a price table, so cost per call comes for free. OpenRouter stays available as a LiteLLM provider for open-weights models.

## 2026-10-15: models in a config file
- **Why:** model ids change every few months. Keeping them in `models.yaml` means swapping a model is a one-line change and every run can record exactly which ids it used.

## 2026-10-15: a mock mode from day one
- **Why:** `--mock` runs the whole pipeline with canned text and no API key, so tests are free and the wiring can be checked before spending anything.

## 2026-10-15: metrics are hashed and locked before runs
- **Options:** trust the git history; lock with a hash.
- **Why:** a hash stored with every run is a one-line proof the metrics weren't changed after seeing results ("pre-registration").

## 2026-10-15: start with the battlecard task
- **Why:** it has the clearest failure mode to measure (invented facts and prices) and reuses company profiles from project 7.

## 2026-10-15: fictional companies in the battlecard test set
- **Options:** real observability vendors (from project 7), fictional ones.
- **Why:** the "no invented facts" metric checks every claim against the profile, so the profile has to be the whole truth. With real companies, a model could state a true fact from its training data and get marked wrong.
