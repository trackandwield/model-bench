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

## 2026-10-15: drop settings a model doesn't support
- **Found:** GPT-5 models reject `temperature=0.2`; reasoning models only accept 1.
- **Options:** fail the call; give each model its own settings; let LiteLLM drop unsupported settings.
- **Why:** dropping keeps one shared config and keeps every model in the comparison. The trade-off is that "same settings for every model" can't hold exactly across reasoning and non-reasoning models, and the write-up has to say so.

## 2026-10-15: SQLite call log doubles as the cache
- **Why:** every call (model, settings, tokens, cost, latency, errors) lands in `runs/bench.db`, and an identical later call is served from it for $0. Errors are logged but never served from cache, so a rate-limited call is retried next time.

## 2026-10-15: Gemini 2.5 Flash → Gemini 3.8 Flash
- **Found:** the first real run failed because Google no longer serves `gemini-2.5-flash` to new API users.
- **Why it matters:** model ids expire within months, so every run records the exact ids it used, and the README results carry a test date.
- **Also found:** LiteLLM warns that Gemini 3+ plans to drop `temperature`, `top_p` and `top_k`, with sampling guidance moving into the system prompt. Another case where "same settings for every model" will stop holding.

## 2026-10-15: battlecard as a 2-step workflow
- **Options:** one prompt; extract-then-draft; research-outline-draft.
- **Why:** splitting "pull out the facts" from "write the card" shows *where* a model invents things: misreading the profile, or embellishing while writing. Two steps keeps cost down; a third can be added in `task.yaml` without code changes.

## 2026-10-15: Jinja2 with strict variables for prompts
- **Why:** a typo in a variable name raises an error instead of sending every model a prompt with a blank in it, which would quietly ruin a whole run.

## 2026-10-15: max_tokens 4096, and cut-off output counts as a failure
- **Found:** Gemini used 645 output tokens on a one-line tagline, mostly hidden reasoning. At the old 1024 limit a full battlecard would likely be cut off on reasoning models.
- **Why:** a truncated extract would silently feed half the facts into the draft step, so a `length` finish is reported as the step where the chain broke.

## 2026-10-15: automatic checks before any LLM judge
- **Why:** anything a rule can catch (an invented price, a missing section, a card over one page) is caught for free and with evidence, so the paid judge only scores what needs judgment.
- **Invented facts, rough version:** flag any dollar amount or number in the card that never appears in the profiles. It misses invented claims with no number in them; that's left to the judge.

## 2026-10-15: own Flesch-Kincaid instead of textstat
- **Found:** the current textstat release downloads an NLTK dictionary on first use, which fails behind proxies and adds a setup step on every machine.
- **Why:** the formula is three lines plus a syllable estimate. A grade that's off by half a point doesn't change any conclusion; a setup failure blocks the whole run.

## 2026-10-15: compare number values, not strings
- **Found:** the first tricky-case run flagged Claude for "10,000 contacts" when the profile said "10k contacts". A false positive.
- **Fix:** numbers and prices are compared by value (10k = 10,000, $1.2M = $1,200,000), with percentages kept distinct (40% is not 40).
- **Still flagged, on purpose:** numbers a model *calculated* (a $100/agent price turned into "$28,800 a year for 20 agents"). Arithmetic on source facts isn't invention, but it's an unsourced claim a rep would repeat, so it goes to the judge to decide.

## 2026-10-15: LLM judge, one call per metric, cross-family
- **Options:** one call scoring all criteria; one call per criterion; pairwise (A vs B) comparisons.
- **Why one call per criterion:** with a single call, a strong section lifts the scores of weak ones (a halo effect), and the reasoning gets thin. Three metrics means three calls per output, which is affordable at battlecard length.
- **Why scores, not pairwise:** scores give an absolute bar ("at least 4 on invented facts") that matches the locked metrics. Pairwise comparison, with sides shuffled to control position bias, belongs in the human blind review, which calibrates this judge.
- **Cross-family judges:** Claude Sonnet 5.5 judges every non-Anthropic output and GPT-5.6 judges Claude's, so no model scores its own family.
- **Evidence before score:** the judge must quote the output and reason before it scores. Quotes are checked against the output, because a judge can invent evidence the same way a writer invents facts.
- **Rubrics live in metrics.yaml**, so they're covered by the same lock as the weights and pass bars. Changing a rubric after seeing scores would be moving the goalposts.
