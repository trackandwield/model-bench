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

## 2026-10-15: run orchestration on top of the call cache
- **Resumable without a job queue:** every finished call is already committed to SQLite, so a resumed run replays the same plan and the cache answers whatever finished. No separate checkpoint state to keep in sync.
- **Repeats are fresh calls:** repeat 2 and 3 get their own cache entries; repeat 1 shares the normal cache, so earlier single runs count as repeat 1.
- **Estimate errs high:** rough token counts (4 characters a token, no tokenizer download) and a 1,500-token default output for models with no history, because reasoning models spend hidden tokens. Once a model has run, its real average output length is used.
- **Budget is a hard stop, with a small overshoot:** once spent reaches the cap, no new call starts; calls already in flight (up to 4 per provider) still finish.
- **Not recorded yet:** the exact snapshot a provider served behind an alias like `claude-haiku-4-5`. The config records the ids and date; the served version needs a call-log column and can come later.

## 2026-10-16: the judge gets a cost estimate and budget too
- **Found:** judging 21 cards cost $0.70, more than writing them. A full run of 240 cards would be about $8 to judge, with no warning.
- **Fix:** `bench judge` estimates first (cached judgments free; output length from each judge's history), refuses to start over the budget, and stops new calls once it's spent. `--sample N` judges a random subset, so a cheap sample can be checked before paying for everything.

## 2026-10-16: rank models only on metrics they all have
- **Found:** with a judge sample, one model can have judge scores and another none. A weighted score over "whatever was measured" then ranks the unjudged model on easy checks alone, and it looks best.
- **Fix:** the ranking uses only metrics every ranked model has at least 5 measured outputs for, with the locked weights re-normalized over them. The report names those metrics and the share of the locked weight they cover, so a checks-only ranking is labelled as one.
- **Not ranked:** models that finished under half their cases. Gemini's 3 of 60 can't support a conclusion.

## 2026-10-16: judge samples per model, and real cost and speed
- **Found:** the first full report ranked on three checks only (25% of the locked weight), because a 10-card judge sample landed 6 on one model and 1 or 2 on the others. Cached outputs also counted as $0 and 0 s, so Gemini's 3 reused cards showed as free and instant, and other models looked cheaper and faster than they are. Llama, run before its price was set, stayed "unknown" for good.
- **Fix:** `--sample N` now means N cards per model, so 5 per model puts the judge metrics into the ranking. Cost and speed come only from outputs a run generated itself. `bench analyze` costs unpriced outputs from their token counts once a price is added to models.yaml. A model that isn't ranked shows "not ranked" instead of a score, and the recommendation doesn't call a model cheaper than one whose price is unknown.

## 2026-10-16: "where" and consistency on the same metrics as the ranking
- **Found:** with judging on a sample, the case-type table scored each card on whatever was measured on it. Most cards had only checks, so every cell read 0.8 to 1.0 while the ranked scores were near 0.6. Claude Haiku looked best on tricky cases (0.94) despite failing every judged fact check. The consistency spread mixed judged and unjudged repeats the same way.
- **Fix:** the table uses the scorecard's metrics and weights, shows how many outputs each cell rests on, and leaves a cell blank when it has no judged card. Consistency compares repeats only on metrics measured on all of them.

## 2026-10-17: web app in Python, no separate front end
- **Choice:** FastAPI serving server-rendered pages from the same repo, instead of Next.js plus FastAPI or Streamlit.
- **Why:** proof-of-concept scale. One `uv run python -m bench serve` on Windows, no Node toolchain or second server, and the pages reuse `bench.analyze` directly, so the browser and `bench analyze` can't disagree. Streamlit would be quicker but reads as a data tool, and the reviewer is a marketer, not an analyst.
- **Trade-off:** less interactive than a React app. The pages are plain HTML over a small API surface, so a React front end can be added later without touching the engine.
- **Runs from the browser:** a run or judging job runs in a background thread of the local app, with the same estimate, budget and config file as the CLI. Jobs live in memory, so a restart loses the progress bar but not the work: finished calls are in the cache and the config can be resumed. Judging is a separate step on the results page, so outputs can be checked before paying for it.

## 2026-10-17: human review scores each card, blind
- **Choice:** the reviewer scores one card at a time, 1 to 5, on the same rubrics as the judge, without seeing the model, the checks or the judge's scores. The mockup had a pairwise "which is better" screen instead.
- **Why:** per-card scores line up one-to-one with the judge's, so agreement is a direct number per metric (same score, within one point, judge higher or lower). Pairwise picks would only say which card a person prefers, not whether the judge's 4 is a person's 4. The same screen also scores `rep_ready`, which has no judge.
- **Trust label:** results say "Judge not checked against a person yet" until 20 cards are reviewed, then the within-one-point agreement. Agreement is shown only after scoring, never on the card being scored, so it can't anchor the reviewer.
- **Order:** a fixed shuffle so cards from one model don't come in a row. A skip is recorded so the card leaves the queue; a later review of the same card replaces the earlier one.
