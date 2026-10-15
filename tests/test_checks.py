from bench import checks, metrics

SRC = {"our_product": "Acme: $18 per host per month; free for up to 5 hosts. 400+ integrations.",
       "competitor": "Northwind: $23 per host per month. Logs are $0.10/GB. 60% of tickets."}

CARD = """## Overview
Acme vs Northwind.
## Differentiators
- 400+ integrations
## Pricing
Acme is $18 per host; Northwind is $23.
## Objection handlers
- Too expensive? Logs cost extra at $0.10/GB.
## Landmines
- Ask about log costs.
"""


def test_required_sections_pass_and_report_missing():
    sections = ["Overview", "Differentiators", "Pricing", "Objection handlers", "Landmines"]
    assert checks.required_sections(CARD, SRC, sections=sections).passed
    r = checks.required_sections("## Overview\ntext", SRC, sections=sections)
    assert not r.passed and "Pricing" in r.detail


def test_max_words():
    assert checks.max_words(CARD, SRC, limit=450).passed
    r = checks.max_words("word " * 451, SRC, limit=450)
    assert not r.passed and r.value == 451


def test_prices_from_the_source_pass():
    assert checks.prices_in_source(CARD, SRC).passed


def test_invented_price_fails_with_the_amount():
    r = checks.prices_in_source("Northwind enterprise starts at $49/seat/month.", SRC)
    assert not r.passed and r.detail == ["$49"]


def test_price_formatting_differences_still_match():
    src = {"x": "Scale $1,200/month"}
    assert checks.prices_in_source("Scale costs $1200 a month ($1,200.00).", src).passed


def test_unsourced_numbers_flags_changed_stat_and_ignores_small_counts():
    r = checks.unsourced_numbers("Auto-triages 70% of tickets. Three differentiators, 2 steps.", SRC)
    assert r.detail == ["70%"]
    assert checks.unsourced_numbers("Auto-triages 60% of tickets across 400+ integrations.", SRC).passed


def test_banned_terms_whole_words_only():
    r = checks.banned_terms("A seamless, world-class rollout.", SRC, terms=["seamless", "world-class", "synergy"])
    assert r.detail == ["seamless", "world-class"]
    assert checks.banned_terms("Seamlessly done.", SRC, terms=["seamless"]).passed


def test_reading_grade_is_lower_for_simple_text():
    simple = checks.reading_grade("The cat sat. The dog ran. We ate.", SRC)
    hard = checks.reading_grade(
        "Comprehensive observability necessitates sophisticated instrumentation methodologies "
        "integrating heterogeneous telemetry infrastructures.", SRC)
    assert simple.value < hard.value and simple.passed and not hard.passed


def test_battlecard_metrics_name_real_checks():
    spec = metrics.load("battlecard")
    assert metrics.validate(spec) == []
    names = [s["name"] for s in checks.check_specs(spec)]
    assert names == ["pricing_matches_source", "required_sections", "one_page",
                     "unsourced_numbers", "banned_terms", "reading_grade"]


def test_unknown_check_is_a_validation_problem():
    spec = metrics.load("battlecard")
    spec["metrics"][1]["check"] = "vibes"
    assert any("check must name" in p for p in metrics.validate(spec))


def test_k_and_m_shorthand_match_written_out_numbers():
    src = {"x": "Growth $400/month (10k contacts), Scale $1.2M ARR, 50k contacts"}
    out = "Growth covers 10,000 contacts; Scale 50,000. Revenue $1,200,000."
    assert checks.unsourced_numbers(out, src).passed
    assert checks.prices_in_source(out, src).passed


def test_percent_and_plain_number_are_different_facts():
    src = {"x": "Deflects 40 tickets a day."}
    assert checks.unsourced_numbers("Deflects 40% of tickets.", src).detail == ["40%"]
