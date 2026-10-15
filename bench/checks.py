"""Automatic checks: cheap, objective rules run on every output.

Anything a rule can catch is caught here, so the LLM judge only has to score
what needs judgment. Each check returns a CheckResult: pass/fail plus the
evidence (what was missing, which numbers were invented, and so on).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


@dataclass
class CheckResult:
    name: str
    passed: bool
    value: float | int | None = None
    detail: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'’$.%,/-]*", text)


def _source_text(inputs: dict) -> str:
    return "\n".join(str(v) for v in inputs.values())


def required_sections(output: str, inputs: dict, *, sections: list[str]) -> CheckResult:
    """Every listed Markdown heading appears (any heading level, case-insensitive)."""
    headings = {h.strip().lower() for h in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", output, re.M)}
    missing = [s for s in sections if s.lower() not in headings]
    return CheckResult("required_sections", not missing, len(sections) - len(missing), missing)


def max_words(output: str, inputs: dict, *, limit: int) -> CheckResult:
    n = len(_words(output))
    return CheckResult("max_words", n <= limit, n, [] if n <= limit else [f"{n} words, limit {limit}"])


MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?\s?(?:[kKmM]\b)?")


def _norm_money(s: str) -> str:
    s = s.replace(" ", "").replace(",", "").lower()
    if s.endswith(".00"):
        s = s[:-3]
    return s


def prices_in_source(output: str, inputs: dict) -> CheckResult:
    """Every dollar amount in the output appears in the inputs. Catches invented prices."""
    source = {_norm_money(m) for m in MONEY.findall(_source_text(inputs))}
    invented = sorted({m.strip() for m in MONEY.findall(output) if _norm_money(m) not in source})
    return CheckResult("prices_in_source", not invented, len(invented), invented)


NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?:\s?%|\s?[kKmM]\b|\+)?")


def _bare(n: str) -> str:
    """'1,200/month' -> '1200', '60%' -> '60', '30k' -> '30k' (scale kept, unit dropped)."""
    n = n.replace(",", "").replace(" ", "").lower().rstrip("+%")
    return n


def unsourced_numbers(output: str, inputs: dict) -> CheckResult:
    """Numbers in the output that never appear in the inputs.

    A rough fact check: it flags "70%" when the source only ever says "60%". It
    can't catch an invented claim with no number in it; that's the judge's job.
    Whole numbers up to 10 are ignored ("three differentiators", "step 2").
    """
    source = {_bare(n) for n in NUMBER.findall(_source_text(inputs))}
    found = set()
    for n in NUMBER.findall(output):
        b = _bare(n)
        if b.isdigit() and int(b) <= 10:
            continue
        if b not in source:
            found.add(n.strip())
    return CheckResult("unsourced_numbers", not found, len(found), sorted(found))


def banned_terms(output: str, inputs: dict, *, terms: list[str]) -> CheckResult:
    low = output.lower()
    hits = [t for t in terms if re.search(rf"\b{re.escape(t.lower())}\b", low)]
    return CheckResult("banned_terms", not hits, len(hits), hits)


def _syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(n, 1)


def reading_grade(output: str, inputs: dict, *, max_grade: float = 12) -> CheckResult:
    """Flesch-Kincaid grade level, computed directly (no NLTK download needed).

    Markdown headings and bullets count as sentences, so terse cards read a bit
    lower than prose would.
    """
    text = re.sub(r"[#*_`>|-]", " ", output)
    sentences = max(len([s for s in re.split(r"[.!?\n]+", text) if s.strip()]), 1)
    words = [w for w in re.findall(r"[A-Za-z]+", text)]
    if not words:
        return CheckResult("reading_grade", False, None, ["no words"])
    syllables = sum(_syllables(w) for w in words)
    grade = round(0.39 * len(words) / sentences + 11.8 * syllables / len(words) - 15.59, 1)
    return CheckResult("reading_grade", grade <= max_grade, grade, [] if grade <= max_grade else [f"grade {grade}"])


CHECKS = {
    "required_sections": required_sections,
    "max_words": max_words,
    "prices_in_source": prices_in_source,
    "unsourced_numbers": unsourced_numbers,
    "banned_terms": banned_terms,
    "reading_grade": reading_grade,
}


def run_checks(output: str, inputs: dict, specs: list[dict]) -> dict[str, CheckResult]:
    """`specs` is a list of {"name": <result key>, "check": <CHECKS key>, "args": {...}}."""
    return {s["name"]: CHECKS[s["check"]](output, inputs, **(s.get("args") or {})) for s in specs}


def check_specs(metrics_spec: dict) -> list[dict]:
    """The checks a task's metrics.yaml asks for: scored check metrics plus unscored diagnostics."""
    specs = [{"name": m["name"], "check": m["check"], "args": m.get("args")}
             for m in metrics_spec.get("metrics", []) if m.get("type") == "check"]
    specs += [{"name": d["name"], "check": d.get("check", d["name"]), "args": d.get("args")}
              for d in metrics_spec.get("diagnostics", [])]
    return specs
