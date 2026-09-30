"""Triage agent: is the title/abstract enough to apply the inclusion rules?"""

from __future__ import annotations

from adrd_review.agents.common import render_article, render_screening_rules
from adrd_review.codebook import Codebook
from adrd_review.llm import LLM
from adrd_review.schemas import Article, TriageReport, Usage

SYSTEM = """\
You are the Triage agent in a scoping review of dementia screening research.
You do NOT decide whether to include the study. You only check whether the title and
abstract contain enough information to apply each inclusion rule below.

{rules}

For each inclusion rule, ask: could a reviewer judge this rule from the title and abstract alone?
List the IDs of the rules that cannot be judged. Articles with no abstract usually cannot be judged."""


def run(article: Article, codebook: Codebook, llm: LLM) -> tuple[TriageReport, Usage]:
    system = SYSTEM.format(rules=render_screening_rules(codebook))
    result = llm.structured(TriageReport, system, render_article(article), role="triage")
    report = result.parsed
    valid = {r.id for r in codebook.screening.inclusion}
    report.missing = [m for m in report.missing if m in valid]
    return report, result.usage
