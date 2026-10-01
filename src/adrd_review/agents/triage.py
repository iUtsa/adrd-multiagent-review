"""Triage agent: can this article be judged from its title and abstract?

Two questions for the model:
  - clearly_ineligible: does the text already show the study cannot qualify?
  - missing: which inclusion rules cannot be judged either way?
The code then sets enough_information = clearly_ineligible or nothing is missing.
(An off-topic paper is fully judgeable: it needs no full text.)
"""

from __future__ import annotations

from adrd_review.agents.common import render_article, render_screening_rules
from adrd_review.codebook import Codebook
from adrd_review.llm import LLM
from adrd_review.schemas import Article, TriageReport, Usage

SYSTEM = """\
You are the Triage agent in a scoping review of dementia screening research.
You do NOT decide whether to include the study. You check whether the title and abstract give a
reviewer enough to judge it.

{rules}

Answer two things:
1. clearly_ineligible: true if the text already shows the study cannot qualify, for example it is
   about a different topic, or an inclusion rule is plainly not satisfied. Such an article needs
   nothing more.
2. missing: if it is NOT clearly ineligible, list the IDs of inclusion rules that cannot be judged
   either way because the title and abstract do not say. A rule that is clearly met, or clearly not
   met, is not missing. An article with no abstract usually has several missing rules."""


def run(article: Article, codebook: Codebook, llm: LLM) -> tuple[TriageReport, Usage]:
    system = SYSTEM.format(rules=render_screening_rules(codebook))
    result = llm.structured(TriageReport, system, render_article(article), role="triage")
    report = result.parsed
    valid = {r.id for r in codebook.screening.inclusion}
    report.missing = [] if report.clearly_ineligible else [m for m in report.missing if m in valid]
    report.enough_information = report.clearly_ineligible or not report.missing
    return report, result.usage
