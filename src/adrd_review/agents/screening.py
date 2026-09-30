"""Screening reviewer: include or exclude, with rule IDs and exact quotes."""

from __future__ import annotations

from adrd_review.agents.common import EVIDENCE_RULES, check_quotes, render_article, render_screening_rules
from adrd_review.codebook import Codebook
from adrd_review.llm import LLM
from adrd_review.schemas import Article, ScreeningAnswer, Usage, Vote

SYSTEM = """\
You are a reviewer screening titles and abstracts for a scoping review of dementia screening research.
{persona}
Apply the rules below exactly as written. Judge only from the article text you are given.

{rules}

{evidence_rules}

Return your decision, the IDs of the rules it rests on (for an exclusion, the rule that fails or
applies), 1-3 exact supporting quotes, a short rationale, and your confidence from 0 to 1."""


def system_prompt(codebook: Codebook, persona: str | None = None) -> str:
    return SYSTEM.format(
        persona=f"Your perspective: {persona}\n" if persona else "",
        rules=render_screening_rules(codebook),
        evidence_rules=EVIDENCE_RULES,
    )


def run(
    article: Article,
    codebook: Codebook,
    llm: LLM,
    reviewer_id: str = "A",
    persona: str | None = None,
) -> tuple[Vote, Usage]:
    result = llm.structured(ScreeningAnswer, system_prompt(codebook, persona), render_article(article), role="screening")
    ans = result.parsed
    valid = {r.id for r in codebook.rules}
    unknown = [r for r in ans.rule_ids if r not in valid]
    warnings = [f"unknown rule IDs dropped: {unknown}"] if unknown else []
    warnings += check_quotes(ans.evidence, article)
    vote = Vote(
        reviewer_id=reviewer_id,
        model=llm.config.label,
        field="screening",
        labels=[ans.decision],
        rule_ids=[r for r in ans.rule_ids if r in valid],
        evidence=ans.evidence,
        rationale=ans.rationale,
        confidence=ans.confidence,
        warnings=warnings,
    )
    return vote, result.usage
