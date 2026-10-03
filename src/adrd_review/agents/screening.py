"""Screening reviewer: judges each codebook rule; the code then decides include/exclude.

The model does not give a verdict. For every inclusion rule it answers
met / not_met / unclear, and for every exclusion rule applies / does_not_apply /
unclear, each with an exact quote. `decide` then applies the title/abstract
screening rule mechanically:

    exclude  only if an inclusion rule is clearly NOT MET or an exclusion rule clearly APPLIES,
             and that judgment rests on a quote that really appears in the article
    include  otherwise ("unclear" never excludes: benefit of the doubt)

Why: in the first real-model run (2026-09-30) reviewers given a free include/exclude
choice treated "not reported in the abstract" as "not met" and excluded all 13
gold includes. Making the rule mechanical removes that failure mode and shows,
rule by rule, why each article was decided as it was.
"""

from __future__ import annotations

from adrd_review.agents.common import EVIDENCE_RULES, check_quotes, render_article, render_screening_rules
from adrd_review.codebook import Codebook
from adrd_review.llm import LLM
from adrd_review.schemas import Article, EvidenceSpan, ScreeningAnswer, Usage, Vote

SYSTEM = """\
You are a reviewer screening titles and abstracts for a scoping review of {topic}.
{persona}
You do NOT decide include or exclude. You judge each rule separately, and the decision is then made
from your judgments. Judge only from the article text you are given.

{rules}

How to judge each INCLUSION rule:
- met: the text shows the rule is satisfied.
- not_met: the text CLEARLY shows the rule is not satisfied (for example, the study is about something else).
- unclear: the text does not say enough to tell.
An abstract that simply does not mention something is UNCLEAR, never not_met. At this stage reviewers
only see the title and abstract; details such as the reference standard or the exact design are often
left out and will be checked in the full text.

How to judge each EXCLUSION rule:
- applies: the text CLEARLY shows the exclusion applies.
- does_not_apply: the text shows it does not.
- unclear: the text does not say enough to tell.

Give an exact quote for every met, not_met and applies. A not_met or applies without a real quote
from the article will be treated as unclear.

{evidence_rules}

Return one judgment per rule, a short rationale, and your confidence from 0 to 1."""


def system_prompt(codebook: Codebook, persona: str | None = None) -> str:
    return SYSTEM.format(
        topic=codebook.topic,
        persona=f"Your perspective: {persona}\n" if persona else "",
        rules=render_screening_rules(codebook),
        evidence_rules=EVIDENCE_RULES,
    )


def decide(answer: ScreeningAnswer, codebook: Codebook, article: Article) -> tuple[str, dict[str, str], list[str], list[EvidenceSpan], list[str]]:
    """Apply the screening rule to the model's rule judgments.

    Returns (decision, checks by rule ID, deciding rule IDs, evidence, warnings).
    """
    warnings: list[str] = []
    evidence: list[EvidenceSpan] = []
    checks: dict[str, str] = {}

    def quote_ok(quote: str | None) -> bool:
        return bool(quote and quote.strip()) and not check_quotes([EvidenceSpan(quote=quote)], article)

    given_inc = {c.rule_id: c for c in answer.inclusion}
    given_exc = {c.rule_id: c for c in answer.exclusion}
    known = {r.id for r in codebook.rules}
    unknown = sorted((set(given_inc) | set(given_exc)) - known)
    if unknown:
        warnings.append(f"unknown rule IDs dropped: {unknown}")

    for rule in codebook.screening.inclusion:
        c = given_inc.get(rule.id)
        if c is None:
            checks[rule.id] = "unclear"
            warnings.append(f"{rule.id}: no judgment given; treated as unclear")
            continue
        status = c.status
        if status in ("met", "not_met") and not quote_ok(c.quote):
            if c.quote:
                warnings.append(f"quote not found in article: {c.quote[:80]!r}")
            if status == "not_met":
                warnings.append(f"{rule.id}: not_met without a real quote; treated as unclear")
                status = "unclear"
        elif c.quote and status != "unclear":
            evidence.append(EvidenceSpan(quote=c.quote))
        checks[rule.id] = status

    for rule in codebook.screening.exclusion:
        c = given_exc.get(rule.id)
        if c is None:
            checks[rule.id] = "unclear"
            continue
        status = c.status
        if status == "applies" and not quote_ok(c.quote):
            if c.quote:
                warnings.append(f"quote not found in article: {c.quote[:80]!r}")
            warnings.append(f"{rule.id}: applies without a real quote; treated as unclear")
            status = "unclear"
        elif status == "applies":
            evidence.append(EvidenceSpan(quote=c.quote))
        checks[rule.id] = status

    against = [rid for rid, s in checks.items() if s in ("not_met", "applies")]
    if against:
        return "exclude", checks, against, evidence, warnings
    return "include", checks, [rid for rid, s in checks.items() if s == "met"], evidence, warnings


def run(
    article: Article,
    codebook: Codebook,
    llm: LLM,
    reviewer_id: str = "A",
    persona: str | None = None,
) -> tuple[Vote, Usage]:
    result = llm.structured(ScreeningAnswer, system_prompt(codebook, persona), render_article(article), role="screening")
    ans = result.parsed
    decision, checks, rule_ids, evidence, warnings = decide(ans, codebook, article)

    # Probability-like score used to rank articles: includes score higher the more rules are
    # clearly met; excludes score lower the surer the reviewer is.
    n_inc = len(codebook.screening.inclusion)
    met = sum(checks[r.id] == "met" for r in codebook.screening.inclusion)
    confidence = 0.5 + 0.5 * met / n_inc if decision == "include" else max(0.5, ans.confidence)

    vote = Vote(
        reviewer_id=reviewer_id,
        model=llm.config.label,
        field="screening",
        labels=[decision],
        rule_ids=rule_ids,
        evidence=evidence,
        rationale=ans.rationale,
        confidence=round(confidence, 4),
        warnings=warnings,
        rule_checks=checks,
    )
    return vote, result.usage
