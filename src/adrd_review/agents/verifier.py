"""Verifier: checks that each final decision is backed by real evidence.

Two layers:
1. Deterministic (always runs, no model): for each decision, the quotes cited by
   the reviewers who support it must appear in the article. If every supporting
   quote is missing, the decision is flagged.
2. Model check (only if a verifier model is configured): does the quoted evidence
   and the cited rules actually support the decision?

A failed decision is sent to a human; the verifier never changes a decision itself.
"""

from __future__ import annotations

from adrd_review.agents.common import check_quotes, render_article, render_field, show_labels
from adrd_review.codebook import Codebook
from adrd_review.llm import LLM
from adrd_review.schemas import Article, Decision, Usage, VerificationReport, VerifierAnswer, Vote

SYSTEM = """\
You are the Verifier in a scoping review of {topic}.
For each decision below, check whether the quoted evidence really appears to say what is claimed,
and whether it satisfies the cited rules. Do not re-review the article from scratch and do not
change any decision: only report whether each decision is supported.

What "supported" means depends on the stage:
- screening (title/abstract) = include: screening keeps an article unless a rule is CLEARLY violated, so
  an include is supported even when some rules are unclear (an abstract often does not report the age or
  design; that is checked later in the full paper). Mark it unsupported ONLY if the text clearly shows
  that an inclusion rule is not met or that an exclusion rule applies.
- screening = exclude: supported only if the evidence clearly shows a rule is not met or an exclusion applies.
- eligibility (full paper) = include: every inclusion rule must be shown to be met.
- eligibility = exclude: supported only if the evidence clearly shows a rule is not met or an exclusion applies.

{rules}"""


def supporting_votes(decision: Decision, votes: list[Vote]) -> list[Vote]:
    """Latest-round votes on the decision's field that agree with it (multi-label: share a label).

    Votes are matched by reviewer AND field: the verifier receives screening and
    annotation votes together, and one reviewer's annotation votes must not hide
    its screening vote.
    """
    latest: dict[str, Vote] = {}
    for v in sorted(votes, key=lambda v: v.round):
        if v.field == decision.field:
            latest[v.reviewer_id] = v
    if decision.field == "screening" or not decision.labels:
        return [v for v in latest.values() if v.labels == decision.labels]
    return [v for v in latest.values() if set(v.labels) & set(decision.labels)]


def deterministic_problems(article: Article, decisions: list[Decision], votes: list[Vote]) -> list[str]:
    problems = []
    for d in decisions:
        support = supporting_votes(d, votes)
        evidence = [e for v in support for e in v.evidence]
        if evidence and len(check_quotes(evidence, article)) == len(evidence):
            problems.append(f"{d.field}: none of the supporting quotes appear in the article")
    return problems


def user_prompt(article: Article, decisions: list[Decision], votes: list[Vote]) -> str:
    parts = [render_article(article), "", "DECISIONS TO CHECK"]
    for d in decisions:
        support = supporting_votes(d, votes)
        parts.append(f"\nDecision: {d.field} = {show_labels(d.labels)}")
        rules = sorted({r for v in support for r in v.rule_ids})
        if rules:
            parts.append(f"  Rules cited: {', '.join(rules)}")
        for q in sorted({e.quote for v in support for e in v.evidence}):
            parts.append(f'  Evidence: "{q}"')
    return "\n".join(parts)


def run(
    article: Article,
    codebook: Codebook,
    decisions: list[Decision],
    votes: list[Vote],
    llm: LLM | None = None,
) -> tuple[VerificationReport, Usage | None]:
    report = VerificationReport(quote_problems=deterministic_problems(article, decisions, votes))
    checkable = [d for d in decisions if d.labels]
    if llm is None or not checkable:
        return report, None

    rules = "\n\n".join(render_field(codebook, f) for f in dict.fromkeys(d.field for d in checkable))
    result = llm.structured(VerifierAnswer, SYSTEM.format(topic=codebook.topic, rules=rules), user_prompt(article, checkable, votes), role="verifier")
    wanted = {d.field for d in checkable}
    report.checks = [c for c in result.parsed.checks if c.field in wanted]
    report.model = llm.config.label
    return report, result.usage
