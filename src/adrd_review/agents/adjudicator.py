"""Adjudicator: settles a field the reviewers disagree on.

It does not override the panel. It weighs each reviewer's position against the
article and the rules (0-1 per reviewer), and the final answer is the panel's
votes combined with those weights. It must also state its own answer; if that
does not match the weighted result, the decision is flagged and sent to a human.
"""

from __future__ import annotations

from adrd_review.agents.common import (
    EVIDENCE_RULES,
    allowed_labels,
    render_article,
    render_field,
    render_votes,
    show_labels,
)
from adrd_review.codebook import Codebook
from adrd_review.consensus import combine_labels, combine_screening
from adrd_review.llm import LLM
from adrd_review.schemas import Adjudication, AdjudicationAnswer, Article, Decision, Usage, Vote

SYSTEM = """\
You are the senior adjudicator in a scoping review of dementia screening research.
Independent reviewers disagreed on one decision for this article. Settle it.

Decision to settle: {field}
Valid answers: {allowed}

{rules}

How to adjudicate:
- Judge the evidence and the rules, not the number of reviewers on each side.
- Check every quote against the article: a quote that is not in the article counts for nothing.
- Give each reviewer a weight from 0 to 1 for how well the text and rules support their position.
- State your own answer, the rules it rests on, and your confidence. If the article is genuinely
  ambiguous, say so with a low confidence: a human will then decide.

{evidence_rules}"""


def system_prompt(codebook: Codebook, field: str) -> str:
    return SYSTEM.format(
        field=field,
        allowed=", ".join(allowed_labels(codebook, field)),
        rules=render_field(codebook, field),
        evidence_rules=EVIDENCE_RULES,
    )


def user_prompt(article: Article, votes: list[Vote]) -> str:
    return render_article(article) + "\n\nREVIEWERS' ANSWERS\n\n" + render_votes(votes)


def run(
    article: Article,
    codebook: Codebook,
    llm: LLM,
    field: str,
    votes: list[Vote],
) -> tuple[Decision, Adjudication, Usage]:
    result = llm.structured(AdjudicationAnswer, system_prompt(codebook, field), user_prompt(article, votes), role="adjudicator")
    ans = result.parsed
    warnings: list[str] = []

    valid = allowed_labels(codebook, field)
    own = [x for x in dict.fromkeys(ans.labels) if x in valid]
    if len(own) != len(ans.labels):
        warnings.append(f"invalid labels dropped: {[x for x in ans.labels if x not in valid]}")

    ids = {v.reviewer_id for v in votes}
    weights = {w.reviewer: w.weight for w in ans.reviewer_weights if w.reviewer in ids}
    for rid in sorted(ids - weights.keys()):
        weights[rid] = 0.5
        warnings.append(f"no weight given for reviewer {rid}; used 0.5")
    if not any(weights.values()):
        weights = dict.fromkeys(ids, 1.0)
        warnings.append("all weights were 0; used equal weights")

    if field == "screening":
        combined = combine_screening(votes, weights)
    else:
        dim = codebook.dimensions[field]
        combined = combine_labels(votes, field, dim.multi_label, dim.label_ids, weights)

    confidence = ans.confidence
    if sorted(own) != sorted(combined.labels):
        warnings.append(
            f"adjudicator's own answer ({show_labels(own)}) differs from its weighted result "
            f"({show_labels(combined.labels)}); sent to a human"
        )
        confidence = min(confidence, 0.0)

    decision = Decision(
        field=field,
        labels=combined.labels,
        method="adjudicator",
        confidence=round(confidence, 4),
        include_score=combined.include_score if field == "screening" else None,
    )
    record = Adjudication(
        field=field,
        model=llm.config.label,
        labels=own,
        reviewer_weights=weights,
        rule_ids=ans.rule_ids,
        rationale=ans.rationale,
        confidence=ans.confidence,
        warnings=warnings,
    )
    return decision, record, result.usage
