"""Debate (variant D): reviewers who disagree get to respond to each other.

Anti-conformity rules, from evidence that agents cave to a visible majority
(Can LLM Agents Really Debate?, 2025):
- Arguments are shown grouped by position, anonymous and de-duplicated, with no
  count of how many reviewers hold each position.
- Reviewers in the minority argue first; the others then see their new arguments.
- A reviewer may change its answer only by citing a quote it did not cite before
  that really appears in the article. Otherwise the change is rejected and its
  previous answer stands.
- Earlier rounds are never overwritten: every round is a new vote.
"""

from __future__ import annotations

from collections import Counter

from adrd_review.agents.common import (
    EVIDENCE_RULES,
    allowed_labels,
    check_quotes,
    render_article,
    render_field,
    render_sides,
    show_labels,
)
from adrd_review.codebook import Codebook
from adrd_review.llm import LLM
from adrd_review.schemas import Article, DebateAnswer, Usage, Vote

SYSTEM = """\
You are a reviewer in a scoping review of dementia screening research.
{persona}Reviewers disagreed on one decision for this article. You will see the arguments made
for each position. You are NOT told how many reviewers hold each position: judge the arguments.

Decision: {field}
Valid answers: {allowed}

{rules}

Reconsider your answer. You may keep it or change it.
- To CHANGE your answer you must cite at least one quote you did not cite before, copied word for
  word from the article. A change without new evidence will not be accepted.
- Answer the strongest opposing argument in your rationale.

{evidence_rules}"""


def system_prompt(codebook: Codebook, field: str, persona: str | None = None) -> str:
    return SYSTEM.format(
        persona=f"Your perspective: {persona}\n" if persona else "",
        field=field,
        allowed=", ".join(allowed_labels(codebook, field)),
        rules=render_field(codebook, field),
        evidence_rules=EVIDENCE_RULES,
    )


def user_prompt(article: Article, own: Vote, all_votes: list[Vote]) -> str:
    return (
        render_article(article)
        + f"\n\nYOUR PREVIOUS ANSWER: {show_labels(own.labels)}\n"
        + (f"Your previous reasoning: {own.rationale}\n" if own.rationale else "")
        + "\nARGUMENTS FOR EACH POSITION (anonymous, not counted)\n\n"
        + render_sides(all_votes)
    )


def minority_first(votes: list[Vote]) -> list[Vote]:
    """Order reviewers so those holding the least common position speak first."""
    counts = Counter(tuple(sorted(v.labels)) for v in votes)
    return sorted(votes, key=lambda v: (counts[tuple(sorted(v.labels))], v.reviewer_id))


def run(
    article: Article,
    codebook: Codebook,
    llm: LLM,
    field: str,
    own: Vote,
    all_votes: list[Vote],
    round_no: int,
    persona: str | None = None,
) -> tuple[Vote, Usage]:
    result = llm.structured(
        DebateAnswer, system_prompt(codebook, field, persona), user_prompt(article, own, all_votes), role="debate"
    )
    ans = result.parsed
    valid = allowed_labels(codebook, field)
    labels = [x for x in dict.fromkeys(ans.labels) if x in valid]
    warnings = []
    if len(labels) != len(ans.labels):
        warnings.append(f"invalid labels dropped: {[x for x in ans.labels if x not in valid]}")
    single_answer = field in ("screening", "eligibility")
    if single_answer and len(labels) != 1:
        warnings.append(f"{field} needs exactly one answer, got {ans.labels}; kept previous answer")
        labels = own.labels
    elif not single_answer and not codebook.dimensions[field].multi_label and len(labels) > 1:
        warnings.append(f"single-label dimension: kept {labels[0]!r}")
        labels = labels[:1]

    fake_quotes = check_quotes(ans.new_evidence, article)
    warnings += fake_quotes
    old_quotes = {" ".join(e.quote.split()).lower() for e in own.evidence}
    genuinely_new = [
        e
        for e in ans.new_evidence
        if " ".join(e.quote.split()).lower() not in old_quotes and not check_quotes([e], article)
    ]
    changed = sorted(labels) != sorted(own.labels)
    if changed and not genuinely_new:
        warnings.append(
            f"change from {show_labels(own.labels)} to {show_labels(labels)} rejected: no new evidence found in the article"
        )
        labels, changed = own.labels, False

    vote = Vote(
        reviewer_id=own.reviewer_id,
        model=llm.config.label,
        field=field,
        labels=labels,
        round=round_no,
        rule_ids=ans.rule_ids,
        evidence=own.evidence + genuinely_new,
        rationale=ans.rationale,
        confidence=ans.confidence,
        warnings=warnings,
    )
    return vote, result.usage
