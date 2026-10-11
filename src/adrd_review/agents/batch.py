"""Batch screening: one model call judges many titles, or a few abstracts, at once.

Why: free model tiers limit requests (and tokens) per day, not quality. Every
call repeats the same instructions and answer form (~2,500 tokens), so judging
one title per call spends most of the budget on repetition. Batching keeps the
same rules and the same per-article decision, and cuts calls about 50-fold for
titles and 5-fold for abstracts.

- Titles: the reviewer lists only the titles that clearly break a rule, each with
  the rule and the exact words of THAT title; every other title is kept (the
  title stage only removes the obviously off topic).
- Abstracts: the reviewer returns the full per-rule form for every article, and
  `screening.decide` makes each article's decision exactly as in single calls.

Quotes are checked against the one article they are about, so evidence cannot
leak between articles. An article the model leaves out of its answer is kept,
with a warning (the safe direction).
"""

from __future__ import annotations

from adrd_review.agents import screening
from adrd_review.agents.common import EVIDENCE_RULES, check_quotes, render_article, render_screening_rules
from adrd_review.agents.title_screening import title_only
from adrd_review.codebook import Codebook
from adrd_review.llm import LLM
from adrd_review.schemas import (
    Article,
    EvidenceSpan,
    ScreeningAnswer,
    ScreeningBatchAnswer,
    TitleBatchAnswer,
    Usage,
    Vote,
)

TITLE_SYSTEM = """\
You are a reviewer screening TITLES ONLY for a scoping review of {topic}.
{persona}
You will see a numbered list of titles. This is the first, quickest pass: its only job is to remove
articles whose TITLE ALONE makes it obvious they are off topic. Anything that might be relevant goes on
to abstract screening.

{rules}

Report a title ONLY if the title itself CLEARLY shows that an inclusion rule is not met (for example it
names a different disease, or the study is plainly about something else) or that an exclusion rule
applies (for example "a systematic review"). For each such title give its number, the rule ID, and the
exact words from that title that show it.
Report an inclusion rule as not met ONLY when the title states something INCOMPATIBLE with it: a
different disease and no dementia or Alzheimer's (for example Parkinson's disease, stroke, depression),
animals, a review. A title that simply does not MENTION dementia, Alzheimer's, AI or machine learning,
detection or prediction, or age is NEVER a reason: many relevant studies have short or technical titles,
and their abstracts are read next. When in doubt, do not report the title.
Most titles will not be reported; that is expected. Titles you do not report are kept."""

ABSTRACT_SYSTEM = """\
You are a reviewer screening titles and abstracts for a scoping review of {topic}.
{persona}
You will see several articles, numbered. Judge EACH article separately, only from its own text; never
use one article's text for another. You do NOT decide include or exclude. For each article you judge
each rule, and the decision is then made from your judgments.

{rules}

How to judge each INCLUSION rule:
- met: the text shows the rule is satisfied.
- not_met: the text CLEARLY shows the rule is not satisfied (for example, the study is about something else).
- unclear: the text does not say enough to tell.
An abstract that simply does not mention something is UNCLEAR, never not_met. Details such as the
participants' age or the exact design are often left out of abstracts and are checked in the full text.

How to judge each EXCLUSION rule:
- applies: the text CLEARLY shows the exclusion applies.
- does_not_apply: the text shows it does not.
- unclear: the text does not say enough to tell.

Give an exact quote from THAT article for every met, not_met and applies. A not_met or applies without
a real quote from that article will be treated as unclear.

{evidence_rules}

Return exactly one entry per article, with its number: one judgment per rule, a short rationale, and
your confidence from 0 to 1."""


def _persona(persona: str | None) -> str:
    return f"Your perspective: {persona}\n" if persona else ""


def title_system(codebook: Codebook, persona: str | None = None) -> str:
    return TITLE_SYSTEM.format(topic=codebook.topic, persona=_persona(persona),
                               rules=render_screening_rules(codebook, decision_rule=False))


def abstract_system(codebook: Codebook, persona: str | None = None) -> str:
    return ABSTRACT_SYSTEM.format(topic=codebook.topic, persona=_persona(persona),
                                  rules=render_screening_rules(codebook), evidence_rules=EVIDENCE_RULES)


def render_titles(articles: list[Article]) -> str:
    return "TITLES\n" + "\n".join(f"{i}. {a.title}" for i, a in enumerate(articles, start=1))


def render_abstracts(articles: list[Article]) -> str:
    return "\n\n".join(f"=== ARTICLE {i} ===\n{render_article(a)}" for i, a in enumerate(articles, start=1))


def _share(usage: Usage, n: int) -> Usage:
    """One article's share of a batch call's tokens and cost (for per-article accounting)."""
    return usage.model_copy(update={
        "input_tokens": round(usage.input_tokens / n), "output_tokens": round(usage.output_tokens / n),
        "cost_usd": None if usage.cost_usd is None else usage.cost_usd / n, "seconds": usage.seconds / n,
        "batch_size": n,
    })


def run_titles(articles: list[Article], codebook: Codebook, llm: LLM, reviewer_id: str = "A",
               persona: str | None = None) -> tuple[dict[str, Vote], dict[str, Usage]]:
    """Votes on the field 'title' for every article in the batch, and each article's share of the cost."""
    result = llm.structured(TitleBatchAnswer, title_system(codebook, persona), render_titles(articles), role="title")
    known = {r.id: r for r in codebook.rules}
    inclusion = {r.id for r in codebook.screening.inclusion}
    hits: dict[int, list] = {}
    stray: list[str] = []
    for h in result.parsed.excluded:
        if 1 <= h.n <= len(articles) and h.rule_id in known:
            hits.setdefault(h.n, []).append(h)
        else:
            stray.append(f"ignored: title {h.n} / rule {h.rule_id!r}")
    votes, usages = {}, {}
    for i, art in enumerate(articles, start=1):
        checks, rule_ids, evidence, warnings = {}, [], [], list(stray)
        for h in hits.get(i, []):
            if not h.quote.strip() or check_quotes([EvidenceSpan(quote=h.quote)], title_only(art)):
                warnings.append(f"{h.rule_id}: quote not in this title; ignored: {h.quote[:80]!r}")
                continue
            checks[h.rule_id] = "not_met" if h.rule_id in inclusion else "applies"
            rule_ids.append(h.rule_id)
            evidence.append(EvidenceSpan(quote=h.quote))
        votes[art.pmid] = Vote(
            reviewer_id=reviewer_id, model=llm.config.label, field="title",
            labels=["exclude" if rule_ids else "include"], rule_ids=rule_ids, evidence=evidence,
            rationale=f"Batch title screening ({len(articles)} titles in one call).",
            confidence=0.8, warnings=warnings, rule_checks=checks,
        )
        usages[art.pmid] = _share(result.usage, len(articles))
    return votes, usages


def run_abstracts(articles: list[Article], codebook: Codebook, llm: LLM, reviewer_id: str = "A",
                  persona: str | None = None) -> tuple[dict[str, Vote], dict[str, Usage]]:
    """Votes on the field 'screening' for every article in the batch (same decision rule as single calls)."""
    result = llm.structured(ScreeningBatchAnswer, abstract_system(codebook, persona), render_abstracts(articles),
                            role="screening")
    by_n = {}
    for item in result.parsed.items:
        by_n.setdefault(item.n, item)
    votes, usages = {}, {}
    n_inc = len(codebook.screening.inclusion)
    for i, art in enumerate(articles, start=1):
        item = by_n.get(i)
        if item is None:
            ans = ScreeningAnswer(inclusion=[], exclusion=[], rationale="", confidence=0.5)
            extra = [f"article {i} missing from the batch answer; kept"]
        else:
            ans, extra = ScreeningAnswer(**item.model_dump(exclude={"n"})), []
        decision, checks, rule_ids, evidence, warnings = screening.decide(ans, codebook, art)
        met = sum(checks[r.id] == "met" for r in codebook.screening.inclusion)
        confidence = 0.5 + 0.5 * met / n_inc if decision == "include" else max(0.5, ans.confidence)
        votes[art.pmid] = Vote(
            reviewer_id=reviewer_id, model=llm.config.label, field="screening", labels=[decision],
            rule_ids=rule_ids, evidence=evidence, rationale=ans.rationale, confidence=round(confidence, 4),
            warnings=extra + warnings, rule_checks=checks,
        )
        usages[art.pmid] = _share(result.usage, len(articles))
    return votes, usages
