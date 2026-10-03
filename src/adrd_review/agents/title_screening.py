"""Title screening reviewer (stage 0): judges each rule from the TITLE ONLY.

The professor's team screened in three passes: titles, then abstracts, then full
papers. This agent is the first pass. It sees nothing but the title, so it only
removes articles whose title alone clearly shows they are off topic; everything
else goes on to abstract screening.

Same per-rule form as the other stages (ScreeningAnswer), same mechanical rule as
abstract screening (screening.decide): exclude only on a clear violation backed by
a quote, and the quote must come from the title itself. The panel then excludes
only when EVERY reviewer excludes (consensus.combine_title): one reviewer keeping
the article is enough, as in human title screening.
"""

from __future__ import annotations

from adrd_review.agents import screening
from adrd_review.agents.common import render_screening_rules
from adrd_review.codebook import Codebook
from adrd_review.llm import LLM
from adrd_review.schemas import Article, ScreeningAnswer, Usage, Vote

SYSTEM = """\
You are a reviewer screening TITLES ONLY for a scoping review of {topic}.
{persona}
You see only the title of each article. You do NOT decide include or exclude. You judge each rule
separately, and the decision is then made from your judgments.

{rules}

This is the first, quickest pass. Its only job is to remove articles whose TITLE ALONE makes it
obvious they are off topic. Anything that might be relevant goes on to abstract screening.

How to judge each INCLUSION rule:
- met: the title shows the rule is satisfied.
- not_met: the title CLEARLY shows the rule is not satisfied (for example, it names a different
  disease, or the study is plainly about something else).
- unclear: the title does not say. Most rules will be unclear from a title; that is expected.
A title that simply does not mention something is UNCLEAR, never not_met.

How to judge each EXCLUSION rule:
- applies: the title CLEARLY shows the exclusion applies (for example "a systematic review").
- does_not_apply: the title shows it does not.
- unclear: the title does not say.

Give an exact quote from the title for every met, not_met and applies. A not_met or applies without
a real quote from the title will be treated as unclear.

Return one judgment per rule, a short rationale, and your confidence from 0 to 1."""


def system_prompt(codebook: Codebook, persona: str | None = None) -> str:
    return SYSTEM.format(
        topic=codebook.topic,
        persona=f"Your perspective: {persona}\n" if persona else "",
        rules=render_screening_rules(codebook, decision_rule=False),
    )


def title_only(article: Article) -> Article:
    """The article as a title screener sees it: abstract, MeSH terms and full text removed,
    so quotes can only be checked against the title."""
    return article.model_copy(update={"abstract": "", "fulltext": None, "mesh_terms": []})


def render_title(article: Article) -> str:
    lines = ["ARTICLE", f"PMID: {article.pmid}", f"Title: {article.title}"]
    if article.year:
        lines.append(f"Year: {article.year}")
    return "\n".join(lines)


def run(
    article: Article,
    codebook: Codebook,
    llm: LLM,
    reviewer_id: str = "A",
    persona: str | None = None,
) -> tuple[Vote, Usage]:
    result = llm.structured(ScreeningAnswer, system_prompt(codebook, persona), render_title(article), role="title")
    ans = result.parsed
    decision, checks, rule_ids, evidence, warnings = screening.decide(ans, codebook, title_only(article))
    vote = Vote(
        reviewer_id=reviewer_id,
        model=llm.config.label,
        field="title",
        labels=[decision],
        rule_ids=rule_ids,
        evidence=evidence,
        rationale=ans.rationale,
        confidence=round(max(0.5, ans.confidence), 4),
        warnings=warnings,
        rule_checks=checks,
    )
    return vote, result.usage
