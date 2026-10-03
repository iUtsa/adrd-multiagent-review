"""Full-text eligibility reviewer (stage 2): judges each rule against the WHOLE paper.

Stage 1 (screening.py) reads the title and abstract and is lenient: unclear keeps
the article. Stage 2 reads the full paper and is strict, like a human reviewer at
full-text assessment:

    exclude  an inclusion rule is clearly NOT MET or an exclusion rule clearly APPLIES,
             backed by a quote that really appears in the paper
    include  EVERY inclusion rule is met and no exclusion rule applies
    unclear  anything else: the paper does not settle a rule -> a human decides

The model's answer form is the same per-rule form as stage 1 (ScreeningAnswer);
only the instructions and the decision rule differ.
"""

from __future__ import annotations

from adrd_review.agents import screening
from adrd_review.agents.common import EVIDENCE_RULES, render_article, render_screening_rules
from adrd_review.codebook import Codebook
from adrd_review.llm import LLM
from adrd_review.schemas import Article, ScreeningAnswer, Usage, Vote

SYSTEM = """\
You are a reviewer assessing the FULL PAPER for eligibility in a scoping review of {topic}.
{persona}
You do NOT decide include or exclude. You judge each rule separately against the whole paper, and the
decision is then made from your judgments.

{rules}

You have the full paper, not just the abstract. Look in the Methods and Results for the study design,
participants, setting, the test being evaluated and how dementia was confirmed.

How to judge each INCLUSION rule:
- met: the paper shows the rule is satisfied. Quote the passage.
- not_met: the paper shows the rule is NOT satisfied. Quote the passage.
- unclear: the paper does not report it anywhere, or reports it ambiguously. With the full paper in
  hand this should be uncommon; use it honestly rather than guessing.

How to judge each EXCLUSION rule:
- applies: the paper clearly shows the exclusion applies. Quote the passage.
- does_not_apply: the paper shows it does not.
- unclear: the paper does not settle it.

Every met, not_met and applies needs an exact quote from the paper; without one it counts as unclear.

{evidence_rules}

Return one judgment per rule, a short rationale, and your confidence from 0 to 1."""

REMINDER = (
    "\n\nREMINDER: judge each rule separately against the full paper above. Quote exact words for every "
    "met, not_met and applies. Do not give an overall include/exclude."
)


def system_prompt(codebook: Codebook, persona: str | None = None) -> str:
    return SYSTEM.format(
        topic=codebook.topic,
        persona=f"Your perspective: {persona}\n" if persona else "",
        rules=render_screening_rules(codebook, decision_rule=False),
        evidence_rules=EVIDENCE_RULES,
    )


def user_prompt(article: Article) -> str:
    # Instructions are repeated after the paper: models attend best to the start and end of long inputs.
    return render_article(article) + REMINDER


def decide(answer: ScreeningAnswer, codebook: Codebook, article: Article):
    """Stage 2 decision from the rule judgments. Returns (decision, checks, rule IDs, evidence, warnings)."""
    stage1, checks, rule_ids, evidence, warnings = screening.decide(answer, codebook, article)
    if stage1 == "exclude":
        return "exclude", checks, rule_ids, evidence, warnings
    inclusion = [r.id for r in codebook.screening.inclusion]
    if all(checks[r] == "met" for r in inclusion):
        return "include", checks, inclusion, evidence, warnings
    open_rules = [r for r in inclusion if checks[r] != "met"]
    open_rules += [r.id for r in codebook.screening.exclusion if checks[r.id] == "unclear" and r.id in
                   {c.rule_id for c in answer.exclusion if c.status == "applies"}]
    return "unclear", checks, open_rules, evidence, warnings


def run(
    article: Article,
    codebook: Codebook,
    llm: LLM,
    reviewer_id: str = "A",
    persona: str | None = None,
) -> tuple[Vote, Usage]:
    if not article.fulltext:
        raise ValueError(f"PMID {article.pmid}: eligibility needs the full text")
    result = llm.structured(ScreeningAnswer, system_prompt(codebook, persona), user_prompt(article), role="eligibility")
    ans = result.parsed
    decision, checks, rule_ids, evidence, warnings = decide(ans, codebook, article)
    vote = Vote(
        reviewer_id=reviewer_id,
        model=llm.config.label,
        field="eligibility",
        labels=[decision],
        rule_ids=rule_ids,
        evidence=evidence,
        rationale=ans.rationale,
        confidence=0.5 if decision == "unclear" else round(max(0.5, ans.confidence), 4),
        warnings=warnings,
        rule_checks=checks,
    )
    return vote, result.usage
