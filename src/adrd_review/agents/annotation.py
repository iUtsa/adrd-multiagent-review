"""Annotation reviewer: label each codebook dimension for an included study."""

from __future__ import annotations

from adrd_review.agents.common import EVIDENCE_RULES, check_quotes, render_article, render_dimensions
from adrd_review.codebook import Codebook
from adrd_review.llm import LLM
from adrd_review.schemas import AnnotationAnswer, Article, ExtractionRecord, Usage, Vote

SYSTEM = """\
You are a reviewer annotating included studies for a scoping review of dementia screening research.
{persona}
For each dimension below, choose labels using ONLY the label IDs listed for it, spelled exactly.
Leave the labels empty if the text does not support any of them.

{dimensions}

{evidence_rules}

Return one entry per dimension, in the order given."""


def system_prompt(codebook: Codebook, persona: str | None = None) -> str:
    return SYSTEM.format(
        persona=f"Your perspective: {persona}\n" if persona else "",
        dimensions=render_dimensions(codebook),
        evidence_rules=EVIDENCE_RULES,
    )


def render_extraction(extraction: ExtractionRecord | None) -> str:
    if extraction is None:
        return ""
    lines = ["", "FACTS ALREADY EXTRACTED (check them against the article):"]
    for name, field in extraction:
        values = field if isinstance(field, list) else [field]
        shown = [v.value for v in values if v.value]
        if shown:
            lines.append(f"- {name}: {'; '.join(shown)}")
    return "\n".join(lines) if len(lines) > 2 else ""


def run(
    article: Article,
    codebook: Codebook,
    llm: LLM,
    extraction: ExtractionRecord | None = None,
    reviewer_id: str = "A",
    persona: str | None = None,
) -> tuple[list[Vote], Usage]:
    user = render_article(article) + render_extraction(extraction)
    result = llm.structured(AnnotationAnswer, system_prompt(codebook, persona), user, role="annotation")
    ans = result.parsed
    by_dim = {a.dimension: a for a in ans.annotations}
    votes = []
    for name, dim in codebook.dimensions.items():
        a = by_dim.get(name)
        warnings = [] if a else ["model returned no entry for this dimension"]
        labels = list(dict.fromkeys(a.labels)) if a else []
        unknown = [x for x in labels if x not in dim.label_ids]
        if unknown:
            warnings.append(f"labels not in codebook dropped: {unknown}")
            labels = [x for x in labels if x in dim.label_ids]
        if not dim.multi_label and len(labels) > 1:
            warnings.append(f"single-label dimension: kept {labels[0]!r}, dropped {labels[1:]}")
            labels = labels[:1]
        evidence = a.evidence if a else []
        votes.append(
            Vote(
                reviewer_id=reviewer_id,
                model=llm.config.label,
                field=name,
                labels=labels,
                evidence=evidence,
                rationale=a.rationale if a else "",
                confidence=ans.confidence,
                warnings=warnings + check_quotes(evidence, article),
            )
        )
    return votes, result.usage
