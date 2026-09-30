"""Prompt pieces shared by all agents. Everything rule-related is rendered from
the codebook, so a new codebook changes every prompt without code changes."""

from __future__ import annotations

from adrd_review.codebook import Codebook
from adrd_review.schemas import Article, EvidenceSpan

EVIDENCE_RULES = """\
Evidence rules:
- Every quote must be copied word for word from the article text below. Never paraphrase or combine sentences.
- Keep quotes short: the words that matter, at most two sentences.
- If the text does not say something, say it is not reported. Do not guess."""


def render_article(article: Article) -> str:
    lines = ["ARTICLE", f"PMID: {article.pmid}", f"Title: {article.title}"]
    meta = [x for x in (article.journal, str(article.year) if article.year else None) if x]
    if meta:
        lines.append(f"Journal: {', '.join(meta)}")
    if article.publication_types:
        lines.append(f"Publication type: {', '.join(article.publication_types)}")
    if article.language and article.language != ["eng"]:
        lines.append(f"Language: {', '.join(article.language)}")
    if article.mesh_terms:
        lines.append(f"MeSH terms: {'; '.join(article.mesh_terms[:20])}")
    lines.append("")
    lines.append("Abstract:")
    lines.append(article.abstract if article.has_abstract else "(no abstract available)")
    return "\n".join(lines)


def render_screening_rules(codebook: Codebook) -> str:
    s = codebook.screening
    lines = [f"Review question: {codebook.question.strip()}", ""]
    lines.append("Inclusion rules (all must be met):")
    lines += [f"- {r.id} {r.name}: {' '.join(r.text.split())}" for r in s.inclusion]
    if s.exclusion:
        lines.append("")
        lines.append("Exclusion rules (any one excludes):")
        lines += [f"- {r.id} {r.name}: {' '.join(r.text.split())}" for r in s.exclusion]
    lines += ["", f"Decision rule: {' '.join(s.decision_rule.split())}"]
    return "\n".join(lines)


def render_dimensions(codebook: Codebook) -> str:
    blocks = []
    for name, dim in codebook.dimensions.items():
        kind = "choose any that apply" if dim.multi_label else "choose at most one"
        lines = [f"Dimension: {name} ({kind})", f"Question: {dim.question}", "Labels:"]
        lines += [f"- {lab.id}: {lab.definition}" for lab in dim.labels]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def check_quotes(evidence: list[EvidenceSpan], article: Article) -> list[str]:
    """Warnings for quotes that do not appear in the article (the Verifier's quick check)."""
    haystack = " ".join(f"{article.title} {article.abstract}".split()).lower()
    return [
        f"quote not found in article: {e.quote[:80]!r}"
        for e in evidence
        if " ".join(e.quote.split()).lower() not in haystack
    ]
