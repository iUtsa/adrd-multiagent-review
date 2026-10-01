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
    if article.fulltext:
        lines += ["", "Full text (excerpt from PubMed Central):", article.fulltext]
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
    haystack = " ".join(f"{article.title} {article.abstract} {article.fulltext or ''}".split()).lower()
    return [
        f"quote not found in article: {e.quote[:80]!r}"
        for e in evidence
        if " ".join(e.quote.split()).lower() not in haystack
    ]


def allowed_labels(codebook: Codebook, field: str) -> list[str]:
    """Valid answers for a field: include/exclude for screening, else the dimension's label IDs."""
    if field == "screening":
        return ["include", "exclude"]
    return codebook.dimensions[field].label_ids


def render_field(codebook: Codebook, field: str) -> str:
    """The rules that govern one field, for agents that work on a single field."""
    if field == "screening":
        return render_screening_rules(codebook)
    dim = codebook.dimensions[field]
    kind = "any that apply" if dim.multi_label else "at most one"
    lines = [f"Dimension: {field} (choose {kind})", f"Question: {dim.question}", "Labels:"]
    lines += [f"- {lab.id}: {lab.definition}" for lab in dim.labels]
    return "\n".join(lines)


def show_labels(labels: list[str]) -> str:
    return ", ".join(labels) if labels else "(no label)"


def render_votes(votes: list, anonymous: bool = False) -> str:
    """Reviewers' answers with their evidence, for the adjudicator and verifier."""
    blocks = []
    for i, v in enumerate(sorted(votes, key=lambda v: v.reviewer_id), start=1):
        name = f"Reviewer {i}" if anonymous else f"Reviewer {v.reviewer_id}"
        lines = [f"{name}: {show_labels(v.labels)}"]
        if getattr(v, "rule_checks", None):
            lines.append("  Rule judgments: " + ", ".join(f"{r} {s}" for r, s in v.rule_checks.items()))
        elif v.rule_ids:
            lines.append(f"  Rules cited: {', '.join(v.rule_ids)}")
        for e in v.evidence:
            lines.append(f'  Evidence: "{e.quote}"')
        if v.rationale:
            lines.append(f"  Reasoning: {v.rationale}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_sides(votes: list) -> str:
    """Arguments grouped by position, WITHOUT saying how many reviewers hold each.

    Used in the debate so there is no visible majority to defer to. Quotes and
    reasons are de-duplicated and sorted, so repeated arguments do not reveal counts.
    """
    sides: dict[str, dict[str, set]] = {}
    for v in votes:
        side = sides.setdefault(show_labels(v.labels), {"quotes": set(), "reasons": set()})
        side["quotes"].update(e.quote for e in v.evidence)
        if v.rationale:
            side["reasons"].add(v.rationale)
    blocks = []
    for position in sorted(sides):
        s = sides[position]
        lines = [f"Position: {position}"]
        lines += [f'  Evidence cited: "{q}"' for q in sorted(s["quotes"])]
        lines += [f"  Argument: {r}" for r in sorted(s["reasons"])]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
