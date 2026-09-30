"""Keyword-rule stand-in for an AI model.

Lets the whole pipeline run without an API key. Its answers follow the same
rules as the keyword baseline (scripts/score_baselines.py), so scores from a
fake run are a plumbing check, never a result.
"""

from __future__ import annotations

import re

from adrd_review.schemas import (
    AnnotationAnswer,
    DimensionAnnotation,
    EvidenceSpan,
    ExtractedField,
    ExtractionRecord,
    ScreeningAnswer,
    TriageReport,
)

INSTRUMENTS = {
    "MMSE": r"\bMMSE\b|mini[- ]mental state",
    "MoCA": r"\bMoCA\b|montreal cognitive assessment",
    "Mini-Cog": r"mini[- ]?cog\b",
    "IQCODE": r"\bIQCODE\b|informant questionnaire on cognitive decline",
}
SETTINGS = {
    "mci_cohort": r"mild cognitive impairment|\bMCI\b",
    "primary_care": r"primary care|general practi",
    "secondary_care": r"hospital|memory clinic|secondary care|psychogeriatric|geriatric (?:service|clinic)",
    "community": r"community|population[- ]based|general population",
}
DEMENTIA = r"dementia|alzheimer"
ACCURACY = r"sensitivity|specificity|accuracy|\bROC\b|\bAUC\b|cut[- ]?off|predictive value"

# Styles let the fake panel disagree, to exercise the consensus step:
#   keyword-rules    include if an instrument AND dementia are named (same as the keyword baseline)
#   keyword-strict   ... AND an accuracy term (sensitivity, AUC, cut-off, ...)
#   keyword-lenient  include if any instrument is named
STYLES = ("keyword-rules", "keyword-strict", "keyword-lenient")


def _article_text(user: str) -> str:
    """The article part of the user prompt (title + abstract)."""
    return user.split("ARTICLE", 1)[-1]


def _sentence_with(text: str, pattern: str) -> str | None:
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = re.sub(r"^(Title|Journal|Publication type|MeSH terms|PMID):\s*", "", sentence.strip())
        if re.search(pattern, sentence, re.I):
            return sentence[:300]
    return None


def default_responders(codebook=None, style: str = "keyword-rules") -> dict:
    if style not in STYLES:
        raise ValueError(f"Unknown fake model style {style!r}; use one of {STYLES}")
    dims = list(codebook.dimensions) if codebook is not None else ["instrument", "setting"]

    def triage(_schema, _system, user):
        has_abstract = "(no abstract available)" not in user
        return TriageReport(
            enough_information=has_abstract,
            missing=[] if has_abstract else ["I2", "I3", "I4"],
            note="" if has_abstract else "No abstract.",
        )

    def screening(_schema, _system, user):
        text = _article_text(user)
        found = [k for k, p in INSTRUMENTS.items() if re.search(p, text, re.I)]
        dementia = bool(re.search(DEMENTIA, text, re.I))
        include = bool(found) and dementia
        if style == "keyword-strict":
            include = include and bool(re.search(ACCURACY, text, re.I))
        elif style == "keyword-lenient":
            include = bool(found)
        quotes = [q for q in {_sentence_with(text, INSTRUMENTS[k]) for k in found} if q][:2]
        return ScreeningAnswer(
            decision="include" if include else "exclude",
            rule_ids=["I1", "I2"] if include else ["I1"],
            evidence=[EvidenceSpan(quote=q) for q in quotes],
            rationale="Keyword rule: names a listed instrument and dementia."
            if include
            else "Keyword rule: no listed instrument together with dementia.",
            confidence=0.7 if include else 0.6,
        )

    def extraction(_schema, _system, user):
        text = _article_text(user)
        return ExtractionRecord(
            instruments=[
                ExtractedField(value=k, quote=_sentence_with(text, p))
                for k, p in INSTRUMENTS.items()
                if re.search(p, text, re.I)
            ]
        )

    def annotation(_schema, _system, user):
        text = _article_text(user)
        tables = {"instrument": INSTRUMENTS, "setting": SETTINGS}
        out = []
        for dim in dims:
            table = tables.get(dim, {})
            labels = [k for k, p in table.items() if re.search(p, text, re.I)]
            out.append(
                DimensionAnnotation(
                    dimension=dim,
                    labels=labels,
                    evidence=[EvidenceSpan(quote=q) for q in [_sentence_with(text, table[k]) for k in labels] if q][:2],
                )
            )
        return AnnotationAnswer(annotations=out, confidence=0.6)

    return {
        TriageReport: triage,
        ScreeningAnswer: screening,
        ExtractionRecord: extraction,
        AnnotationAnswer: annotation,
    }
