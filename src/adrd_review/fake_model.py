"""Keyword-rule stand-in for an AI model.

Lets the whole pipeline run without an API key. Its answers follow the same
rules as the keyword baseline (scripts/score_baselines.py), so scores from a
fake run are a plumbing check, never a result.

With the placeholder codebook it uses the built-in patterns below. With any codebook
whose rules carry `keywords:` (e.g. configs/codebook_aiml_draft.yaml) it follows those
instead, so a new topic can be run end to end without a key.
"""

from __future__ import annotations

import re

from adrd_review.schemas import (
    AdjudicationAnswer,
    AnnotationAnswer,
    CodebookProposal,
    CuratorAnswer,
    DebateAnswer,
    ReviewerWeight,
    VerifierAnswer,
    VerifierCheck,
    DimensionAnnotation,
    EvidenceSpan,
    ExtractedField,
    ExclusionCheck,
    ExtractionRecord,
    InclusionCheck,
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
    """Title + abstract (+ full-text excerpt) from the user prompt, without the metadata lines
    (journal, publication type, MeSH terms), which are not article text and cannot be quoted."""
    body = user.split("ARTICLE", 1)[-1]
    title = re.search(r"^Title: (.*)$", body, re.M)
    rest = body.split("\nAbstract:\n", 1)
    text = rest[1] if len(rest) == 2 else ""
    text = text.split("\n\nFACTS ALREADY EXTRACTED", 1)[0]
    return (title.group(1) if title else "") + "\n" + text


def _sentence_with(text: str, pattern: str) -> str | None:
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = re.sub(r"^(Title|Journal|Publication type|MeSH terms|PMID):\s*", "", sentence.strip())
        if re.search(pattern, sentence, re.I):
            return sentence[:300]
    return None


def _label_pattern(label) -> str:
    if label.keywords:
        return label.keywords
    return r"\b" + re.escape(label.id).replace("_", "[ _-]") + r"\b"


def default_responders(codebook=None, style: str = "keyword-rules") -> dict:
    if style not in STYLES:
        raise ValueError(f"Unknown fake model style {style!r}; use one of {STYLES}")
    dims = list(codebook.dimensions) if codebook is not None else ["instrument", "setting"]
    generic = codebook is not None and any(r.keywords for r in codebook.screening.inclusion)
    topic = _generic_responders(codebook, style) if generic else _builtin_responders(dims, style)
    return topic | _shared_responders()


def _generic_responders(codebook, style: str) -> dict:
    """Keyword rules taken from the codebook (`keywords:` on rules and labels).

    Strictness by style: keyword-strict treats every missing inclusion keyword as not met,
    keyword-rules only the first two (the topic rules), keyword-lenient only the first.
    At the title stage only the first rule can be not met (a title says little).
    """
    inc = codebook.screening.inclusion
    hard = {"keyword-strict": len(inc), "keyword-rules": 2, "keyword-lenient": 1}[style]

    def screening(_schema, system, user):
        text = _article_text(user)
        m = re.search(r"^Title: (.*)$", user, re.M)
        title = m.group(1).strip() if m else ""
        titles_only = "TITLES ONLY" in system
        full_paper = "FULL PAPER" in system
        checks = []
        for i, rule in enumerate(inc):
            quote = _sentence_with(text, rule.keywords) if rule.keywords else None
            if quote:
                checks.append(InclusionCheck(rule_id=rule.id, status="met", quote=quote))
            elif rule.keywords and not full_paper and i < (min(hard, 1) if titles_only else hard) \
                    and not (titles_only and style == "keyword-lenient"):
                checks.append(InclusionCheck(rule_id=rule.id, status="not_met", quote=title))
            else:
                checks.append(InclusionCheck(rule_id=rule.id, status="unclear"))
        excl = []
        for rule in codebook.screening.exclusion:
            hit = rule.keywords and re.search(rule.keywords, title, re.I)
            excl.append(ExclusionCheck(rule_id=rule.id, status="applies", quote=title) if hit
                        else ExclusionCheck(rule_id=rule.id, status="unclear"))
        return ScreeningAnswer(inclusion=checks, exclusion=excl, rationale=f"Keyword stand-in ({style}).", confidence=0.7)

    def triage(_schema, _system, user):
        text = _article_text(user)
        has_abstract = "(no abstract available)" not in user
        first = inc[0]
        off_topic = has_abstract and bool(first.keywords) and not re.search(first.keywords, text, re.I)
        missing = [r.id for r in inc if not (r.keywords and re.search(r.keywords, text, re.I))]
        return TriageReport(clearly_ineligible=off_topic, missing=[] if off_topic else missing, note="")

    def annotation(_schema, _system, user):
        text = _article_text(user)
        out = []
        for name, dim in codebook.dimensions.items():
            found = [lab for lab in dim.labels if re.search(_label_pattern(lab), text, re.I)]
            if not dim.multi_label:
                found = found[:1]
            quotes = [q for q in (_sentence_with(text, _label_pattern(lab)) for lab in found) if q]
            out.append(DimensionAnnotation(dimension=name, labels=[lab.id for lab in found],
                                           evidence=[EvidenceSpan(quote=q) for q in quotes][:2]))
        return AnnotationAnswer(annotations=out, confidence=0.6)

    return {TriageReport: triage, ScreeningAnswer: screening, AnnotationAnswer: annotation}


def _builtin_responders(dims: list[str], style: str) -> dict:
    """The placeholder topic (cognitive screening tests), with built-in patterns."""

    def triage(_schema, _system, user):
        text = _article_text(user)
        has_abstract = "(no abstract available)" not in user
        off_topic = has_abstract and not any(re.search(p, text, re.I) for p in INSTRUMENTS.values())
        return TriageReport(
            clearly_ineligible=off_topic,
            missing=[] if has_abstract else ["I2", "I3", "I4"],
            note="" if has_abstract else "No abstract.",
        )

    def screening(_schema, _system, user):
        if "TITLES ONLY" in _system:
            # Title stage: only a title naming no instrument and no dementia term is off topic.
            m = re.search(r"^Title: (.*)$", user, re.M)
            title = m.group(1).strip() if m else ""
            topical = any(re.search(p, title, re.I) for p in [*INSTRUMENTS.values(), DEMENTIA, r"cognit|screen|memory"])
            off = not topical and style != "keyword-lenient"
            return ScreeningAnswer(
                inclusion=[InclusionCheck(rule_id="I1", status="not_met" if off else "unclear", quote=title if off else None),
                           *[InclusionCheck(rule_id=r, status="unclear") for r in ("I2", "I3", "I4")]],
                exclusion=[ExclusionCheck(rule_id=r, status="unclear") for r in ("E1", "E2", "E3")],
                rationale=f"Keyword stand-in ({style}), title only.",
                confidence=0.7,
            )
        # Keyword rules expressed as rule judgments. The decision is made by screening.decide():
        #   keyword-rules    I1 needs an instrument, I2 needs dementia
        #   keyword-strict   ... and I3 needs an accuracy term
        #   keyword-lenient  only I1 is judged
        text = _article_text(user)
        m = re.search(r"^Title: (.*)$", user, re.M)
        title = m.group(1).strip() if m else ""
        found = [k for k, p in INSTRUMENTS.items() if re.search(p, text, re.I)]
        inst_quote = next((q for q in (_sentence_with(text, INSTRUMENTS[k]) for k in found) if q), None)
        checks = [InclusionCheck(rule_id="I1", status="met" if found else "not_met", quote=inst_quote if found else title)]
        if style != "keyword-lenient":
            dem = _sentence_with(text, DEMENTIA)
            checks.append(InclusionCheck(rule_id="I2", status="met" if dem else "not_met", quote=dem or title))
        else:
            checks.append(InclusionCheck(rule_id="I2", status="unclear"))
        full_paper = "FULL PAPER" in _system  # stage 2: the eligibility prompt
        acc = _sentence_with(text, ACCURACY)
        if style == "keyword-strict":
            checks.append(InclusionCheck(rule_id="I3", status="met" if acc else "not_met", quote=acc or title))
        elif full_paper and acc:
            checks.append(InclusionCheck(rule_id="I3", status="met", quote=acc))
        else:
            checks.append(InclusionCheck(rule_id="I3", status="unclear"))
        setting = next((q for q in (_sentence_with(text, p) for p in SETTINGS.values()) if q), None)
        if full_paper and setting:
            checks.append(InclusionCheck(rule_id="I4", status="met", quote=setting))
        else:
            checks.append(InclusionCheck(rule_id="I4", status="unclear"))
        return ScreeningAnswer(
            inclusion=checks,
            exclusion=[ExclusionCheck(rule_id=r, status="unclear") for r in ("E1", "E2", "E3")],
            rationale=f"Keyword stand-in ({style}).",
            confidence=0.7,
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

    return {TriageReport: triage, ScreeningAnswer: screening, AnnotationAnswer: annotation}


def _shared_responders() -> dict:
    """Agents whose stand-in does not depend on the topic."""

    def extraction(_schema, _system, user):
        text = _article_text(user)
        return ExtractionRecord(
            instruments=[
                ExtractedField(value=k, quote=_sentence_with(text, p))
                for k, p in INSTRUMENTS.items()
                if re.search(p, text, re.I)
            ]
        )

    def adjudication(_schema, _system, user):
        # Weigh each reviewer by how much evidence it cited; answer = weighted majority.
        blocks = re.findall(r"^Reviewer (\w+): (.*)$((?:\n  .*)*)", user, re.M)
        weights, support = [], {}
        for rid, labels, body in blocks:
            w = min(0.9, 0.4 + 0.2 * body.count("Evidence:"))
            weights.append(ReviewerWeight(reviewer=rid, weight=w, reason="Weighted by evidence cited."))
            key = "" if labels == "(no label)" else labels
            support[key] = support.get(key, 0.0) + w
        total = sum(support.values()) or 1.0
        best = max(sorted(support), key=lambda k: support[k]) if support else ""
        labels = [x.strip() for x in best.split(",") if x.strip()]
        return AdjudicationAnswer(
            labels=labels,
            reviewer_weights=weights,
            rule_ids=[],
            rationale="Keyword stand-in: the position backed by more cited evidence.",
            confidence=round(support.get(best, 0) / total, 3),
        )

    def debate(_schema, _system, user):
        prev = re.search(r"YOUR PREVIOUS ANSWER: (.*)", user)
        labels = [] if not prev or prev.group(1) == "(no label)" else [x.strip() for x in prev.group(1).split(",")]
        return DebateAnswer(labels=labels, rationale="Keyword stand-in keeps its position.", confidence=0.6)

    def verification(_schema, _system, user):
        checks = []
        for field, labels, body in re.findall(r"^Decision: (\w+) = (.*)$((?:\n  .*)*)", user, re.M):
            ok = labels != "include" or "Evidence:" in body
            checks.append(VerifierCheck(field=field, supported=ok, problem="" if ok else "Include with no evidence."))
        return VerifierAnswer(checks=checks)

    def curation(_schema, _system, user):
        rules = re.findall(r"Rules the agents cited: (.*)", user)
        pmids = list(dict.fromkeys(re.findall(r"^PMID (\d+):", user, re.M)))
        counts: dict[str, int] = {}
        for line in rules:
            for r in line.split(", "):
                counts[r] = counts.get(r, 0) + 1
        top = sorted(counts, key=lambda r: (-counts[r], r))[:3] or ["new"]
        return CuratorAnswer(
            proposals=[
                CodebookProposal(
                    target=r,
                    problem=f"Keyword stand-in: rule {r} was cited in {counts.get(r, 0)} disagreements.",
                    suggested_text="(A real model proposes wording here.)",
                    example_pmids=pmids[:3],
                )
                for r in top
            ]
        )

    return {
        ExtractionRecord: extraction,
        AdjudicationAnswer: adjudication,
        DebateAnswer: debate,
        VerifierAnswer: verification,
        CuratorAnswer: curation,
    }
