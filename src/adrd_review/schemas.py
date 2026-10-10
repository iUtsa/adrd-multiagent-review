"""Shared data records used across tools, agents and evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AbstractSection(BaseModel):
    """One labelled part of a structured abstract (e.g. METHODS)."""

    label: str | None = None
    text: str


class Article(BaseModel):
    """One PubMed article, as every agent will read it."""

    pmid: str
    title: str
    abstract: str = ""
    abstract_sections: list[AbstractSection] = Field(default_factory=list)
    journal: str | None = None
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    pmcid: str | None = None
    language: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    source: str = "pubmed"
    fetched_at: str | None = None
    fulltext: str | None = None  # PubMed Central excerpt, attached by the pipeline when used

    @property
    def has_abstract(self) -> bool:
        return bool(self.abstract.strip())


# --------------------------------------------------------------------------- #
# What the AI models must return (structured output).
# Field descriptions are sent to the model as instructions.
# --------------------------------------------------------------------------- #


class EvidenceSpan(BaseModel):
    quote: str = Field(description="Exact words copied from the title or abstract. Never paraphrase.")
    source: Literal["title", "abstract", "fulltext"] = Field(
        default="abstract", description="Where the quote comes from."
    )


class TriageReport(BaseModel):
    clearly_ineligible: bool = Field(
        default=False,
        description="True if the text already shows the study cannot qualify: an inclusion rule is clearly "
        "not met (for example the paper is about another topic) or an exclusion rule clearly applies.",
    )
    missing: list[str] = Field(
        default_factory=list,
        description="IDs of inclusion rules that cannot be judged EITHER WAY from the title and abstract, "
        "e.g. ['I2']. Leave empty when clearly_ineligible is true.",
    )
    note: str = Field(default="", description="One sentence on what is missing, if anything.")
    # Set by the code, not the model: the article can be judged from what we have.
    enough_information: bool = Field(default=True, description="Leave as is; computed from the fields above.")


class InclusionCheck(BaseModel):
    rule_id: str = Field(description="Inclusion rule ID exactly as given, e.g. 'I2'.")
    status: Literal["met", "not_met", "unclear"] = Field(
        description="met = the text shows the rule is satisfied. not_met = the text CLEARLY shows it is not "
        "satisfied. unclear = the text does not say enough to tell. Missing information is ALWAYS unclear, never not_met."
    )
    quote: str | None = Field(
        default=None,
        description="Exact words from the article that show this. Required for met and not_met; null for unclear.",
    )


class ExclusionCheck(BaseModel):
    rule_id: str = Field(description="Exclusion rule ID exactly as given, e.g. 'E1'.")
    status: Literal["applies", "does_not_apply", "unclear"] = Field(
        description="applies = the text CLEARLY shows this exclusion applies. does_not_apply = the text shows it "
        "does not. unclear = the text does not say enough to tell."
    )
    quote: str | None = Field(default=None, description="Exact words from the article; required for applies.")


class ScreeningAnswer(BaseModel):
    """The reviewer judges each rule; the code, not the model, turns that into include/exclude."""

    inclusion: list[InclusionCheck] = Field(description="One entry per inclusion rule, in the order given.")
    exclusion: list[ExclusionCheck] = Field(default_factory=list, description="One entry per exclusion rule.")
    rationale: str = Field(description="At most three sentences on the rules that matter most for this article.")
    confidence: float = Field(ge=0, le=1, description="How sure you are of your rule judgments overall, 0 to 1.")


class ExtractedField(BaseModel):
    value: str | None = Field(default=None, description="Short answer, or null if the text does not say.")
    quote: str | None = Field(default=None, description="Exact words the value comes from, or null.")


class ExtractionRecord(BaseModel):
    instruments: list[ExtractedField] = Field(
        default_factory=list, description="Every cognitive screening instrument or test the study evaluates."
    )
    target_condition: ExtractedField = Field(default_factory=ExtractedField, description="The condition being detected, e.g. all-cause dementia.")
    reference_standard: ExtractedField = Field(default_factory=ExtractedField, description="How dementia was confirmed, e.g. DSM-IV clinical diagnosis.")
    study_design: ExtractedField = Field(default_factory=ExtractedField, description="E.g. cross-sectional, longitudinal, case-control.")
    setting: ExtractedField = Field(default_factory=ExtractedField, description="Where participants came from, e.g. community, primary care, memory clinic.")
    population: ExtractedField = Field(default_factory=ExtractedField, description="Who was studied, e.g. adults 65+, people with MCI.")
    sample_size: ExtractedField = Field(default_factory=ExtractedField, description="Number of participants.")
    country: ExtractedField = Field(default_factory=ExtractedField, description="Country where the study took place.")


class DimensionAnnotation(BaseModel):
    dimension: str = Field(description="The dimension name exactly as given.")
    labels: list[str] = Field(
        default_factory=list, description="Label IDs exactly as listed for this dimension. Empty if none apply."
    )
    evidence: list[EvidenceSpan] = Field(default_factory=list, description="Exact quotes supporting the labels.")
    rationale: str = Field(default="", description="One or two sentences.")


class AnnotationAnswer(BaseModel):
    annotations: list[DimensionAnnotation] = Field(description="One entry per dimension.")
    confidence: float = Field(ge=0, le=1, description="How sure you are overall, from 0 to 1.")


# --------------------------------------------------------------------------- #
# What the pipeline stores (shared by every stage and, later, the consensus step)
# --------------------------------------------------------------------------- #


class Vote(BaseModel):
    """One reviewer's answer on one field ('screening' or an annotation dimension)."""

    reviewer_id: str
    model: str
    field: str
    labels: list[str]
    round: int = 0  # 0 = independent round; >0 only in the debate variant
    rule_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    rationale: str = ""
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)
    rule_checks: dict[str, str] = Field(default_factory=dict)  # screening: rule ID -> met/not_met/unclear/applies/...


class Decision(BaseModel):
    """The final answer for one field, and how it was reached."""

    field: str
    labels: list[str]
    method: str  # "single", "unanimous", "majority", "tie_include", "adjudicator", "human"
    confidence: float | None = None  # share of reviewer weight behind the chosen labels
    include_score: float | None = None  # screening only: 0-1 likelihood of include (ranks articles for WSS@95)


class Usage(BaseModel):
    """Cost and time of one model call."""

    role: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    seconds: float = 0.0
    cached: bool = False  # answered from our response cache (no call made)
    prompt_cache_read_tokens: int = 0  # input tokens billed at the provider's cheaper cache rate


# --------------------------------------------------------------------------- #
# Day 6-8 agents: answer forms
# --------------------------------------------------------------------------- #


class ReviewerWeight(BaseModel):
    reviewer: str = Field(description="Reviewer ID exactly as given, e.g. 'A'.")
    weight: float = Field(ge=0, le=1, description="How well the text and rules support this reviewer's position, 0-1.")
    reason: str = Field(default="", description="One sentence.")


class AdjudicationAnswer(BaseModel):
    labels: list[str] = Field(
        description="Your answer: for screening ['include'] or ['exclude']; for a dimension, the label IDs (empty if none)."
    )
    reviewer_weights: list[ReviewerWeight] = Field(description="One entry per reviewer.")
    rule_ids: list[str] = Field(default_factory=list, description="Codebook rule IDs your answer rests on.")
    rationale: str = Field(description="At most four sentences: which evidence settles the disagreement and why.")
    confidence: float = Field(ge=0, le=1, description="How sure you are, 0-1. Be honest: low when the text is ambiguous.")


class DebateAnswer(BaseModel):
    labels: list[str] = Field(
        description="Your answer after reading the arguments: ['include']/['exclude'] for screening, or label IDs."
    )
    new_evidence: list[EvidenceSpan] = Field(
        default_factory=list,
        description="Quotes you did NOT cite before. Required if you change your answer; a change without new evidence is not accepted.",
    )
    rule_ids: list[str] = Field(default_factory=list, description="Codebook rule IDs your answer rests on.")
    rationale: str = Field(description="At most three sentences, answering the strongest opposing argument.")
    confidence: float = Field(ge=0, le=1, description="How sure you are, 0-1.")


class VerifierCheck(BaseModel):
    field: str = Field(description="'screening' or the dimension name, exactly as given.")
    supported: bool = Field(description="True if the quoted evidence and cited rules actually support this decision.")
    problem: str = Field(default="", description="If not supported: one sentence on what is wrong.")


class VerifierAnswer(BaseModel):
    checks: list[VerifierCheck] = Field(description="One check per decision listed.")


class CodebookProposal(BaseModel):
    target: str = Field(description="Rule ID (e.g. 'I4') or 'dimension:label' (e.g. 'setting:primary_care') or 'new'.")
    problem: str = Field(description="What in the current wording caused disagreements, with reference to the examples.")
    suggested_text: str = Field(description="Proposed clearer wording or a borderline example to add.")
    example_pmids: list[str] = Field(default_factory=list, description="PMIDs of the disagreements this addresses.")


class CuratorAnswer(BaseModel):
    proposals: list[CodebookProposal] = Field(description="3-8 proposals, most useful first.")


# --------------------------------------------------------------------------- #
# Day 6-8 pipeline records
# --------------------------------------------------------------------------- #

EscalationReason = Literal[
    "split_exclusion", "low_confidence", "verifier_fail", "no_fulltext", "unclear_fulltext"
]


class Escalation(BaseModel):
    """A decision sent to a human."""

    field: str
    reason: EscalationReason
    detail: str = ""


class Adjudication(BaseModel):
    field: str
    model: str
    labels: list[str]  # the adjudicator's own answer
    reviewer_weights: dict[str, float]
    rule_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float
    warnings: list[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    quote_problems: list[str] = Field(default_factory=list)  # deterministic check
    checks: list[VerifierCheck] = Field(default_factory=list)  # model check (empty if no verifier model)
    model: str | None = None

    @property
    def failed_fields(self) -> list[str]:
        return [c.field for c in self.checks if not c.supported]
