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
    enough_information: bool = Field(
        description="True if the title and abstract give enough information to apply every inclusion rule."
    )
    missing: list[str] = Field(
        default_factory=list,
        description="IDs of inclusion rules that cannot be judged from the title and abstract, e.g. ['I2'].",
    )
    note: str = Field(default="", description="One sentence on what is missing, if anything.")


class ScreeningAnswer(BaseModel):
    decision: Literal["include", "exclude"] = Field(description="Include or exclude, following the decision rule.")
    rule_ids: list[str] = Field(
        description="IDs of the rules this decision rests on, e.g. ['I1', 'I2'] to include or ['E1'] / ['I3'] to exclude."
    )
    evidence: list[EvidenceSpan] = Field(
        default_factory=list, description="1-3 exact quotes that support the decision."
    )
    rationale: str = Field(description="At most three sentences explaining the decision with reference to the rules.")
    confidence: float = Field(ge=0, le=1, description="How sure you are, from 0 to 1.")


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


class Decision(BaseModel):
    """The final answer for one field, and how it was reached."""

    field: str
    labels: list[str]
    method: str  # "single", "unanimous", "majority", "adjudicator", "human"
    confidence: float | None = None


class Usage(BaseModel):
    """Cost and time of one model call."""

    role: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    seconds: float = 0.0
    cached: bool = False
