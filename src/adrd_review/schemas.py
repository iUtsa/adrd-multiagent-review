"""Shared data records used across tools, agents and evaluation."""

from __future__ import annotations

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
