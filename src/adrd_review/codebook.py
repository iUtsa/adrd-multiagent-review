"""Load and validate the review codebook (configs/codebook.yaml)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_CODEBOOK = Path("configs/codebook.yaml")


class Rule(BaseModel):
    id: str
    name: str
    text: str
    source: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class Label(BaseModel):
    id: str
    definition: str = ""


class Dimension(BaseModel):
    multi_label: bool = True
    question: str = ""
    labels: list[Label]

    @property
    def label_ids(self) -> list[str]:
        return [label.id for label in self.labels]


class Screening(BaseModel):
    decision_rule: str = ""
    inclusion: list[Rule]
    exclusion: list[Rule] = Field(default_factory=list)


class Annotation(BaseModel):
    dimensions: dict[str, Dimension] = Field(default_factory=dict)


class Codebook(BaseModel):
    version: str
    status: str = ""
    question: str = ""
    sources: dict[str, str] = Field(default_factory=dict)
    screening: Screening
    annotation: Annotation = Field(default_factory=Annotation)
    sha256: str = ""  # of the YAML file, set by load_codebook

    @property
    def rules(self) -> list[Rule]:
        return self.screening.inclusion + self.screening.exclusion

    @property
    def dimensions(self) -> dict[str, Dimension]:
        return self.annotation.dimensions

    @model_validator(mode="after")
    def _check_ids(self) -> "Codebook":
        rule_ids = [r.id for r in self.rules]
        dupes = {i for i in rule_ids if rule_ids.count(i) > 1}
        if dupes:
            raise ValueError(f"Duplicate rule IDs: {sorted(dupes)}")
        for name, dim in self.dimensions.items():
            ids = dim.label_ids
            dupes = {i for i in ids if ids.count(i) > 1}
            if dupes:
                raise ValueError(f"Duplicate labels in dimension {name!r}: {sorted(dupes)}")
        unknown = {s for r in self.rules for s in r.source} - set(self.sources)
        if self.sources and unknown:
            raise ValueError(f"Rules cite unknown sources: {sorted(unknown)}")
        return self


def load_codebook(path: str | Path = DEFAULT_CODEBOOK) -> Codebook:
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw)
    data["sha256"] = hashlib.sha256(raw).hexdigest()
    return Codebook.model_validate(data)
