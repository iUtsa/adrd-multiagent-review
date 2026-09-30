"""The gold standard: human decisions every agent run is scored against.

Standard file format (CSV, one row per article), documented in docs/GOLD_STANDARD.md:

    pmid        required  PubMed ID
    screening   required  include | exclude
    fulltext    optional  include | exclude (final decision after full-text reading)
    <dimension> optional  one column per codebook annotation dimension; several labels
                          separated by ';'; empty = no label
    <field>__<reviewer>   optional per-reviewer answers, e.g. screening__r1, instrument__r2
    score       optional  0-1 likelihood of "include" (agent predictions; used for WSS@95)
    source      optional  free text: where the row came from

Agent predictions use this same format, so gold and predictions load the same way.

Any source (the professor's file, the CLEF placeholder) is converted to this format once,
so nothing downstream depends on where the gold standard came from.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from adrd_review.codebook import Codebook

Decision = Literal["include", "exclude"]
_DECISIONS = {"include": "include", "included": "include", "exclude": "exclude", "excluded": "exclude"}
SEP = ";"


class GoldError(ValueError):
    """The gold file has problems; `problems` lists every one found."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        shown = "\n  - ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        super().__init__(f"{len(problems)} problem(s) in gold file:\n  - {shown}{more}")


class GoldRecord(BaseModel):
    pmid: str
    screening: Decision
    fulltext: Decision | None = None
    labels: dict[str, list[str]] = Field(default_factory=dict)
    reviewers: dict[str, dict[str, str]] = Field(default_factory=dict)
    score: float | None = None
    source: str | None = None


def _decision(value: str) -> str | None:
    return _DECISIONS.get(value.strip().lower())


def _split_labels(cell: str) -> list[str]:
    return [x.strip() for x in cell.split(SEP) if x.strip()]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_gold(path: str | Path, codebook: Codebook) -> list[GoldRecord]:
    """Read and validate a gold file. Raises GoldError listing every problem."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    problems: list[str] = []

    for col in ("pmid", "screening"):
        if col not in df.columns:
            problems.append(f"missing required column {col!r}")
    if problems:
        raise GoldError(problems)

    dims = codebook.dimensions
    dim_cols = [d for d in dims if d in df.columns]
    reviewer_cols = [c for c in df.columns if "__" in c]

    records: list[GoldRecord] = []
    seen: Counter = Counter()
    for i, row in df.iterrows():
        line = i + 2  # header is line 1
        pmid = row["pmid"].strip().removesuffix(".0")
        if not pmid.isdigit():
            problems.append(f"line {line}: PMID {row['pmid']!r} is not a number")
            continue
        seen[pmid] += 1

        screening = _decision(row["screening"])
        if screening is None:
            problems.append(f"line {line} (PMID {pmid}): screening {row['screening']!r} is not include/exclude")
            continue

        fulltext = None
        if "fulltext" in df.columns and row["fulltext"].strip():
            fulltext = _decision(row["fulltext"])
            if fulltext is None:
                problems.append(f"line {line} (PMID {pmid}): fulltext {row['fulltext']!r} is not include/exclude")

        labels: dict[str, list[str]] = {}
        for dim in dim_cols:
            values = _split_labels(row[dim])
            unknown = [v for v in values if v not in dims[dim].label_ids]
            if unknown:
                problems.append(f"line {line} (PMID {pmid}): {dim} labels not in codebook: {unknown}")
            if len(values) > 1 and not dims[dim].multi_label:
                problems.append(f"line {line} (PMID {pmid}): {dim} allows one label, got {values}")
            if values:
                labels[dim] = values
        if labels and screening == "exclude":
            problems.append(f"line {line} (PMID {pmid}): excluded article has annotation labels")

        reviewers: dict[str, dict[str, str]] = {}
        for col in reviewer_cols:
            field, reviewer = col.split("__", 1)
            if row[col].strip():
                reviewers.setdefault(reviewer, {})[field] = row[col].strip()

        score = None
        if "score" in df.columns and row["score"].strip():
            try:
                score = float(row["score"])
            except ValueError:
                problems.append(f"line {line} (PMID {pmid}): score {row['score']!r} is not a number")
            else:
                if not 0 <= score <= 1:
                    problems.append(f"line {line} (PMID {pmid}): score {score} is outside 0-1")

        records.append(
            GoldRecord(
                pmid=pmid,
                screening=screening,
                fulltext=fulltext,
                labels=labels,
                reviewers=reviewers,
                score=score,
                source=row["source"].strip() or None if "source" in df.columns else None,
            )
        )

    dupes = sorted(p for p, n in seen.items() if n > 1)
    if dupes:
        problems.append(f"duplicate PMIDs: {dupes[:10]}")
    if problems:
        raise GoldError(problems)
    return records


def write_gold(records: list[GoldRecord], path: str | Path, dimensions: list[str]) -> None:
    """Write records in the standard format."""
    rows = []
    for r in records:
        row = {"pmid": r.pmid, "screening": r.screening, "fulltext": r.fulltext or ""}
        for dim in dimensions:
            row[dim] = SEP.join(r.labels.get(dim, []))
        if any(x.score is not None for x in records):
            row["score"] = "" if r.score is None else r.score
        row["source"] = r.source or ""
        rows.append(row)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize(records: list[GoldRecord]) -> dict:
    """Counts for a quick look at a gold file."""
    out: dict = {
        "articles": len(records),
        "screening": dict(Counter(r.screening for r in records)),
    }
    if any(r.fulltext for r in records):
        out["fulltext"] = dict(Counter(r.fulltext for r in records if r.fulltext))
    dims = sorted({d for r in records for d in r.labels})
    for dim in dims:
        out[dim] = dict(Counter(v for r in records for v in r.labels.get(dim, [])).most_common())
    out["reviewers"] = sorted({rv for r in records for rv in r.reviewers})
    return out
