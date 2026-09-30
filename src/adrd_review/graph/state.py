"""The shared record every node in the graph reads and writes (one per article)."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from adrd_review.schemas import (
    Adjudication,
    Article,
    Decision,
    Escalation,
    ExtractionRecord,
    TriageReport,
    Usage,
    VerificationReport,
    Vote,
)


class ArticleState(TypedDict, total=False):
    article: Article
    triage: TriageReport
    fulltext_info: dict  # whether PubMed Central full text was fetched and used
    # Lists marked operator.add are appended to, never overwritten, so parallel
    # reviewers can each add their vote and debate rounds keep earlier votes.
    screening_votes: Annotated[list[Vote], operator.add]
    screening_decision: Decision
    screening_round: int  # debate rounds done on screening (0 = independent votes only)
    extraction: ExtractionRecord
    annotation_votes: Annotated[list[Vote], operator.add]
    annotation_decisions: dict[str, Decision]
    annotation_round: int
    adjudications: Annotated[list[Adjudication], operator.add]
    escalations: Annotated[list[Escalation], operator.add]
    verification: VerificationReport
    human_answers: Annotated[list[dict], operator.add]
    usage: Annotated[list[Usage], operator.add]
