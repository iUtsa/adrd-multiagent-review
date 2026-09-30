"""The shared record every node in the graph reads and writes (one per article)."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from adrd_review.schemas import Article, Decision, ExtractionRecord, TriageReport, Usage, Vote


class ArticleState(TypedDict, total=False):
    article: Article
    triage: TriageReport
    # Lists marked operator.add are collected, never overwritten, so parallel
    # reviewers (Day 5) can each add their vote.
    screening_votes: Annotated[list[Vote], operator.add]
    screening_decision: Decision
    extraction: ExtractionRecord
    annotation_votes: Annotated[list[Vote], operator.add]
    annotation_decisions: dict[str, Decision]
    usage: Annotated[list[Usage], operator.add]
