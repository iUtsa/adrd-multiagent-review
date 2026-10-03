"""Extraction agent: key study facts, each with the exact words it came from."""

from __future__ import annotations

from adrd_review.agents.common import EVIDENCE_RULES, render_article
from adrd_review.llm import LLM
from adrd_review.schemas import Article, ExtractionRecord, Usage

SYSTEM = """\
You are the Extraction agent in a scoping review of {topic}.
Fill in each field from the article text only. For every value, give the exact words it comes from.
Use null for anything the text does not report.

{evidence_rules}"""


def run(article: Article, llm: LLM, topic: str = "dementia screening research") -> tuple[ExtractionRecord, Usage]:
    result = llm.structured(
        ExtractionRecord, SYSTEM.format(topic=topic, evidence_rules=EVIDENCE_RULES), render_article(article), role="extraction"
    )
    return result.parsed, result.usage
