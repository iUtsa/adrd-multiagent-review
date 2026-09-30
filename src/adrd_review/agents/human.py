"""Human review: where escalated decisions get a person's answer.

Two ways to answer:
- GoldHuman: the gold-standard label stands in for the person. Used to score
  E4 on the calibration/test sets without new human work; the escalation count
  is the measure of human effort.
- QueueHuman: the pipeline really pauses (LangGraph `interrupt`) and waits. A person
  answers with `python -m adrd_review.human_review <run folder>`, and the run resumes.
"""

from __future__ import annotations

from typing import Protocol

from langgraph.types import interrupt

from adrd_review.gold import GoldRecord


class HumanReviewer(Protocol):
    mode: str

    def answer(self, item: dict) -> list[str]: ...


class GoldHuman:
    mode = "gold"

    def __init__(self, gold: dict[str, GoldRecord]):
        self.gold = gold

    def answer(self, item: dict) -> list[str]:
        g = self.gold.get(item["pmid"])
        if g is None:
            raise KeyError(f"No gold record for PMID {item['pmid']}")
        if item["field"] == "screening":
            return [g.screening]
        return list(g.labels.get(item["field"], [])) if g.screening == "include" else []


class QueueHuman:
    mode = "queue"

    def answer(self, item: dict) -> list[str]:
        value = interrupt(item)  # pauses the graph; the resume value comes back here
        if isinstance(value, str):
            value = [x.strip() for x in value.split(",") if x.strip()]
        return list(value)
