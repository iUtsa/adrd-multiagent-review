"""E1 pipeline: one reviewer agent, four steps.

    START -> triage -> screen --include--> extract -> annotate -> END
                            \\--exclude--> END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from adrd_review.agents import annotation, extraction, screening, triage
from adrd_review.codebook import Codebook
from adrd_review.graph.state import ArticleState
from adrd_review.llm import LLM
from adrd_review.schemas import Decision

ROLES = ("triage", "screening", "extraction", "annotation")


def build_pipeline(llms: dict[str, LLM], codebook: Codebook, personas: dict[str, str] | None = None):
    missing = [r for r in ROLES if r not in llms]
    if missing:
        raise ValueError(f"No model configured for: {missing}")
    personas = personas or {}

    def triage_node(state: ArticleState) -> dict:
        report, usage = triage.run(state["article"], codebook, llms["triage"])
        return {"triage": report, "usage": [usage]}

    def screen_node(state: ArticleState) -> dict:
        vote, usage = screening.run(
            state["article"], codebook, llms["screening"], reviewer_id="A", persona=personas.get("screening")
        )
        decision = Decision(field="screening", labels=vote.labels, method="single", confidence=vote.confidence)
        return {"screening_votes": [vote], "screening_decision": decision, "usage": [usage]}

    def after_screening(state: ArticleState) -> str:
        return "extract" if state["screening_decision"].labels == ["include"] else END

    def extract_node(state: ArticleState) -> dict:
        record, usage = extraction.run(state["article"], llms["extraction"])
        return {"extraction": record, "usage": [usage]}

    def annotate_node(state: ArticleState) -> dict:
        votes, usage = annotation.run(
            state["article"],
            codebook,
            llms["annotation"],
            extraction=state.get("extraction"),
            reviewer_id="A",
            persona=personas.get("annotation"),
        )
        decisions = {
            v.field: Decision(field=v.field, labels=v.labels, method="single", confidence=v.confidence)
            for v in votes
        }
        return {"annotation_votes": votes, "annotation_decisions": decisions, "usage": [usage]}

    g = StateGraph(ArticleState)
    g.add_node("triage", triage_node)
    g.add_node("screen", screen_node)
    g.add_node("extract", extract_node)
    g.add_node("annotate", annotate_node)
    g.add_edge(START, "triage")
    g.add_edge("triage", "screen")
    g.add_conditional_edges("screen", after_screening, ["extract", END])
    g.add_edge("extract", "annotate")
    g.add_edge("annotate", END)
    return g.compile()
