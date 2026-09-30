"""The review pipeline as a LangGraph graph. One graph serves every condition:
E1 is a panel of one reviewer, E2 a panel of three (or more).

    START -> triage -> [screen_reviewer x N, in parallel] -> screen_consensus
          --include--> extract -> [annotate_reviewer x N, in parallel] -> annotate_consensus -> END
          --exclude--> END

Parallel reviewers use LangGraph's Send API. Each reviewer sees only the article
and the codebook, never the other reviewers' votes (independent round).
Votes are collected by the `operator.add` reducers in ArticleState.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from adrd_review.agents import annotation, extraction, screening, triage
from adrd_review.codebook import Codebook
from adrd_review.consensus import combine_labels, combine_screening
from adrd_review.graph.state import ArticleState
from adrd_review.llm import LLM


@dataclass
class Reviewer:
    id: str
    llm: LLM  # screening
    annotation_llm: LLM
    persona: str | None = None
    weight: float = 1.0


def build_pipeline(codebook: Codebook, triage_llm: LLM, extraction_llm: LLM, reviewers: list[Reviewer]):
    if not reviewers:
        raise ValueError("At least one reviewer is needed")
    ids = [r.id for r in reviewers]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Reviewer IDs must be unique: {ids}")
    by_id = {r.id: r for r in reviewers}
    weights = {r.id: r.weight for r in reviewers}

    def triage_node(state: ArticleState) -> dict:
        report, usage = triage.run(state["article"], codebook, triage_llm)
        return {"triage": report, "usage": [usage]}

    def fan_out_screening(state: ArticleState) -> list[Send]:
        return [Send("screen_reviewer", {"article": state["article"], "reviewer_id": r.id}) for r in reviewers]

    def screen_reviewer(payload: dict) -> dict:
        r = by_id[payload["reviewer_id"]]
        vote, usage = screening.run(payload["article"], codebook, r.llm, reviewer_id=r.id, persona=r.persona)
        return {"screening_votes": [vote], "usage": [usage]}

    def screen_consensus(state: ArticleState) -> dict:
        return {"screening_decision": combine_screening(state["screening_votes"], weights)}

    def after_screening(state: ArticleState) -> str:
        return "extract" if state["screening_decision"].labels == ["include"] else END

    def extract_node(state: ArticleState) -> dict:
        record, usage = extraction.run(state["article"], extraction_llm)
        return {"extraction": record, "usage": [usage]}

    def fan_out_annotation(state: ArticleState) -> list[Send]:
        return [
            Send(
                "annotate_reviewer",
                {"article": state["article"], "extraction": state.get("extraction"), "reviewer_id": r.id},
            )
            for r in reviewers
        ]

    def annotate_reviewer(payload: dict) -> dict:
        r = by_id[payload["reviewer_id"]]
        votes, usage = annotation.run(
            payload["article"],
            codebook,
            r.annotation_llm,
            extraction=payload.get("extraction"),
            reviewer_id=r.id,
            persona=r.persona,
        )
        return {"annotation_votes": votes, "usage": [usage]}

    def annotate_consensus(state: ArticleState) -> dict:
        by_field = defaultdict(list)
        for v in state["annotation_votes"]:
            by_field[v.field].append(v)
        decisions = {
            name: combine_labels(by_field[name], name, dim.multi_label, dim.label_ids, weights)
            for name, dim in codebook.dimensions.items()
            if by_field[name]
        }
        return {"annotation_decisions": decisions}

    g = StateGraph(ArticleState)
    g.add_node("triage", triage_node)
    g.add_node("screen_reviewer", screen_reviewer)
    g.add_node("screen_consensus", screen_consensus)
    g.add_node("extract", extract_node)
    g.add_node("annotate_reviewer", annotate_reviewer)
    g.add_node("annotate_consensus", annotate_consensus)

    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", fan_out_screening, ["screen_reviewer"])
    g.add_edge("screen_reviewer", "screen_consensus")
    g.add_conditional_edges("screen_consensus", after_screening, ["extract", END])
    g.add_conditional_edges("extract", fan_out_annotation, ["annotate_reviewer"])
    g.add_edge("annotate_reviewer", "annotate_consensus")
    g.add_edge("annotate_consensus", END)
    return g.compile()
