"""The review pipeline as a LangGraph graph. One graph serves every condition;
PipelineOptions switch steps on:

    E1  1 reviewer
    E2  3 reviewers + weighted majority vote
    E3  + adjudicator on split fields, + verifier
    E4  + human review of escalated decisions
    D   E3 + debate rounds before the adjudicator

    START -> triage -> fulltext (PubMed Central, if configured) -> [screen_reviewer x N] -> screen_consensus
       -> (split, rounds left) screen_debate -> screen_consensus ...
       -> (split, adjudicator) screen_adjudicate
       -> screen_checks -> (escalated, human) screen_human
       --include--> extract -> [annotate_reviewer x N] -> annotate_consensus
              -> (split, rounds left) annotate_debate -> annotate_consensus ...
              -> (split, adjudicator) annotate_adjudicate
       -> finalize (verifier) -> (escalated, human) final_human -> END
       --exclude--> finalize -> ...

Parallel reviewers use LangGraph's Send API; each sees only the article and the
codebook in the independent round. Votes are appended by the reducers in
ArticleState, so debate rounds never overwrite earlier votes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from adrd_review.agents import adjudicator, annotation, debate, extraction, screening, triage, verifier
from adrd_review.agents.common import allowed_labels
from adrd_review.agents.human import HumanReviewer
from adrd_review.codebook import Codebook
from adrd_review.consensus import combine_labels, combine_screening, is_split, latest_votes
from adrd_review.graph.state import ArticleState
from adrd_review.llm import LLM
from adrd_review.schemas import Decision, Escalation


@dataclass
class Reviewer:
    id: str
    llm: LLM  # screening (and screening debate)
    annotation_llm: LLM  # annotation (and annotation debate)
    persona: str | None = None
    weight: float = 1.0


@dataclass
class PipelineOptions:
    adjudicator: LLM | None = None  # E3+: settle split fields
    escalate_below: float = 0.6  # adjudications less confident than this go to a human
    verify: bool = False  # E3+: run the verifier step
    verifier: LLM | None = None  # model for the verifier's rule check (None = quote check only)
    debate_rounds: int = 0  # variant D: rounds of debate on split fields before adjudication
    human: HumanReviewer | None = None  # E4: answers escalated decisions
    fields_debated: list[str] = field(default_factory=list)  # empty = all fields
    fulltext: str = "never"  # never | when_needed (triage says abstract is not enough) | always
    fulltext_max_chars: int = 12000
    fulltext_fetcher: Callable | None = None  # article -> FullText | None (default: PubMed Central)


def build_pipeline(
    codebook: Codebook,
    triage_llm: LLM,
    extraction_llm: LLM,
    reviewers: list[Reviewer],
    options: PipelineOptions | None = None,
    checkpointer=None,
):
    if not reviewers:
        raise ValueError("At least one reviewer is needed")
    ids = [r.id for r in reviewers]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Reviewer IDs must be unique: {ids}")
    opt = options or PipelineOptions()
    by_id = {r.id: r for r in reviewers}
    weights = {r.id: r.weight for r in reviewers}
    dims = codebook.dimensions

    def debating(field_name: str) -> bool:
        return opt.debate_rounds > 0 and (not opt.fields_debated or field_name in opt.fields_debated)

    # ---- triage and screening ----------------------------------------------------

    def triage_node(state: ArticleState) -> dict:
        report, usage = triage.run(state["article"], codebook, triage_llm)
        return {"triage": report, "usage": [usage]}

    def fulltext_node(state: ArticleState) -> dict:
        art = state["article"]
        wanted = opt.fulltext == "always" or (
            opt.fulltext == "when_needed" and not state["triage"].enough_information
        )
        if not wanted:
            return {"fulltext_info": {"used": False, "reason": "not needed" if opt.fulltext != "never" else "off"}}
        if not art.pmcid:
            return {"fulltext_info": {"used": False, "reason": "no PMCID"}}
        from adrd_review.tools.pmc import excerpt, fetch_fulltext

        try:
            ft = (opt.fulltext_fetcher or fetch_fulltext)(art)
        except Exception as e:  # full text is a bonus: never fail the article over it
            return {"fulltext_info": {"used": False, "reason": f"fetch failed: {type(e).__name__}"}}
        if ft is None:
            return {"fulltext_info": {"used": False, "reason": "no PMC full text", "pmcid": art.pmcid}}
        text = excerpt(ft, opt.fulltext_max_chars)
        return {
            "article": art.model_copy(update={"fulltext": text}),
            "fulltext_info": {"used": True, "pmcid": ft.pmcid, "chars": len(text), "license": ft.license},
        }

    def fan_out_screening(state: ArticleState) -> list[Send]:
        return [Send("screen_reviewer", {"article": state["article"], "reviewer_id": r.id}) for r in reviewers]

    def screen_reviewer(payload: dict) -> dict:
        r = by_id[payload["reviewer_id"]]
        vote, usage = screening.run(payload["article"], codebook, r.llm, reviewer_id=r.id, persona=r.persona)
        return {"screening_votes": [vote], "usage": [usage]}

    def screen_consensus(state: ArticleState) -> dict:
        return {"screening_decision": combine_screening(latest_votes(state["screening_votes"]), weights)}

    def route_screen_consensus(state: ArticleState) -> str:
        split = is_split(latest_votes(state["screening_votes"]))
        if split and debating("screening") and state.get("screening_round", 0) < opt.debate_rounds:
            return "screen_debate"
        if split and opt.adjudicator is not None:
            return "screen_adjudicate"
        return "screen_checks"

    def screen_debate(state: ArticleState) -> dict:
        rnd = state.get("screening_round", 0) + 1
        current = latest_votes(state["screening_votes"])
        new, usages = [], []
        for v in debate.minority_first(current):
            r = by_id[v.reviewer_id]
            nv, usage = debate.run(state["article"], codebook, r.llm, "screening", v, current, rnd, r.persona)
            current = [nv if x.reviewer_id == nv.reviewer_id else x for x in current]
            new.append(nv)
            usages.append(usage)
        return {"screening_votes": new, "screening_round": rnd, "usage": usages}

    def screen_adjudicate(state: ArticleState) -> dict:
        votes = latest_votes(state["screening_votes"])
        decision, record, usage = adjudicator.run(state["article"], codebook, opt.adjudicator, "screening", votes)
        return {"screening_decision": decision, "adjudications": [record], "usage": [usage]}

    def screen_checks(state: ArticleState) -> dict:
        d = state["screening_decision"]
        out = []
        if d.method == "adjudicator" and (d.confidence or 0) < opt.escalate_below:
            out.append(Escalation(field="screening", reason="low_confidence", detail=f"adjudicator confidence {d.confidence}"))
        if d.labels == ["exclude"] and is_split(latest_votes(state["screening_votes"])):
            out.append(Escalation(field="screening", reason="split_exclusion", detail="reviewers disagreed; exclusion needs a human"))
        return {"escalations": out}

    def route_screen_checks(state: ArticleState) -> str:
        if opt.human is not None and _pending(state, ["screening"]):
            return "screen_human"
        return after_screening(state)

    def after_screening(state: ArticleState) -> str:
        return "extract" if state["screening_decision"].labels == ["include"] else "finalize"

    def human_node(fields: list[str] | None):
        def node(state: ArticleState) -> dict:
            updates: dict = {"human_answers": []}
            decisions = dict(state.get("annotation_decisions") or {})
            for esc in _pending(state, fields):
                current = (
                    state["screening_decision"] if esc.field == "screening" else decisions.get(esc.field)
                )
                item = _review_item(state, esc, current)
                labels = opt.human.answer(item)
                updates["human_answers"].append({"field": esc.field, "reason": esc.reason, "labels": labels, "mode": opt.human.mode})
                if esc.field == "screening":
                    inc = labels == ["include"]
                    updates["screening_decision"] = Decision(
                        field="screening", labels=labels, method="human", confidence=1.0, include_score=1.0 if inc else 0.0
                    )
                    if not inc:
                        decisions = {}
                else:
                    decisions[esc.field] = Decision(field=esc.field, labels=labels, method="human", confidence=1.0)
            if fields is None or any(f != "screening" for f in fields):
                updates["annotation_decisions"] = decisions
            return updates

        return node

    # ---- extraction and annotation ------------------------------------------------

    def extract_node(state: ArticleState) -> dict:
        record, usage = extraction.run(state["article"], extraction_llm)
        return {"extraction": record, "usage": [usage]}

    def fan_out_annotation(state: ArticleState) -> list[Send]:
        return [
            Send("annotate_reviewer", {"article": state["article"], "extraction": state.get("extraction"), "reviewer_id": r.id})
            for r in reviewers
        ]

    def annotate_reviewer(payload: dict) -> dict:
        r = by_id[payload["reviewer_id"]]
        votes, usage = annotation.run(
            payload["article"], codebook, r.annotation_llm, extraction=payload.get("extraction"), reviewer_id=r.id, persona=r.persona
        )
        return {"annotation_votes": votes, "usage": [usage]}

    def annotate_consensus(state: ArticleState) -> dict:
        decisions = {}
        for name, dim in dims.items():
            votes = latest_votes(state["annotation_votes"], name)
            if votes:
                decisions[name] = combine_labels(votes, name, dim.multi_label, dim.label_ids, weights)
        return {"annotation_decisions": decisions}

    def split_dims(state: ArticleState) -> list[str]:
        return [n for n in dims if is_split(latest_votes(state.get("annotation_votes", []), n))]

    def route_annotate_consensus(state: ArticleState) -> str:
        split = split_dims(state)
        if any(debating(f) for f in split) and state.get("annotation_round", 0) < opt.debate_rounds:
            return "annotate_debate"
        if split and opt.adjudicator is not None:
            return "annotate_adjudicate"
        return "finalize"

    def annotate_debate(state: ArticleState) -> dict:
        rnd = state.get("annotation_round", 0) + 1
        new, usages = [], []
        for name in [f for f in split_dims(state) if debating(f)]:
            current = latest_votes(state["annotation_votes"], name)
            for v in debate.minority_first(current):
                r = by_id[v.reviewer_id]
                nv, usage = debate.run(state["article"], codebook, r.annotation_llm, name, v, current, rnd, r.persona)
                current = [nv if x.reviewer_id == nv.reviewer_id else x for x in current]
                new.append(nv)
                usages.append(usage)
        return {"annotation_votes": new, "annotation_round": rnd, "usage": usages}

    def annotate_adjudicate(state: ArticleState) -> dict:
        decisions = dict(state["annotation_decisions"])
        records, usages, escalations = [], [], []
        for name in split_dims(state):
            votes = latest_votes(state["annotation_votes"], name)
            d, rec, usage = adjudicator.run(state["article"], codebook, opt.adjudicator, name, votes)
            decisions[name] = d
            records.append(rec)
            usages.append(usage)
            if (d.confidence or 0) < opt.escalate_below:
                escalations.append(Escalation(field=name, reason="low_confidence", detail=f"adjudicator confidence {d.confidence}"))
        return {"annotation_decisions": decisions, "adjudications": records, "usage": usages, "escalations": escalations}

    # ---- verification and final human review ---------------------------------------

    def finalize(state: ArticleState) -> dict:
        if not opt.verify:
            return {}
        decisions = [state["screening_decision"]]
        if state["screening_decision"].labels == ["include"]:
            decisions += list((state.get("annotation_decisions") or {}).values())
        votes = state.get("screening_votes", []) + state.get("annotation_votes", [])
        report, usage = verifier.run(state["article"], codebook, decisions, votes, opt.verifier)
        failed = {p.split(":", 1)[0] for p in report.quote_problems} | set(report.failed_fields)
        out: dict = {
            "verification": report,
            "escalations": [
                Escalation(field=f, reason="verifier_fail", detail="; ".join(
                    [p for p in report.quote_problems if p.startswith(f + ":")]
                    + [c.problem for c in report.checks if c.field == f and not c.supported]
                ))
                for f in sorted(failed)
            ],
        }
        if usage:
            out["usage"] = [usage]
        return out

    def route_finalize(state: ArticleState) -> str:
        if opt.human is not None and _pending(state, None):
            return "final_human"
        return END

    # ---- graph ------------------------------------------------------------------

    g = StateGraph(ArticleState)
    for name, fn in [
        ("triage", triage_node),
        ("fulltext", fulltext_node),
        ("screen_reviewer", screen_reviewer),
        ("screen_consensus", screen_consensus),
        ("screen_debate", screen_debate),
        ("screen_adjudicate", screen_adjudicate),
        ("screen_checks", screen_checks),
        ("screen_human", human_node(["screening"])),
        ("extract", extract_node),
        ("annotate_reviewer", annotate_reviewer),
        ("annotate_consensus", annotate_consensus),
        ("annotate_debate", annotate_debate),
        ("annotate_adjudicate", annotate_adjudicate),
        ("finalize", finalize),
        ("final_human", human_node(None)),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "triage")
    g.add_edge("triage", "fulltext")
    g.add_conditional_edges("fulltext", fan_out_screening, ["screen_reviewer"])
    g.add_edge("screen_reviewer", "screen_consensus")
    g.add_conditional_edges("screen_consensus", route_screen_consensus, ["screen_debate", "screen_adjudicate", "screen_checks"])
    g.add_edge("screen_debate", "screen_consensus")
    g.add_edge("screen_adjudicate", "screen_checks")
    g.add_conditional_edges("screen_checks", route_screen_checks, ["screen_human", "extract", "finalize"])
    g.add_conditional_edges("screen_human", after_screening, ["extract", "finalize"])
    g.add_conditional_edges("extract", fan_out_annotation, ["annotate_reviewer"])
    g.add_edge("annotate_reviewer", "annotate_consensus")
    g.add_conditional_edges(
        "annotate_consensus", route_annotate_consensus, ["annotate_debate", "annotate_adjudicate", "finalize"]
    )
    g.add_edge("annotate_debate", "annotate_consensus")
    g.add_edge("annotate_adjudicate", "finalize")
    g.add_conditional_edges("finalize", route_finalize, ["final_human", END])
    g.add_edge("final_human", END)
    return g.compile(checkpointer=checkpointer)


def _pending(state: dict, fields: list[str] | None) -> list[Escalation]:
    """Escalations not yet answered by a human (a human's answer on a field is final)."""
    answered = {a["field"] for a in state.get("human_answers", [])}
    out, seen = [], set()
    for e in state.get("escalations", []):
        if e.field in answered or e.field in seen or (fields is not None and e.field not in fields):
            continue
        if e.field != "screening" and state.get("screening_decision") and state["screening_decision"].labels != ["include"]:
            continue  # annotation of an excluded article needs no answer
        seen.add(e.field)
        out.append(e)
    return out


def _review_item(state: dict, esc: Escalation, current: Decision | None) -> dict:
    """What a human reviewer is shown for one escalated decision."""
    art = state["article"]
    votes = state.get("screening_votes", []) if esc.field == "screening" else state.get("annotation_votes", [])
    return {
        "pmid": art.pmid,
        "title": art.title,
        "abstract": art.abstract,
        "field": esc.field,
        "reason": esc.reason,
        "detail": esc.detail,
        "current": current.labels if current else [],
        "votes": [
            {"reviewer": v.reviewer_id, "round": v.round, "labels": v.labels, "rationale": v.rationale,
             "evidence": [e.quote for e in v.evidence]}
            for v in latest_votes(votes, esc.field)
        ],
    }


def allowed_for_item(codebook: Codebook, item: dict) -> list[str]:
    return allowed_labels(codebook, item["field"])
