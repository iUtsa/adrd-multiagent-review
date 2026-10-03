"""The review pipeline as a LangGraph graph. One graph serves every condition;
PipelineOptions switch steps on:

    E1  1 reviewer
    E2  3 reviewers + weighted majority vote
    E3  + adjudicator on split fields, + verifier
    E4  + human review of escalated decisions
    D   E3 + debate rounds before the adjudicator
    two-stage (eligibility=True): after abstract screening, the full paper is fetched and
        judged strictly before extraction and annotation, as human reviewers do
    three-stage (title_stage=True, with eligibility=True): titles are screened first, as the
        professor's team did: titles -> abstracts -> full papers

    START -> [title_stage only] [title_reviewer x N] -> title_consensus
              --every reviewer excludes--> finalize
    START -> triage -> fulltext (optional excerpt) -> [screen_reviewer x N] -> screen_consensus
       -> (split, rounds left) screen_debate -> screen_consensus ...
       -> (split, adjudicator) screen_adjudicate
       -> screen_checks -> (escalated, human) screen_human
       --exclude--> finalize
       --include--> [two-stage only] get_fulltext
              -> (no full text) elig_checks: flagged, a human decides
              -> [elig_reviewer x N] -> elig_consensus -> (debate) -> (adjudicate)
              -> elig_checks -> (escalated, human) elig_human
              --exclude--> finalize
       --include--> extract -> [annotate_reviewer x N] -> annotate_consensus
              -> (split, rounds left) annotate_debate -> annotate_consensus ...
              -> (split, adjudicator) annotate_adjudicate
       -> finalize (verifier) -> (escalated, human) final_human -> END

Parallel reviewers use LangGraph's Send API; each sees only the article and the
codebook in the independent round. Votes are appended by the reducers in
ArticleState, so debate rounds never overwrite earlier votes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from adrd_review.agents import (
    adjudicator,
    annotation,
    debate,
    eligibility,
    extraction,
    screening,
    title_screening,
    triage,
    verifier,
)
from adrd_review.agents.common import allowed_labels
from adrd_review.agents.human import HumanReviewer
from adrd_review.codebook import Codebook
from adrd_review.consensus import (
    combine_eligibility,
    combine_labels,
    combine_screening,
    combine_title,
    is_split,
    latest_votes,
)
from adrd_review.graph.state import ArticleState
from adrd_review.llm import LLM
from adrd_review.schemas import Decision, Escalation


@dataclass
class Reviewer:
    id: str
    llm: LLM  # screening and eligibility (and their debate rounds)
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
    # Stage 0: title-only screening before the abstract (removes only the obviously off-topic)
    title_stage: bool = False
    # Stage 1 extra: show a PubMed Central excerpt at abstract screening
    fulltext: str = "never"  # never | when_needed (triage says abstract is not enough) | always
    fulltext_max_chars: int = 12000
    fulltext_fetcher: Callable | None = None  # article -> FullText | None (default: PubMed Central)
    # Stage 2: full-text eligibility
    eligibility: bool = False
    eligibility_max_chars: int = 80000  # whole paper for almost every article (longest sampled: 67,100)
    fulltext_source: Callable | None = None  # article -> FullTextResult (default: tools.fulltext.get_fulltext)


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

    def debate_round(state: ArticleState, field_name: str, votes_key: str, llm_of) -> tuple[list, list]:
        """One debate round on one field: minority first, later speakers see earlier replies."""
        rnd = state.get(f"{votes_key.split('_')[0]}_round", 0) + 1
        current = latest_votes(state[votes_key], field_name)
        new, usages = [], []
        for v in debate.minority_first(current):
            r = by_id[v.reviewer_id]
            nv, usage = debate.run(state["article"], codebook, llm_of(r), field_name, v, current, rnd, r.persona)
            current = [nv if x.reviewer_id == nv.reviewer_id else x for x in current]
            new.append(nv)
            usages.append(usage)
        return new, usages

    # ---- stage 0: title screening ----------------------------------------------------

    def route_start(state: ArticleState):
        if not opt.title_stage:
            return "triage"
        return [Send("title_reviewer", {"article": state["article"], "reviewer_id": r.id}) for r in reviewers]

    def title_reviewer(payload: dict) -> dict:
        r = by_id[payload["reviewer_id"]]
        vote, usage = title_screening.run(payload["article"], codebook, r.llm, reviewer_id=r.id, persona=r.persona)
        return {"title_votes": [vote], "usage": [usage]}

    def title_consensus(state: ArticleState) -> dict:
        d = combine_title(latest_votes(state["title_votes"]), weights)
        out: dict = {"title_decision": d}
        if d.labels == ["exclude"]:
            # Excluded at the title: the abstract is never screened. Recorded as a screening exclusion
            # so scoring and the rest of the pipeline treat it like any other stage-1 exclusion.
            out["screening_decision"] = Decision(field="screening", labels=["exclude"], method="title_excluded",
                                                 confidence=d.confidence, include_score=d.include_score)
        return out

    def route_title(state: ArticleState) -> str:
        return "finalize" if state["title_decision"].labels == ["exclude"] else "triage"

    # ---- triage and stage 1: title/abstract screening ------------------------------

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
        new, usages = debate_round(state, "screening", "screening_votes", lambda r: r.llm)
        return {"screening_votes": new, "screening_round": state.get("screening_round", 0) + 1, "usage": usages}

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
        if state["screening_decision"].labels != ["include"]:
            return "finalize"
        return "get_fulltext" if opt.eligibility else "extract"

    # ---- stage 2: full-text eligibility ----------------------------------------------

    def get_fulltext_node(state: ArticleState) -> dict:
        from adrd_review.tools.fulltext import fit_text, get_fulltext

        art = state["article"]
        res = (opt.fulltext_source or get_fulltext)(art)
        if not res.found:
            return {"fulltext_info": {"used": False, "stage": 2, "reason": res.reason, "link": res.link}}
        text, truncated = fit_text(res.fulltext, opt.eligibility_max_chars)
        return {
            "article": art.model_copy(update={"fulltext": text}),
            "fulltext_info": {
                "used": True, "stage": 2, "source": res.source, "detail": res.reason, "chars": len(text),
                "truncated": truncated, "sections": len(res.fulltext.sections),
            },
        }

    def route_after_fulltext(state: ArticleState):
        if not state["article"].fulltext:
            return "elig_checks"
        return [Send("elig_reviewer", {"article": state["article"], "reviewer_id": r.id}) for r in reviewers]

    def elig_reviewer(payload: dict) -> dict:
        r = by_id[payload["reviewer_id"]]
        vote, usage = eligibility.run(payload["article"], codebook, r.llm, reviewer_id=r.id, persona=r.persona)
        return {"eligibility_votes": [vote], "usage": [usage]}

    def elig_consensus(state: ArticleState) -> dict:
        return {"eligibility_decision": combine_eligibility(latest_votes(state["eligibility_votes"]), weights)}

    def route_elig_consensus(state: ArticleState) -> str:
        split = is_split(latest_votes(state["eligibility_votes"]))
        if split and debating("eligibility") and state.get("eligibility_round", 0) < opt.debate_rounds:
            return "elig_debate"
        if split and opt.adjudicator is not None:
            return "elig_adjudicate"
        return "elig_checks"

    def elig_debate(state: ArticleState) -> dict:
        new, usages = debate_round(state, "eligibility", "eligibility_votes", lambda r: r.llm)
        return {"eligibility_votes": new, "eligibility_round": state.get("eligibility_round", 0) + 1, "usage": usages}

    def elig_adjudicate(state: ArticleState) -> dict:
        votes = latest_votes(state["eligibility_votes"])
        decision, record, usage = adjudicator.run(state["article"], codebook, opt.adjudicator, "eligibility", votes)
        return {"eligibility_decision": decision, "adjudications": [record], "usage": [usage]}

    def elig_checks(state: ArticleState) -> dict:
        info = state.get("fulltext_info") or {}
        if not state["article"].fulltext:
            # No full text anywhere: never an exclusion. Flag it and let a human decide.
            return {
                "eligibility_decision": Decision(field="eligibility", labels=["unclear"], method="no_fulltext", include_score=0.5),
                "escalations": [Escalation(field="eligibility", reason="no_fulltext",
                                           detail=f"{info.get('reason', 'no full text')}; get it: {info.get('link')}")],
            }
        d = state["eligibility_decision"]
        out = []
        if d.labels == ["unclear"]:
            out.append(Escalation(field="eligibility", reason="unclear_fulltext", detail="the full paper does not settle every rule"))
        elif d.method == "adjudicator" and (d.confidence or 0) < opt.escalate_below:
            out.append(Escalation(field="eligibility", reason="low_confidence", detail=f"adjudicator confidence {d.confidence}"))
        if d.labels == ["exclude"] and is_split(latest_votes(state["eligibility_votes"])):
            out.append(Escalation(field="eligibility", reason="split_exclusion", detail="reviewers disagreed; exclusion needs a human"))
        return {"escalations": out}

    def route_elig_checks(state: ArticleState) -> str:
        if opt.human is not None and _pending(state, ["eligibility"]):
            return "elig_human"
        return after_eligibility(state)

    def after_eligibility(state: ArticleState) -> str:
        # "unclear" with nobody to ask stays in (flagged): a study is never dropped for lack of an answer.
        return "finalize" if state["eligibility_decision"].labels == ["exclude"] else "extract"

    # ---- human review (used after screening, after eligibility, and at the end) --------

    def human_node(fields: list[str] | None):
        def node(state: ArticleState) -> dict:
            updates: dict = {"human_answers": []}
            decisions = dict(state.get("annotation_decisions") or {})
            for esc in _pending(state, fields):
                current = (
                    state.get(f"{esc.field}_decision") if esc.field in ("screening", "eligibility") else decisions.get(esc.field)
                )
                item = _review_item(state, esc, current)
                labels = opt.human.answer(item)
                updates["human_answers"].append({"field": esc.field, "reason": esc.reason, "labels": labels, "mode": opt.human.mode})
                if esc.field in ("screening", "eligibility"):
                    inc = labels == ["include"]
                    updates[f"{esc.field}_decision"] = Decision(
                        field=esc.field, labels=labels, method="human", confidence=1.0, include_score=1.0 if inc else 0.0
                    )
                    if not inc:
                        decisions = {}
                else:
                    decisions[esc.field] = Decision(field=esc.field, labels=labels, method="human", confidence=1.0)
            if fields is None:
                updates["annotation_decisions"] = decisions
            return updates

        return node

    # ---- extraction and annotation ------------------------------------------------

    def extract_node(state: ArticleState) -> dict:
        record, usage = extraction.run(state["article"], extraction_llm, codebook.topic)
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
        new, usages = [], []
        for name in [f for f in split_dims(state) if debating(f)]:
            n, u = debate_round(state, name, "annotation_votes", lambda r: r.annotation_llm)
            new += n
            usages += u
        return {"annotation_votes": new, "annotation_round": state.get("annotation_round", 0) + 1, "usage": usages}

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
        if not opt.verify or state["screening_decision"].method == "title_excluded":
            return {}  # title exclusions need every reviewer to agree; nothing for the verifier to check
        decisions = [state["screening_decision"]]
        elig = state.get("eligibility_decision")
        if elig is not None and elig.labels != ["unclear"]:
            decisions.append(elig)
        if finally_included(state):
            decisions += list((state.get("annotation_decisions") or {}).values())
        # A person's decision is final: the verifier checks agents' decisions only.
        decisions = [d for d in decisions if d.method != "human"]
        if not decisions:
            return {}
        votes = state.get("screening_votes", []) + state.get("eligibility_votes", []) + state.get("annotation_votes", [])
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
        ("title_reviewer", title_reviewer),
        ("title_consensus", title_consensus),
        ("triage", triage_node),
        ("fulltext", fulltext_node),
        ("screen_reviewer", screen_reviewer),
        ("screen_consensus", screen_consensus),
        ("screen_debate", screen_debate),
        ("screen_adjudicate", screen_adjudicate),
        ("screen_checks", screen_checks),
        ("screen_human", human_node(["screening"])),
        ("get_fulltext", get_fulltext_node),
        ("elig_reviewer", elig_reviewer),
        ("elig_consensus", elig_consensus),
        ("elig_debate", elig_debate),
        ("elig_adjudicate", elig_adjudicate),
        ("elig_checks", elig_checks),
        ("elig_human", human_node(["eligibility"])),
        ("extract", extract_node),
        ("annotate_reviewer", annotate_reviewer),
        ("annotate_consensus", annotate_consensus),
        ("annotate_debate", annotate_debate),
        ("annotate_adjudicate", annotate_adjudicate),
        ("finalize", finalize),
        ("final_human", human_node(None)),
    ]:
        g.add_node(name, fn)

    next_after_screen = ["get_fulltext", "extract", "finalize"]
    g.add_conditional_edges(START, route_start, ["title_reviewer", "triage"])
    g.add_edge("title_reviewer", "title_consensus")
    g.add_conditional_edges("title_consensus", route_title, ["triage", "finalize"])
    g.add_edge("triage", "fulltext")
    g.add_conditional_edges("fulltext", fan_out_screening, ["screen_reviewer"])
    g.add_edge("screen_reviewer", "screen_consensus")
    g.add_conditional_edges("screen_consensus", route_screen_consensus, ["screen_debate", "screen_adjudicate", "screen_checks"])
    g.add_edge("screen_debate", "screen_consensus")
    g.add_edge("screen_adjudicate", "screen_checks")
    g.add_conditional_edges("screen_checks", route_screen_checks, ["screen_human", *next_after_screen])
    g.add_conditional_edges("screen_human", after_screening, next_after_screen)

    g.add_conditional_edges("get_fulltext", route_after_fulltext, ["elig_reviewer", "elig_checks"])
    g.add_edge("elig_reviewer", "elig_consensus")
    g.add_conditional_edges("elig_consensus", route_elig_consensus, ["elig_debate", "elig_adjudicate", "elig_checks"])
    g.add_edge("elig_debate", "elig_consensus")
    g.add_edge("elig_adjudicate", "elig_checks")
    g.add_conditional_edges("elig_checks", route_elig_checks, ["elig_human", "extract", "finalize"])
    g.add_conditional_edges("elig_human", after_eligibility, ["extract", "finalize"])

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


def finally_included(state: dict) -> bool:
    """Included after every stage that ran (an unresolved 'unclear' at full text stays in, flagged)."""
    if state["screening_decision"].labels != ["include"]:
        return False
    elig = state.get("eligibility_decision")
    return elig is None or elig.labels != ["exclude"]


def _pending(state: dict, fields: list[str] | None) -> list[Escalation]:
    """Escalations not yet answered by a human (a human's answer on a field is final)."""
    answered = {a["field"] for a in state.get("human_answers", [])}
    out, seen = [], set()
    for e in state.get("escalations", []):
        if e.field in answered or e.field in seen or (fields is not None and e.field not in fields):
            continue
        if e.field not in ("screening", "eligibility") and state.get("screening_decision") and not finally_included(state):
            continue  # annotation of an excluded article needs no answer
        if e.field == "eligibility" and state["screening_decision"].labels != ["include"]:
            continue
        seen.add(e.field)
        out.append(e)
    return out


def _review_item(state: dict, esc: Escalation, current: Decision | None) -> dict:
    """What a human reviewer is shown for one escalated decision."""
    art = state["article"]
    key = {"screening": "screening_votes", "eligibility": "eligibility_votes"}.get(esc.field, "annotation_votes")
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
            for v in latest_votes(state.get(key, []), esc.field)
        ],
    }


def allowed_for_item(codebook: Codebook, item: dict) -> list[str]:
    return allowed_labels(codebook, item["field"])
