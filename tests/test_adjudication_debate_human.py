"""Day 6-8: adjudicator, verifier, screening safety rule, human review, debate, curator, freeze."""

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from adrd_review import freeze
from adrd_review.agents import adjudicator, debate, verifier
from adrd_review.agents.common import render_sides
from adrd_review.agents.human import GoldHuman, QueueHuman
from adrd_review.codebook import load_codebook
from adrd_review.fake_model import default_responders
from adrd_review.gold import GoldRecord
from adrd_review.graph.pipeline import PipelineOptions, Reviewer, build_pipeline
from adrd_review.llm import FakeLLM, ModelConfig
from adrd_review.schemas import (
    AdjudicationAnswer,
    Article,
    DebateAnswer,
    Decision,
    EvidenceSpan,
    ReviewerWeight,
    ScreeningAnswer,
    VerifierAnswer,
    VerifierCheck,
    Vote,
)

ROOT = Path(__file__).resolve().parents[1]
CODEBOOK = load_codebook(ROOT / "configs/codebook.yaml")
ART = Article(
    pmid="42",
    title="The MMSE for dementia in primary care.",
    abstract="We gave the MMSE to 300 primary care patients. Dementia was diagnosed with DSM-IV. Sensitivity was 0.8.",
)


def fake(responders=None, name="keyword-rules"):
    base = default_responders(CODEBOOK)
    return FakeLLM(ModelConfig(provider="fake", model=name), {**base, **(responders or {})})


def vote(rid, labels, quotes=(), field="screening", rnd=0, conf=0.8):
    return Vote(reviewer_id=rid, model="m", field=field, labels=list(labels), round=rnd, confidence=conf,
                evidence=[EvidenceSpan(quote=q) for q in quotes], rationale=f"{rid} reasons")


def screening_says(decision):
    return {ScreeningAnswer: lambda *_: ScreeningAnswer(
        decision=decision, rule_ids=["I1"], evidence=[EvidenceSpan(quote="We gave the MMSE to 300 primary care patients.")],
        rationale="r", confidence=0.8)}


def adjudicator_says(labels, weights, confidence):
    return {AdjudicationAnswer: lambda *_: AdjudicationAnswer(
        labels=labels, reviewer_weights=[ReviewerWeight(reviewer=r, weight=w) for r, w in weights.items()],
        rationale="r", confidence=confidence)}


# ---- adjudicator ---------------------------------------------------------------


def test_adjudicator_weights_the_panel_rather_than_overriding_it():
    votes = [vote("A", ["include"]), vote("B", ["exclude"]), vote("C", ["exclude"])]
    llm = fake(adjudicator_says(["include"], {"A": 0.9, "B": 0.2, "C": 0.2}, 0.8))
    d, rec, _ = adjudicator.run(ART, CODEBOOK, llm, "screening", votes)
    assert d.labels == ["include"] and d.method == "adjudicator" and d.confidence == 0.8
    assert rec.reviewer_weights == {"A": 0.9, "B": 0.2, "C": 0.2} and not rec.warnings


def test_adjudicator_inconsistent_answer_goes_to_a_human():
    votes = [vote("A", ["include"]), vote("B", ["exclude"]), vote("C", ["exclude"])]
    llm = fake(adjudicator_says(["include"], {"A": 0.1, "B": 0.9, "C": 0.9}, 0.9))  # says include, weights exclude
    d, rec, _ = adjudicator.run(ART, CODEBOOK, llm, "screening", votes)
    assert d.labels == ["exclude"] and d.confidence == 0.0
    assert any("differs" in w for w in rec.warnings)


def test_adjudicator_fills_missing_weights_and_drops_bad_labels():
    votes = [vote("A", ["MMSE"], field="instrument"), vote("B", ["MoCA"], field="instrument")]
    llm = fake(adjudicator_says(["MMSE", "SLUMS"], {"A": 0.9, "Z": 1.0}, 0.7))
    d, rec, _ = adjudicator.run(ART, CODEBOOK, llm, "instrument", votes)
    assert rec.reviewer_weights == {"A": 0.9, "B": 0.5}
    assert rec.labels == ["MMSE"] and any("SLUMS" in w for w in rec.warnings)
    assert d.labels == ["MMSE"]


# ---- verifier --------------------------------------------------------------------


def test_verifier_flags_decisions_resting_only_on_invented_quotes():
    d = Decision(field="screening", labels=["include"], method="unanimous")
    good = [vote("A", ["include"], ["We gave the MMSE to 300 primary care patients."])]
    bad = [vote("A", ["include"], ["The MoCA was superior."])]
    assert verifier.run(ART, CODEBOOK, [d], good)[0].quote_problems == []
    assert verifier.run(ART, CODEBOOK, [d], bad)[0].quote_problems == [
        "screening: none of the supporting quotes appear in the article"
    ]


def test_verifier_matches_votes_by_reviewer_and_field():
    # Regression: a reviewer's later annotation vote must not hide its screening vote.
    d = Decision(field="screening", labels=["include"], method="unanimous")
    votes = [
        vote("A", ["include"], ["We gave the MMSE to 300 primary care patients."]),
        vote("A", ["MMSE"], ["We gave the MMSE to 300 primary care patients."], field="instrument"),
    ]
    support = verifier.supporting_votes(d, votes)
    assert [(v.reviewer_id, v.field) for v in support] == [("A", "screening")]
    assert 'Evidence: "We gave the MMSE' in verifier.user_prompt(ART, [d], votes)


def test_verifier_model_check_reports_failed_fields():
    d = Decision(field="screening", labels=["include"], method="unanimous")
    llm = fake({VerifierAnswer: lambda *_: VerifierAnswer(checks=[VerifierCheck(field="screening", supported=False, problem="I2 not shown")])})
    report, usage = verifier.run(ART, CODEBOOK, [d], [vote("A", ["include"], ["We gave the MMSE to 300 primary care patients."])], llm)
    assert report.failed_fields == ["screening"] and usage.role == "verifier"


# ---- debate rules ------------------------------------------------------------------


def debate_says(labels, new_quotes=()):
    return fake({DebateAnswer: lambda *_: DebateAnswer(
        labels=labels, new_evidence=[EvidenceSpan(quote=q) for q in new_quotes], rationale="r", confidence=0.7)})


def test_change_without_new_evidence_is_rejected():
    own = vote("B", ["exclude"], ["We gave the MMSE to 300 primary care patients."])
    nv, _ = debate.run(ART, CODEBOOK, debate_says(["include"]), "screening", own, [own], 1)
    assert nv.labels == ["exclude"] and nv.round == 1
    assert any("rejected" in w for w in nv.warnings)


def test_change_with_invented_or_repeated_quote_is_rejected():
    own = vote("B", ["exclude"], ["We gave the MMSE to 300 primary care patients."])
    for quotes in (["A made-up sentence."], ["We gave the MMSE to 300 primary care patients."]):
        nv, _ = debate.run(ART, CODEBOOK, debate_says(["include"], quotes), "screening", own, [own], 1)
        assert nv.labels == ["exclude"]


def test_change_with_genuinely_new_evidence_is_accepted():
    own = vote("B", ["exclude"], ["We gave the MMSE to 300 primary care patients."])
    nv, _ = debate.run(ART, CODEBOOK, debate_says(["include"], ["Dementia was diagnosed with DSM-IV."]), "screening", own, [own], 1)
    assert nv.labels == ["include"] and len(nv.evidence) == 2 and not nv.warnings


def test_minority_speaks_first_and_arguments_hide_counts():
    votes = [vote("A", ["include"], ["q1"]), vote("B", ["exclude"], ["q2"]), vote("C", ["include"], ["q1"])]
    assert [v.reviewer_id for v in debate.minority_first(votes)] == ["B", "A", "C"]
    sides = render_sides(votes)
    assert sides.count('"q1"') == 1  # duplicate arguments merged: no hint of how many hold it
    assert "Reviewer" not in sides and "reviewers" not in sides.lower()
    assert sides.count("Position:") == 2  # one block per position, however many hold it


# ---- the pipeline with adjudication, safety rule, verifier and humans ----------------------


def panel_graph(decisions, options, adj=None, checkpointer=None, reviewer_extra=None):
    reviewers = [Reviewer(rid, fake({**screening_says(d), **(reviewer_extra or {})}), fake()) for rid, d in zip("ABC", decisions)]
    base = fake(adj or {})
    options.adjudicator = options.adjudicator and base
    return build_pipeline(CODEBOOK, base, base, reviewers, options, checkpointer=checkpointer)


def test_split_goes_to_adjudicator_and_split_exclusion_is_escalated():
    g = panel_graph(["include", "exclude", "exclude"], PipelineOptions(adjudicator=True),
                    adjudicator_says(["exclude"], {"A": 0.3, "B": 0.8, "C": 0.8}, 0.9))
    s = g.invoke({"article": ART})
    assert s["screening_decision"].method == "adjudicator" and s["screening_decision"].labels == ["exclude"]
    assert [e.reason for e in s["escalations"]] == ["split_exclusion"]
    assert len(s["adjudications"]) == 1


def test_low_confidence_adjudication_is_escalated():
    g = panel_graph(["include", "include", "exclude"], PipelineOptions(adjudicator=True, escalate_below=0.6),
                    adjudicator_says(["include"], {"A": 0.6, "B": 0.6, "C": 0.5}, 0.4))
    s = g.invoke({"article": ART})
    assert s["screening_decision"].labels == ["include"]
    assert "low_confidence" in [e.reason for e in s["escalations"]]


def test_unanimous_needs_no_adjudicator_and_no_human():
    g = panel_graph(["include"] * 3, PipelineOptions(adjudicator=True, verify=True))
    s = g.invoke({"article": ART})
    assert s["screening_decision"].method == "unanimous" and not s.get("adjudications") and not s.get("escalations")
    assert s["verification"].quote_problems == []


def test_gold_human_answers_escalations():
    gold = {"42": GoldRecord(pmid="42", screening="include", labels={"instrument": ["MMSE"]})}
    g = panel_graph(["include", "exclude", "exclude"], PipelineOptions(human=GoldHuman(gold)))
    s = g.invoke({"article": ART})  # E2 majority says exclude on a split -> safety rule -> human
    assert s["screening_decision"].labels == ["include"] and s["screening_decision"].method == "human"
    assert s["human_answers"][0]["mode"] == "gold"
    assert "annotation_decisions" in s  # human include sends it on to annotation


def test_queue_human_pauses_and_resumes():
    g = panel_graph(["include", "exclude", "exclude"], PipelineOptions(human=QueueHuman()), checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "42"}}
    first = g.invoke({"article": ART}, cfg)
    assert first["__interrupt__"][0].value["field"] == "screening"
    assert first["__interrupt__"][0].value["reason"] == "split_exclusion"
    done = g.invoke(Command(resume=["exclude"]), cfg)
    assert done["screening_decision"].labels == ["exclude"] and done["screening_decision"].method == "human"
    assert not g.get_state(cfg).interrupts


def test_debate_rounds_run_before_adjudication():
    g = panel_graph(["include", "exclude", "exclude"], PipelineOptions(adjudicator=True, debate_rounds=2),
                    adjudicator_says(["exclude"], {"A": 0.3, "B": 0.8, "C": 0.8}, 0.9),
                    reviewer_extra={DebateAnswer: lambda *_: DebateAnswer(labels=["exclude"], rationale="keep", confidence=0.6)})
    s = g.invoke({"article": ART})
    # the fake debater always answers 'exclude': B, C keep; A's switch has no new evidence -> rejected
    rounds = sorted({v.round for v in s["screening_votes"]})
    assert rounds == [0, 1, 2] and s["screening_round"] == 2
    assert s["screening_decision"].method == "adjudicator"
    assert any("rejected" in w for v in s["screening_votes"] for w in v.warnings)


# ---- curator + freeze ------------------------------------------------------------------


def test_curator_refuses_test_runs(tmp_path):
    from adrd_review import curate

    (tmp_path / "run_meta.json").write_text(json.dumps({"part": "test", "config": {}}))
    assert curate.main([str(tmp_path), "--model", "fake:keyword-rules"]) == 1


def test_freeze_detects_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "FROZEN_DIR", tmp_path / "frozen")
    for name in ("cb.yaml", "gold.csv", "split.json"):
        (tmp_path / name).write_text(name)
    cfg = tmp_path / "E9.yaml"
    cfg.write_text(f"name: E9\ncodebook: {tmp_path}/cb.yaml\ngold: {tmp_path}/gold.csv\nsplit: {tmp_path}/split.json\n")
    assert "no frozen manifest" in freeze.check(cfg)[0]
    freeze.create(cfg)
    assert freeze.check(cfg) == []
    with pytest.raises(FileExistsError):
        freeze.create(cfg)
    (tmp_path / "cb.yaml").write_text("edited after freeze")
    assert freeze.check(cfg) == ["codebook_sha256 changed since the freeze"]


def test_runner_refuses_unfrozen_test_run():
    from adrd_review.graph.runner import execute, load_run_config

    cfg = ROOT / "configs/runs/E2_fake.yaml"
    with pytest.raises(SystemExit, match="Refusing to run on the TEST set"):
        execute(load_run_config(cfg), part="test", config_path=str(cfg))
