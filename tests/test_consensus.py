"""The consensus step (E2) and the 3-reviewer panel pipeline."""

from pathlib import Path

import pytest

from adrd_review.codebook import load_codebook
from adrd_review.consensus import combine_labels, combine_screening, panel_agreement
from adrd_review.fake_model import default_responders
from adrd_review.graph.pipeline import Reviewer, build_pipeline
from adrd_review.llm import FakeLLM, ModelConfig
from adrd_review.schemas import Article, Vote

CODEBOOK = load_codebook(Path(__file__).resolve().parents[1] / "configs/codebook.yaml")
INSTRUMENTS = CODEBOOK.dimensions["instrument"].label_ids


def sv(rid, decision, conf=0.8):
    return Vote(reviewer_id=rid, model="m", field="screening", labels=[decision], confidence=conf)


def av(rid, labels, field="instrument"):
    return Vote(reviewer_id=rid, model="m", field=field, labels=labels, confidence=0.7)


# ---- screening ---------------------------------------------------------------


def test_single_vote_is_taken_as_is():
    d = combine_screening([sv("A", "exclude", 0.9)])
    assert d.labels == ["exclude"] and d.method == "single" and d.confidence == 0.9
    assert d.include_score == pytest.approx(0.1)


def test_unanimous_and_majority():
    d = combine_screening([sv("A", "include"), sv("B", "include"), sv("C", "include")])
    assert (d.labels, d.method, d.confidence) == (["include"], "unanimous", 1.0)
    d = combine_screening([sv("A", "include"), sv("B", "exclude"), sv("C", "exclude")])
    assert (d.labels, d.method) == (["exclude"], "majority")
    assert d.confidence == pytest.approx(2 / 3, abs=1e-4)


def test_weights_can_flip_a_majority():
    votes = [sv("A", "include"), sv("B", "exclude"), sv("C", "exclude")]
    assert combine_screening(votes, {"A": 3.0, "B": 1.0, "C": 1.0}).labels == ["include"]


def test_tie_goes_to_include():
    d = combine_screening([sv("A", "include"), sv("B", "exclude")])
    assert (d.labels, d.method) == (["include"], "tie_include")


def test_include_score_is_weighted_mean_probability():
    # A: include 0.9 -> 0.9 ; B: exclude 0.6 -> 0.4 ; C: exclude 0.8 -> 0.2
    d = combine_screening([sv("A", "include", 0.9), sv("B", "exclude", 0.6), sv("C", "exclude", 0.8)])
    assert d.include_score == pytest.approx((0.9 + 0.4 + 0.2) / 3, abs=1e-4)


# ---- annotation labels ---------------------------------------------------------


def test_multilabel_keeps_labels_backed_by_half_the_weight():
    votes = [av("A", ["MMSE", "MoCA"]), av("B", ["MMSE"]), av("C", ["MMSE", "IQCODE"])]
    d = combine_labels(votes, "instrument", True, INSTRUMENTS)
    assert d.labels == ["MMSE"] and d.method == "majority"
    votes = [av("A", ["MoCA", "MMSE"]), av("B", ["MMSE", "MoCA"]), av("C", ["MMSE"])]
    assert combine_labels(votes, "instrument", True, INSTRUMENTS).labels == ["MMSE", "MoCA"]  # codebook order


def test_multilabel_unanimous_including_empty():
    d = combine_labels([av("A", []), av("B", [])], "instrument", True, INSTRUMENTS)
    assert (d.labels, d.method) == ([], "unanimous")


def test_single_label_plurality_and_empty_winning():
    votes = [av("A", ["MoCA"]), av("B", ["MMSE"]), av("C", ["MMSE"])]
    assert combine_labels(votes, "x", False, INSTRUMENTS).labels == ["MMSE"]
    votes = [av("A", []), av("B", []), av("C", ["MoCA"])]
    assert combine_labels(votes, "x", False, INSTRUMENTS).labels == []
    votes = [av("A", ["MoCA"]), av("B", ["MMSE"])]  # tie -> first in codebook order
    assert combine_labels(votes, "x", False, INSTRUMENTS).labels == ["MMSE"]


# ---- the panel graph --------------------------------------------------------------

ART = Article(
    pmid="1",
    title="The MMSE in older adults with dementia.",
    abstract="We gave the MMSE to 200 community-dwelling adults aged 65+ with suspected dementia.",
)  # names instrument + dementia, but no accuracy term -> strict reviewer excludes


def panel():
    llms = {s: FakeLLM(ModelConfig(provider="fake", model=s), default_responders(CODEBOOK, s))
            for s in ("keyword-rules", "keyword-strict", "keyword-lenient")}
    base = llms["keyword-rules"]
    reviewers = [Reviewer("A", llms["keyword-rules"], base), Reviewer("B", llms["keyword-strict"], base),
                 Reviewer("C", llms["keyword-lenient"], base)]
    return build_pipeline(CODEBOOK, base, base, reviewers)


def test_panel_runs_three_independent_reviewers_and_takes_the_majority():
    state = panel().invoke({"article": ART})
    votes = {v.reviewer_id: v.labels[0] for v in state["screening_votes"]}
    assert votes == {"A": "include", "B": "exclude", "C": "include"}
    d = state["screening_decision"]
    assert (d.labels, d.method) == (["include"], "majority")
    assert len(state["annotation_votes"]) == 3 * len(CODEBOOK.dimensions)
    assert state["annotation_decisions"]["instrument"].labels == ["MMSE"]
    assert state["annotation_decisions"]["instrument"].method == "unanimous"
    roles = sorted(u.role for u in state["usage"])
    assert roles.count("screening") == 3 and roles.count("annotation") == 3


def test_panel_agreement_fleiss():
    s1 = {"screening_votes": [sv("A", "include"), sv("B", "include"), sv("C", "include")]}
    s2 = {"screening_votes": [sv("A", "exclude"), sv("B", "exclude"), sv("C", "include")]}
    out = panel_agreement([s1, s2], ["A", "B", "C"], [])
    assert out["screening"]["items"] == 2 and out["screening"]["unanimous"] == 1
    assert panel_agreement([s1], ["A"], []) is None
