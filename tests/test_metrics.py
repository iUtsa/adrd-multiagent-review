"""Metrics are checked two ways: hand-worked examples with known answers, and
agreement with scikit-learn / statsmodels on random data."""

import random
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
from statsmodels.stats.inter_rater import aggregate_raters
from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss

from adrd_review.codebook import load_codebook
from adrd_review.eval.metrics import (
    binary_scores,
    cohen_kappa,
    fleiss_kappa,
    multilabel_scores,
    usage_summary,
    wss,
    wss_at_recall,
)
from adrd_review.eval.score import evaluate
from adrd_review.gold import GoldRecord

CODEBOOK = load_codebook(Path(__file__).resolve().parents[1] / "configs/codebook.yaml")


# ---- hand-worked examples ---------------------------------------------------


def test_binary_scores_worked_example():
    # The example from the guide: humans include 100; agents include 105 = 95 right + 10 wrong.
    gold = [True] * 100 + [False] * 100
    pred = [True] * 95 + [False] * 5 + [True] * 10 + [False] * 90
    s = binary_scores(gold, pred)
    assert (s.tp, s.fp, s.fn, s.tn) == (95, 10, 5, 90)
    assert s.recall == 0.95
    assert s.precision == pytest.approx(95 / 105)
    assert s.specificity == 0.9


def test_binary_scores_no_positive_predictions_gives_zero_not_error():
    s = binary_scores([True, False], [False, False])
    assert s.precision == 0.0 and s.recall == 0.0 and s.f1 == 0.0


def test_cohen_kappa_known_value():
    # Textbook 2x2: agree 20 yes + 15 no of 50; rater A yes 25, rater B yes 30.
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    # po = 0.7, pe = 0.5*0.6 + 0.5*0.4 = 0.5 -> kappa 0.4
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_edge_cases():
    assert cohen_kappa(["a", "b"], ["a", "b"]) == 1.0
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0  # constant raters that agree
    with pytest.raises(ValueError, match="Length mismatch"):
        cohen_kappa(["a"], ["a", "b"])


def test_fleiss_kappa_perfect_and_textbook():
    assert fleiss_kappa([["i", "i", "i"], ["e", "e", "e"]]) == 1.0
    # Fleiss (1971) style small example, value checked against statsmodels below too.
    rows = [["a", "a", "b"], ["a", "b", "b"], ["b", "b", "b"], ["a", "a", "a"]]
    table, _ = aggregate_raters(np.array(rows))
    assert fleiss_kappa(rows) == pytest.approx(sm_fleiss(table))


def test_fleiss_rejects_uneven_raters():
    with pytest.raises(ValueError, match="same number of raters"):
        fleiss_kappa([["a", "b"], ["a"]])


def test_wss_binary_and_ranked():
    gold = [True] * 10 + [False] * 90
    # Perfect decisions: 90 of 100 not read, nothing missed -> 0.90
    assert wss(gold, gold) == pytest.approx(0.90)
    # Perfect ranking: 95% recall = 10 relevant (ceil 9.5) read first -> (100-10)/100 - 0.05
    scores = [1.0] * 10 + [0.0] * 90
    assert wss_at_recall(gold, scores) == pytest.approx(0.85)
    # Worst ranking: relevant last -> must read all 100 -> -0.05
    assert wss_at_recall(gold, [0.0] * 10 + [1.0] * 90) == pytest.approx(-0.05)


def test_multilabel_worked_example():
    gold = [{"MMSE"}, {"MMSE", "MoCA"}, {"IQCODE"}]
    pred = [{"MMSE"}, {"MMSE"}, {"IQCODE", "MoCA"}]
    r = multilabel_scores(gold, pred, ["MMSE", "MoCA", "Mini-Cog", "IQCODE"])
    assert r["per_label"]["MMSE"]["f1"] == 1.0
    assert r["per_label"]["MoCA"] == pytest.approx(
        {"precision": 0.0, "recall": 0.0, "f1": 0.0, "kappa": -0.5, "support": 1, "predicted": 1}
    )
    assert r["micro"]["precision"] == pytest.approx(3 / 4) and r["micro"]["recall"] == pytest.approx(3 / 4)
    assert r["macro"]["f1"] == pytest.approx((1 + 0 + 1) / 3)  # Mini-Cog has no support: left out
    assert r["exact_match"] == pytest.approx(1 / 3)


def test_usage_summary():
    u = [{"input_tokens": 1000, "output_tokens": 100, "cost_usd": 0.01, "seconds": 2}] * 4
    s = usage_summary(u, n_articles=2, wall_seconds=10, human_hours=1)
    assert s["total"]["input_tokens"] == 4000
    assert s["per_article"]["cost_usd"] == pytest.approx(0.02)
    assert s["speedup_vs_human"] == pytest.approx(360)


# ---- agreement with standard libraries on random data ----------------------


@pytest.mark.parametrize("seed", range(5))
def test_matches_sklearn_on_random_data(seed):
    rng = random.Random(seed)
    gold = [rng.random() < 0.3 for _ in range(300)]
    pred = [g if rng.random() < 0.8 else not g for g in gold]
    s = binary_scores(gold, pred)
    p, r, f, _ = precision_recall_fscore_support(gold, pred, average="binary", zero_division=0)
    assert (s.precision, s.recall, s.f1) == pytest.approx((p, r, f))
    assert cohen_kappa(gold, pred) == pytest.approx(cohen_kappa_score(gold, pred))


@pytest.mark.parametrize("seed", range(5))
def test_fleiss_matches_statsmodels_on_random_data(seed):
    rng = random.Random(seed)
    cats = ["include", "exclude", "unsure"]
    rows = [[rng.choice(cats) for _ in range(3)] for _ in range(200)]
    table, _ = aggregate_raters(np.array(rows))
    assert fleiss_kappa(rows) == pytest.approx(sm_fleiss(table))


# ---- evaluate(): the full report --------------------------------------------


def rec(pmid, screening, instrument=None, score=None, reviewers=None):
    labels = {"instrument": instrument} if instrument else {}
    return GoldRecord(pmid=pmid, screening=screening, labels=labels, score=score, reviewers=reviewers or {})


def test_evaluate_end_to_end():
    gold = [rec("1", "include", ["MMSE"]), rec("2", "include", ["MoCA"]), rec("3", "exclude"), rec("4", "exclude")]
    pred = [
        rec("1", "include", ["MMSE"], score=0.9),
        rec("2", "exclude", score=0.4),  # missed include: also costs annotation recall
        rec("3", "include", ["IQCODE"], score=0.6),
        rec("4", "exclude", score=0.1),
    ]
    r = evaluate(gold, pred, CODEBOOK)
    assert r["screening"]["recall"] == 0.5 and r["screening"]["precision"] == 0.5
    assert "wss_at_95" in r["screening"]
    inst = r["annotation"]["instrument"]
    assert inst["per_label"]["MMSE"]["recall"] == 1.0
    assert inst["per_label"]["MoCA"]["recall"] == 0.0
    assert r["annotation"].get("setting") is None  # gold has no setting labels here
    assert r["human_agreement"] is None


def test_evaluate_restricts_to_split_and_requires_predictions():
    gold = [rec("1", "include", ["MMSE"]), rec("2", "exclude")]
    pred = [rec("1", "include", ["MMSE"])]
    assert evaluate(gold, pred, CODEBOOK, pmids=["1"])["n_articles"] == 1
    with pytest.raises(ValueError, match="No prediction for 1"):
        evaluate(gold, pred, CODEBOOK)


def test_evaluate_reports_human_fleiss_when_reviewers_present():
    rv = lambda a, b, c: {"r1": {"screening": a}, "r2": {"screening": b}, "r3": {"screening": c}}  # noqa: E731
    gold = [
        rec("1", "include", reviewers=rv("include", "include", "include")),
        rec("2", "exclude", reviewers=rv("exclude", "exclude", "include")),
        rec("3", "exclude", reviewers=rv("exclude", "exclude", "exclude")),
    ]
    r = evaluate(gold, gold, CODEBOOK)
    h = r["human_agreement"]
    assert h["reviewers"] == ["r1", "r2", "r3"] and h["items"] == 3
    assert h["screening_fleiss_kappa"] == pytest.approx(
        fleiss_kappa([["include"] * 3, ["exclude", "exclude", "include"], ["exclude"] * 3])
    )


def test_gwet_ac1_stays_high_when_one_category_dominates():
    from adrd_review.eval.metrics import fleiss_kappa, gwet_ac1

    # 3 raters, 100 items: 90 unanimous 'include', 10 where one rater says 'exclude'
    rows = [["include"] * 3] * 90 + [["include", "include", "exclude"]] * 10
    assert fleiss_kappa(rows) < 0  # the kappa paradox: high agreement, negative kappa
    assert gwet_ac1(rows) > 0.9
    assert gwet_ac1([["a", "a"], ["b", "b"]]) == 1.0 and gwet_ac1([["a", "a"]]) == 1.0
