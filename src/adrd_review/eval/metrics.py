"""Evaluation metrics. Pure functions: no files, no network.

Each function is cross-checked against scikit-learn / statsmodels in
tests/test_metrics.py.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass


def _div(num: float, den: float) -> float:
    """Division that returns 0.0 for an empty denominator (same as sklearn zero_division=0)."""
    return num / den if den else 0.0


def _clean(x: float) -> float:
    """Round away float noise such as 1 - 0.95 - 0.05 = -1e-17 (prints as -0.000)."""
    return round(x, 12) + 0.0


def _check_pair(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("No items to score")


# --------------------------------------------------------------------------- #
# Binary decisions (screening: include = positive)
# --------------------------------------------------------------------------- #


@dataclass
class BinaryScores:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    specificity: float
    accuracy: float

    def as_dict(self) -> dict:
        return asdict(self)


def binary_scores(gold: Sequence[bool], pred: Sequence[bool]) -> BinaryScores:
    """Confusion counts plus precision, recall, F1, specificity, accuracy."""
    _check_pair(gold, pred)
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum(p and not g for g, p in zip(gold, pred))
    fn = sum(g and not p for g, p in zip(gold, pred))
    tn = len(gold) - tp - fp - fn
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    return BinaryScores(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=_div(2 * precision * recall, precision + recall),
        specificity=_div(tn, tn + fp),
        accuracy=_div(tp + tn, len(gold)),
    )


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Agreement between two raters beyond chance. 1 = perfect, 0 = chance level.

    If both raters give the same single label to everything, chance agreement is
    total; we return 1.0 when they agree perfectly (sklearn returns NaN here).
    """
    _check_pair(a, b)
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / n**2
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def fleiss_kappa(ratings: Sequence[Sequence[Hashable]]) -> float:
    """Agreement among a fixed number of raters (>= 2) over many items.

    `ratings` has one row per item, one entry per rater:
        [["include", "include", "exclude"], ["exclude", "exclude", "exclude"], ...]
    """
    if not ratings:
        raise ValueError("No items to score")
    n_raters = len(ratings[0])
    if n_raters < 2 or any(len(r) != n_raters for r in ratings):
        raise ValueError("Every item needs the same number of raters (at least 2)")
    n_items = len(ratings)
    categories = sorted({c for row in ratings for c in row}, key=str)
    counts = [[Counter(row)[c] for c in categories] for row in ratings]

    p_items = [(sum(x * x for x in row) - n_raters) / (n_raters * (n_raters - 1)) for row in counts]
    p_bar = sum(p_items) / n_items
    p_cat = [sum(row[j] for row in counts) / (n_items * n_raters) for j in range(len(categories))]
    p_e = sum(p * p for p in p_cat)
    if p_e == 1:
        return 1.0 if p_bar == 1 else 0.0
    return (p_bar - p_e) / (1 - p_e)


# --------------------------------------------------------------------------- #
# Multi-label annotation (e.g. instrument = {MMSE, MoCA})
# --------------------------------------------------------------------------- #


def multilabel_scores(
    gold: Sequence[set[str]],
    pred: Sequence[set[str]],
    labels: Sequence[str],
) -> dict:
    """Per-label, micro and macro precision/recall/F1, per-label kappa, exact match.

    Macro averages over labels that appear in the gold data (support > 0), so a
    label nobody used cannot drag the average to zero.
    """
    _check_pair(gold, pred)
    per_label: dict[str, dict] = {}
    tp = fp = fn = 0
    for label in labels:
        g = [label in s for s in gold]
        p = [label in s for s in pred]
        s = binary_scores(g, p)
        tp, fp, fn = tp + s.tp, fp + s.fp, fn + s.fn
        per_label[label] = {
            "precision": s.precision,
            "recall": s.recall,
            "f1": s.f1,
            "kappa": cohen_kappa(g, p),
            "support": s.tp + s.fn,
            "predicted": s.tp + s.fp,
        }
    present = [v for v in per_label.values() if v["support"] > 0]
    micro_p, micro_r = _div(tp, tp + fp), _div(tp, tp + fn)
    return {
        "per_label": per_label,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": _div(2 * micro_p * micro_r, micro_p + micro_r)},
        "macro": {
            k: _div(sum(v[k] for v in present), len(present)) for k in ("precision", "recall", "f1")
        },
        "exact_match": _div(sum(g == p for g, p in zip(gold, pred)), len(gold)),
        "n_items": len(gold),
    }


# --------------------------------------------------------------------------- #
# Work saved over sampling (screening workload)
# --------------------------------------------------------------------------- #


def wss(gold: Sequence[bool], pred: Sequence[bool]) -> float:
    """WSS for yes/no decisions: share of articles a human no longer reads,
    minus the share of relevant articles missed.  (TN + FN) / N - (1 - recall)."""
    s = binary_scores(gold, pred)
    return _clean((s.tn + s.fn) / len(gold) - (1 - s.recall))


def wss_at_recall(gold: Sequence[bool], scores: Sequence[float], recall: float = 0.95) -> float:
    """WSS@R for a ranking (Cohen et al. 2006). Articles are read from highest
    score down until `recall` of the relevant ones are found; WSS is the share
    left unread minus (1 - recall). Ties keep input order."""
    _check_pair(gold, scores)
    n_pos = sum(gold)
    if n_pos == 0:
        raise ValueError("No relevant articles in gold; WSS is undefined")
    target = math.ceil(recall * n_pos)
    order = sorted(range(len(gold)), key=lambda i: -scores[i])
    found = 0
    for read, i in enumerate(order, start=1):
        found += gold[i]
        if found >= target:
            return _clean((len(gold) - read) / len(gold) - (1 - recall))
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------- #
# Cost and time
# --------------------------------------------------------------------------- #


def usage_summary(
    usages: Sequence[dict],
    n_articles: int,
    wall_seconds: float | None = None,
    human_hours: float | None = None,
) -> dict:
    """Totals and per-article means of tokens, cost and time.

    `usages` holds one dict per model call with any of: input_tokens,
    output_tokens, cost_usd, seconds. Speed-up uses wall-clock time, since
    calls run in parallel.
    """
    fields = ("input_tokens", "output_tokens", "cost_usd", "seconds")
    totals = {f: sum(u.get(f, 0) or 0 for u in usages) for f in fields}
    out = {
        "calls": len(usages),
        "articles": n_articles,
        "total": totals,
        "per_article": {f: _div(totals[f], n_articles) for f in fields},
    }
    if wall_seconds is not None:
        out["wall_seconds"] = wall_seconds
        if human_hours:
            out["speedup_vs_human"] = _div(human_hours * 3600, wall_seconds)
    return out
