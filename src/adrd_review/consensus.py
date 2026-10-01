"""Combine several reviewers' votes into one decision per field.

This is the E2 consensus step: unanimous answers are accepted, otherwise a
weighted majority decides. The adjudicator (Day 6) will take over split cases.

Weights come from the run config (default 1 per reviewer). They are meant to
be set from each reviewer's accuracy on the calibration set, never the test set.
"""

from __future__ import annotations

from collections import defaultdict

from adrd_review.eval.metrics import fleiss_kappa
from adrd_review.schemas import Decision, Vote


def latest_votes(votes: list[Vote], field: str | None = None) -> list[Vote]:
    """Each reviewer's most recent vote (highest debate round), optionally for one field."""
    latest: dict[str, Vote] = {}
    for v in sorted(votes, key=lambda v: v.round):
        if field is None or v.field == field:
            latest[v.reviewer_id] = v
    return [latest[r] for r in sorted(latest)]


def is_split(votes: list[Vote]) -> bool:
    return len({tuple(sorted(v.labels)) for v in votes}) > 1


def _weight(v: Vote, weights: dict[str, float]) -> float:
    return weights.get(v.reviewer_id, 1.0)


def _include_probability(v: Vote) -> float:
    conf = 0.5 if v.confidence is None else v.confidence
    return conf if v.labels == ["include"] else 1 - conf


def combine_screening(votes: list[Vote], weights: dict[str, float] | None = None) -> Decision:
    """Include/exclude from screening votes.

    - one vote: taken as is ("single")
    - all agree: "unanimous"
    - otherwise weighted majority ("majority"); an exact tie goes to include
      ("tie_include") so a split never silently drops a study
    `include_score` is the weighted mean of each reviewer's probability of include
    (its confidence if it voted include, 1 - confidence if exclude).
    """
    if not votes:
        raise ValueError("No screening votes to combine")
    weights = weights or {}
    total = sum(_weight(v, weights) for v in votes)
    include_w = sum(_weight(v, weights) for v in votes if v.labels == ["include"])
    share = include_w / total
    score = sum(_weight(v, weights) * _include_probability(v) for v in votes) / total

    if len(votes) == 1:
        label, method = votes[0].labels[0], "single"
    elif len({tuple(v.labels) for v in votes}) == 1:
        label, method = votes[0].labels[0], "unanimous"
    elif share > 0.5:
        label, method = "include", "majority"
    elif share < 0.5:
        label, method = "exclude", "majority"
    else:
        label, method = "include", "tie_include"

    backing = share if label == "include" else 1 - share
    if method == "single":
        backing = votes[0].confidence
    return Decision(
        field="screening",
        labels=[label],
        method=method,
        confidence=None if backing is None else round(backing, 4),
        include_score=round(score, 4),
    )


def combine_eligibility(votes: list[Vote], weights: dict[str, float] | None = None) -> Decision:
    """Full-text eligibility: include / exclude / unclear.

    Unanimous or a weighted majority (> half) decides; with no majority the answer is
    "unclear", which is sent to a human. `include_score` treats an unclear vote as 0.5.
    """
    if not votes:
        raise ValueError("No eligibility votes to combine")
    weights = weights or {}
    total = sum(_weight(v, weights) for v in votes)
    share: dict[str, float] = defaultdict(float)
    for v in votes:
        share[v.labels[0]] += _weight(v, weights) / total

    def p_include(v: Vote) -> float:
        return 0.5 if v.labels == ["unclear"] else _include_probability(v)

    score = sum(_weight(v, weights) * p_include(v) for v in votes) / total
    top = max(share, key=lambda k: share[k])
    if len(votes) == 1:
        label, method, conf = votes[0].labels[0], "single", votes[0].confidence
    elif len(share) == 1:
        label, method, conf = top, "unanimous", 1.0
    elif share[top] > 0.5:
        label, method, conf = top, "majority", share[top]
    else:
        label, method, conf = "unclear", "no_majority", share[top]
    return Decision(
        field="eligibility",
        labels=[label],
        method=method,
        confidence=None if conf is None else round(conf, 4),
        include_score=round(score, 4),
    )


def combine_labels(
    votes: list[Vote],
    field: str,
    multi_label: bool,
    label_order: list[str],
    weights: dict[str, float] | None = None,
) -> Decision:
    """Labels for one annotation dimension.

    Multi-label: keep each label backed by at least half of the reviewer weight.
    Single-label: the label (or "no label") with the most weight; ties go to
    the label listed first in the codebook, and a label beats "no label".
    """
    if not votes:
        raise ValueError(f"No votes for {field!r}")
    weights = weights or {}
    total = sum(_weight(v, weights) for v in votes)

    if len(votes) == 1:
        v = votes[0]
        return Decision(field=field, labels=v.labels, method="single", confidence=v.confidence)
    if len({tuple(sorted(v.labels)) for v in votes}) == 1:
        return Decision(field=field, labels=votes[0].labels, method="unanimous", confidence=1.0)

    support: dict[str, float] = defaultdict(float)
    for v in votes:
        for label in set(v.labels):
            support[label] += _weight(v, weights)
    rank = {label: i for i, label in enumerate(label_order)}

    if multi_label:
        kept = sorted((x for x, w in support.items() if w / total >= 0.5), key=lambda x: rank.get(x, 99))
        backing = [support[x] / total for x in kept]
        conf = sum(backing) / len(backing) if backing else 1 - max(support.values(), default=0) / total
        return Decision(field=field, labels=kept, method="majority", confidence=round(conf, 4))

    empty_w = sum(_weight(v, weights) for v in votes if not v.labels)
    options = [(w, -rank.get(x, 99), x) for x, w in support.items()]
    best = max(options) if options else None
    if best is None or empty_w > best[0]:
        return Decision(field=field, labels=[], method="majority", confidence=round(empty_w / total, 4))
    return Decision(field=field, labels=[best[2]], method="majority", confidence=round(best[0] / total, 4))


def panel_agreement(states: list[dict], reviewer_ids: list[str], fields: list[str]) -> dict | None:
    """Fleiss' kappa among the reviewers, per field, from their independent votes.

    Only articles where every reviewer voted on the field are used. For
    annotation, each reviewer's whole label set counts as one category.
    """
    if len(reviewer_ids) < 2:
        return None
    out: dict = {"reviewers": reviewer_ids}
    for field in ["screening", *fields]:
        rows = []
        for s in states:
            votes = s.get("screening_votes", []) if field == "screening" else s.get("annotation_votes", [])
            by = {v.reviewer_id: "|".join(sorted(v.labels)) for v in votes if v.field == field and v.round == 0}
            if all(r in by for r in reviewer_ids):
                rows.append([by[r] for r in reviewer_ids])
        if rows:
            unanimous = sum(len(set(r)) == 1 for r in rows)
            out[field] = {"items": len(rows), "fleiss_kappa": round(fleiss_kappa(rows), 4), "unanimous": unanimous}
    return out
