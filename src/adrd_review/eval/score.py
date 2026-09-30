"""Score a predictions file against the gold standard.

Predictions use the same CSV format as the gold file (docs/GOLD_STANDARD.md),
optionally with a `score` column (0-1 likelihood of include) for WSS@95.

    python -m adrd_review.eval.score --pred data/runs/E1/predictions.csv
    python -m adrd_review.eval.score --pred ... --part test --json report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from adrd_review.codebook import DEFAULT_CODEBOOK, Codebook, load_codebook
from adrd_review.eval.metrics import (
    binary_scores,
    cohen_kappa,
    fleiss_kappa,
    multilabel_scores,
    wss,
    wss_at_recall,
)
from adrd_review.gold import GoldRecord, load_gold
from adrd_review.split import load_split

DEFAULT_GOLD = "data/gold/gold.csv"
DEFAULT_SPLIT = "configs/splits/placeholder.json"


def human_agreement(gold: list[GoldRecord]) -> dict | None:
    """Fleiss' kappa among the human reviewers, if the gold file has their answers."""
    reviewers = sorted({r for g in gold for r in g.reviewers})
    if len(reviewers) < 2:
        return None
    rows = [
        [g.reviewers[r]["screening"] for r in reviewers]
        for g in gold
        if all("screening" in g.reviewers.get(r, {}) for r in reviewers)
    ]
    if not rows:
        return None
    return {"reviewers": reviewers, "items": len(rows), "screening_fleiss_kappa": fleiss_kappa(rows)}


def evaluate(
    gold: list[GoldRecord],
    pred: list[GoldRecord],
    codebook: Codebook,
    pmids: Iterable[str] | None = None,
) -> dict:
    """Compare predictions with gold on `pmids` (default: all gold articles).

    Annotation is scored on the articles the gold standard includes. If the
    agents excluded one of those, its predicted labels count as empty, so a
    screening miss also costs annotation recall.
    """
    gold_by = {g.pmid: g for g in gold}
    pred_by = {p.pmid: p for p in pred}
    ids = list(pmids) if pmids is not None else list(gold_by)
    unknown = [i for i in ids if i not in gold_by]
    if unknown:
        raise ValueError(f"{len(unknown)} PMIDs are not in the gold file: {unknown[:5]}")
    missing = [i for i in ids if i not in pred_by]
    if missing:
        raise ValueError(f"No prediction for {len(missing)} articles: {missing[:5]}")

    g_inc = [gold_by[i].screening == "include" for i in ids]
    p_inc = [pred_by[i].screening == "include" for i in ids]
    screening = binary_scores(g_inc, p_inc).as_dict()
    screening["cohen_kappa"] = cohen_kappa(g_inc, p_inc)
    screening["wss"] = wss(g_inc, p_inc)
    scores = [pred_by[i].score for i in ids]
    if all(s is not None for s in scores) and any(g_inc):
        screening["wss_at_95"] = wss_at_recall(g_inc, scores, 0.95)

    included = [i for i in ids if gold_by[i].screening == "include"]
    annotation = {}
    for dim, spec in codebook.dimensions.items():
        if not included:
            break
        gold_sets = [set(gold_by[i].labels.get(dim, [])) for i in included]
        if not any(gold_sets):
            continue  # this gold file has no labels for this dimension
        pred_sets = [set(pred_by[i].labels.get(dim, [])) for i in included]
        annotation[dim] = multilabel_scores(gold_sets, pred_sets, spec.label_ids)

    return {
        "n_articles": len(ids),
        "gold_includes": sum(g_inc),
        "predicted_includes": sum(p_inc),
        "screening": screening,
        "annotation": annotation,
        "human_agreement": human_agreement([gold_by[i] for i in ids]),
        "codebook_version": codebook.version,
    }


def format_report(report: dict, title: str = "") -> str:
    s = report["screening"]
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    lines += [
        f"Articles: {report['n_articles']}   gold includes: {report['gold_includes']}   "
        f"predicted includes: {report['predicted_includes']}",
        "",
        "SCREENING (include = positive)",
        f"  recall      {s['recall']:.3f}   ({s['tp']} of {s['tp'] + s['fn']} gold includes found)",
        f"  precision   {s['precision']:.3f}   ({s['tp']} of {s['tp'] + s['fp']} predicted includes correct)",
        f"  F1          {s['f1']:.3f}",
        f"  specificity {s['specificity']:.3f}",
        f"  Cohen kappa {s['cohen_kappa']:.3f}",
        f"  WSS         {s['wss']:.3f}",
    ]
    if "wss_at_95" in s:
        lines.append(f"  WSS@95      {s['wss_at_95']:.3f}")
    for dim, a in report["annotation"].items():
        lines += [
            "",
            f"ANNOTATION: {dim} (on gold includes)",
            f"  micro F1 {a['micro']['f1']:.3f}   macro F1 {a['macro']['f1']:.3f}   exact match {a['exact_match']:.3f}",
            f"  {'label':28} {'P':>6} {'R':>6} {'F1':>6} {'kappa':>6} {'gold n':>7}",
        ]
        for label, v in a["per_label"].items():
            if v["support"] or v["predicted"]:
                lines.append(
                    f"  {label:28} {v['precision']:6.3f} {v['recall']:6.3f} {v['f1']:6.3f} "
                    f"{v['kappa']:6.3f} {v['support']:7}"
                )
    h = report["human_agreement"]
    lines += ["", "HUMAN AGREEMENT"]
    lines.append(
        f"  Fleiss kappa among {len(h['reviewers'])} reviewers: {h['screening_fleiss_kappa']:.3f} ({h['items']} items)"
        if h
        else "  not available (gold file has no per-reviewer answers)"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adrd_review.eval.score")
    ap.add_argument("--pred", required=True, help="Predictions CSV (gold format)")
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--codebook", default=str(DEFAULT_CODEBOOK))
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--part", choices=["calibration", "test", "all"], default="calibration")
    ap.add_argument("--json", help="Also save the full report as JSON")
    args = ap.parse_args(argv)

    codebook = load_codebook(args.codebook)
    gold = load_gold(args.gold, codebook)
    pred = load_gold(args.pred, codebook)
    pmids = None if args.part == "all" else load_split(args.split)[args.part]
    if args.part == "test":
        print("NOTE: scoring on the TEST set. Only do this for final, frozen runs.\n", file=sys.stderr)

    report = evaluate(gold, pred, codebook, pmids)
    report["part"] = args.part
    print(format_report(report, f"{Path(args.pred).name} on {args.part}"))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nSaved: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
