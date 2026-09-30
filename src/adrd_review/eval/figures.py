"""Figures for the paper and slides (task 10.3). Each figure is saved as PNG + SVG,
with a CSV of the exact numbers plotted (the table view of the chart).

    python -m adrd_review.eval.figures conditions --part calibration
    python -m adrd_review.eval.figures effort data/runs/E3_gemini/<timestamp> [--mark data/runs/E4_gemini/<timestamp>]

Style follows the project's data-viz rules: validated colour-blind-safe palette
(first three slots, checked all-pairs), one y-axis, thin marks with a surface gap,
solid hairline grid, legend for every multi-series chart, and a table twin.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from adrd_review.codebook import load_codebook  # noqa: E402
from adrd_review.eval.aggregate import CONDITION_ORDER  # noqa: E402
from adrd_review.eval.metrics import binary_scores, cohen_kappa  # noqa: E402
from adrd_review.gold import load_gold  # noqa: E402

# Validated palette (docs/EVALUATION.md, "Figures"): slots 1-3 pass all-pairs CVD checks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _style(ax, title: str, subtitle: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=24 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=INK2)


def _save(fig, out: Path, rows: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Figure 1: conditions compared
# --------------------------------------------------------------------------- #


def conditions_figure(summary_csv: Path, out: Path) -> Path:
    rows = list(csv.DictReader(open(summary_csv)))
    order = {c: i for i, c in enumerate(CONDITION_ORDER)}
    rows.sort(key=lambda r: (order.get(r["condition"], 99), r["name"]))
    metrics = [("recall", "Recall"), ("precision", "Precision"), ("kappa", "Cohen's κ")]
    labels = [f"{r['condition']}\n{r['name']}" for r in rows]

    fig, ax = plt.subplots(figsize=(max(6, 1.35 * len(rows) + 2), 4.2))
    width = 0.26
    table = []
    for i, (key, name) in enumerate(metrics):
        means = [float(r[f"{key}_mean"] or 0) for r in rows]
        sds = [float(r[f"{key}_sd"]) if r.get(f"{key}_sd") else 0.0 for r in rows]
        xs = [j + (i - 1) * width for j in range(len(rows))]
        ax.bar(xs, means, width, color=SERIES[i], label=name, edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
        for r, m, s in zip(rows, means, sds):
            table.append({"condition": r["condition"], "config": r["name"], "metric": name, "mean": m, "sd": s,
                          "runs": r["runs"]})
    ax.set_xticks(range(len(rows)), labels)
    lo = min(0.0, min(t["mean"] - t["sd"] for t in table) - 0.05)
    ax.set_ylim(lo, 1.0)
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    part = summary_csv.parent.name
    n = rows[0]["articles"] if rows else "?"
    _style(ax, "Screening performance by condition", f"{part} set, {n} articles; bars = mean over repeats, whiskers = SD")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3, frameon=False, fontsize=9, labelcolor=INK2)
    if any("fake" in r["name"] for r in rows):
        ax.text(1, -0.3, "Keyword stand-in model: plumbing check, not results", transform=ax.transAxes,
                ha="right", fontsize=8, color=MUTED)
    _save(fig, out, table)
    return out


# --------------------------------------------------------------------------- #
# Figure 2: accuracy vs human effort
# --------------------------------------------------------------------------- #


def effort_curve(run_dir: Path) -> list[dict]:
    """Send the least certain articles to a human first (gold answer), one at a time.

    Certainty = distance of the article's include score from 0.5. Returns one row
    per number of articles reviewed.
    """
    meta = json.loads((run_dir / "run_meta.json").read_text())
    gold = {g.pmid: g.screening == "include" for g in load_gold(meta["config"]["gold"], load_codebook(meta["config"]["codebook"]))}
    preds = list(csv.DictReader(open(run_dir / "predictions.csv")))
    for p in preds:
        p["inc"] = p["screening"] == "include"
        p["score"] = float(p["score"]) if p.get("score") not in (None, "") else (1.0 if p["inc"] else 0.0)
    order = sorted(preds, key=lambda p: (abs(p["score"] - 0.5), p["pmid"]))
    truth = [gold[p["pmid"]] for p in order]
    rows = []
    for k in range(len(order) + 1):
        decided = [truth[i] if i < k else order[i]["inc"] for i in range(len(order))]
        s = binary_scores(truth, decided)
        rows.append({"reviewed": k, "reviewed_share": round(k / len(order), 4), "recall": round(s.recall, 4),
                     "precision": round(s.precision, 4), "f1": round(s.f1, 4), "kappa": round(cohen_kappa(truth, decided), 4)})
    return rows


def effort_figure(run_dir: Path, out: Path, mark: Path | None = None) -> Path:
    rows = effort_curve(run_dir)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = [100 * r["reviewed_share"] for r in rows]
    for i, (key, name) in enumerate([("f1", "F1"), ("recall", "Recall")]):
        ax.plot(x, [r[key] for r in rows], color=SERIES[i], linewidth=2, label=name, zorder=2)
    if mark is not None:
        rep = json.loads((mark / "report.json").read_text())
        mx = 100 * rep["escalations"]["rate"]
        for i, key in enumerate(["f1", "recall"]):
            ax.scatter([mx], [rep["screening"][key]], s=64, color=SERIES[i], edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(f"E4: {mx:.0f}% of articles to a human", (mx, rep["screening"]["f1"]), xytext=(8, -18),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Articles reviewed by a human, least certain first (%)", fontsize=9, color=INK2)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    _style(ax, "Accuracy vs human effort", f"{meta['name']} on {meta['part']} ({len(rows) - 1} articles); human answer = gold label")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    if "fake" in meta["name"]:
        ax.text(1, -0.22, "Keyword stand-in model: plumbing check, not results", transform=ax.transAxes,
                ha="right", fontsize=8, color=MUTED)
    _save(fig, out, rows)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adrd_review.eval.figures")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("conditions", help="Bar chart of all conditions (from eval.aggregate output)")
    c.add_argument("--part", default="calibration")
    c.add_argument("--results", default="data/results")
    e = sub.add_parser("effort", help="Accuracy as more articles go to a human")
    e.add_argument("run_dir")
    e.add_argument("--mark", help="An E4 run folder to mark its actual escalation rate")
    e.add_argument("--results", default="data/results")
    args = ap.parse_args(argv)
    if args.cmd == "conditions":
        base = Path(args.results) / args.part
        out = conditions_figure(base / "summary.csv", base / "fig_conditions")
    else:
        run_dir = Path(args.run_dir)
        meta = json.loads((run_dir / "run_meta.json").read_text())
        out = effort_figure(run_dir, Path(args.results) / meta["part"] / f"fig_effort_{meta['name']}",
                            Path(args.mark) if args.mark else None)
    print(f"Saved: {out}.png, .svg, .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
