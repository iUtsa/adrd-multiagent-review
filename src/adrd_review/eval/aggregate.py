"""Combine finished runs into one results table: mean ± SD over repeats, per condition.

    python -m adrd_review.eval.aggregate                       # calibration runs
    python -m adrd_review.eval.aggregate --part test           # final runs

Only runs that are directly comparable are averaged: same run config, codebook,
gold file, article set and agent code (prompt fingerprint). For each config the
most recent such group is used, so older runs made before a code change are
never mixed in. Quick checks (--limit) and unfinished runs are skipped.

Writes data/results/<part>/summary.csv, summary.md and runs_used.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from adrd_review.eval.metrics import cohen_kappa

CONDITION_ORDER = ["E0", "E1", "E2", "E3", "E4", "D"]

# (column name, how to read it from report.json)
METRICS: list[tuple[str, callable]] = [
    ("recall", lambda r: r["screening"]["recall"]),
    ("precision", lambda r: r["screening"]["precision"]),
    ("f1", lambda r: r["screening"]["f1"]),
    ("specificity", lambda r: r["screening"]["specificity"]),
    ("kappa", lambda r: r["screening"]["cohen_kappa"]),
    ("wss", lambda r: r["screening"]["wss"]),
    ("wss95", lambda r: r["screening"].get("wss_at_95")),
    ("ft_recall", lambda r: r["fulltext"]["recall"]),
    ("ft_precision", lambda r: r["fulltext"]["precision"]),
    ("ft_f1", lambda r: r["fulltext"]["f1"]),
    ("ft_kappa", lambda r: r["fulltext"]["cohen_kappa"]),
    ("panel_fleiss", lambda r: ((r.get("panel_agreement") or {}).get("screening") or {}).get("fleiss_kappa")),
    ("escalation_rate", lambda r: (r.get("escalations") or {}).get("rate")),
    ("adjudications", lambda r: r.get("adjudications")),
    ("calls", lambda r: r["usage"]["calls"]),
    ("tokens_in_per_article", lambda r: r["usage"]["per_article"]["input_tokens"]),
    ("tokens_out_per_article", lambda r: r["usage"]["per_article"]["output_tokens"]),
    ("cost_usd", lambda r: r["usage"]["total"]["cost_usd"] or None),
    ("wall_seconds", lambda r: r["usage"].get("wall_seconds")),
    ("failures", lambda r: r.get("failures")),
]
MAIN = ["recall", "precision", "f1", "kappa", "wss95", "escalation_rate"]


def _get(fn, report):
    try:
        v = fn(report)
    except (KeyError, TypeError):
        return None
    return None if v is None else float(v)


def fingerprint(meta: dict) -> str:
    parts = [
        json.dumps(meta.get("config"), sort_keys=True),
        meta.get("codebook_sha256"),
        meta.get("gold_sha256"),
        meta.get("prompts_sha256"),
        json.dumps(sorted(meta.get("pmids", []))),
        meta.get("part"),
    ]
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


def find_runs(root: Path, part: str, include_limited: bool = False) -> list[dict]:
    runs = []
    for meta_file in sorted(root.glob("*/*/run_meta.json")):
        run_dir = meta_file.parent
        report_file = run_dir / "report.json"
        if not report_file.exists():
            continue
        meta = json.loads(meta_file.read_text())
        report = json.loads(report_file.read_text())
        if meta.get("part") != part or report.get("pending_human") or "screening" not in report:
            continue
        if meta.get("limit") and not include_limited:
            continue
        runs.append({"dir": run_dir, "meta": meta, "report": report, "fp": fingerprint(meta)})
    return runs


def latest_groups(runs: list[dict]) -> dict[str, list[dict]]:
    """For each config name, the most recent group of comparable runs."""
    by_name: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        by_name[r["meta"]["name"]][r["fp"]].append(r)
    out = {}
    for name, groups in by_name.items():
        out[name] = max(groups.values(), key=lambda g: max(r["meta"]["started_utc"] for r in g))
    return out


def run_to_run(group: list[dict]) -> dict:
    """Agreement between repeats on screening decisions (only meaningful with >= 2 runs)."""
    if len(group) < 2:
        return {"repeat_kappa": None, "includes_not_stable": None}
    preds = []
    for r in group:
        rows = list(csv.DictReader(open(r["dir"] / "predictions.csv")))
        preds.append({row["pmid"]: row["screening"] for row in rows})
    pmids = sorted(set.intersection(*(set(p) for p in preds)))
    kappas = [cohen_kappa([a[x] for x in pmids], [b[x] for x in pmids]) for a, b in combinations(preds, 2)]
    unstable = sum(len({p[x] for p in preds}) > 1 for x in pmids)
    return {"repeat_kappa": round(statistics.mean(kappas), 4), "includes_not_stable": unstable}


def summarize(groups: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for name, group in groups.items():
        meta = group[0]["meta"]
        row: dict = {"name": name, "condition": meta.get("condition", ""), "runs": len(group),
                     "articles": len(meta.get("pmids", [])), "codebook": meta.get("codebook_version")}
        dims = sorted({d for r in group for d in r["report"].get("annotation", {})})
        metrics = METRICS + [(f"{d}_micro_f1", (lambda d: lambda r: r["annotation"][d]["micro"]["f1"])(d)) for d in dims]
        for col, fn in metrics:
            vals = [v for v in (_get(fn, r["report"]) for r in group) if v is not None]
            row[f"{col}_mean"] = round(statistics.mean(vals), 4) if vals else None
            row[f"{col}_sd"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else None
        row |= run_to_run(group)
        rows.append(row)
    order = {c: i for i, c in enumerate(CONDITION_ORDER)}
    return sorted(rows, key=lambda r: (order.get(r["condition"], 99), r["name"]))


def fmt(mean, sd) -> str:
    if mean is None:
        return "–"
    return f"{mean:.3f}" + (f" ± {sd:.3f}" if sd is not None else "")


def to_markdown(rows: list[dict], part: str) -> str:
    dims = sorted({k[: -len("_micro_f1_mean")] for r in rows for k in r if k.endswith("_micro_f1_mean")})
    cols = MAIN + [f"{d}_micro_f1" for d in dims]
    head = ["Condition", "Config", "Runs"] + [c.replace("_micro_f1", " F1") for c in cols] + ["Repeat κ", "Cost $"]
    lines = [f"# Results: {part} set", "", "Mean ± SD over repeats. Screening: include = positive.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["condition"], r["name"], str(r["runs"])]
        cells += [fmt(r.get(f"{c}_mean"), r.get(f"{c}_sd")) for c in cols]
        cells += [fmt(r.get("repeat_kappa"), None), fmt(r.get("cost_usd_mean"), r.get("cost_usd_sd"))]
        lines.append("| " + " | ".join(cells) + " |")
    if any("fake" in r["name"] for r in rows):
        lines += ["", "Rows ending in `_fake` used the keyword stand-in model: plumbing checks, not results."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adrd_review.eval.aggregate")
    ap.add_argument("--runs", default="data/runs")
    ap.add_argument("--part", default="calibration", choices=["calibration", "test", "all"])
    ap.add_argument("--out", default="data/results")
    ap.add_argument("--include-limited", action="store_true", help="Also use --limit quick-check runs")
    args = ap.parse_args(argv)

    runs = find_runs(Path(args.runs), args.part, args.include_limited)
    if not runs:
        print(f"No finished {args.part} runs found under {args.runs}.")
        return 1
    groups = latest_groups(runs)
    rows = summarize(groups)

    out = Path(args.out) / args.part
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.md").write_text(to_markdown(rows, args.part))
    with open(out / "runs_used.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "run_folder", "repeat", "fingerprint", "started_utc"])
        for name, group in groups.items():
            for r in group:
                w.writerow([name, str(r["dir"]), r["meta"].get("repeat", 0), r["fp"], r["meta"]["started_utc"]])
    print(to_markdown(rows, args.part))
    print(f"Saved: {out}/summary.csv, summary.md, runs_used.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
