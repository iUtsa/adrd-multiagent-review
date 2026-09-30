"""Reviewer weights from a CALIBRATION run (protocol step 5).

Each reviewer's weight comes from how often its independent (round 0) screening
vote matched the gold standard on the calibration set:

    weight = log(accuracy / (1 - accuracy)),  accuracy clipped to [0.55, 0.95]

This is the optimal weighting for a majority vote among independent voters of
known accuracy (Nitzan & Paroush, 1982). Clipping keeps one lucky or unlucky
calibration set from giving a reviewer all or nothing. Weights are rescaled to
average 1 so they read naturally.

    python -m adrd_review.eval.weights data/runs/E2_gemini/<timestamp>
    # writes configs/weights/E2_gemini.json; point a run config at it with
    #   weights_file: configs/weights/E2_gemini.json

Refuses test-set runs: weights must never be tuned on the test set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from adrd_review.codebook import load_codebook
from adrd_review.eval.metrics import cohen_kappa
from adrd_review.gold import load_gold

CLIP = (0.55, 0.95)


def log_odds(acc: float) -> float:
    a = min(max(acc, CLIP[0]), CLIP[1])
    return math.log(a / (1 - a))


def reviewer_weights(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    if meta["part"] != "calibration":
        raise SystemExit(f"Refusing: this run is on the {meta['part']!r} set. Weights come from calibration runs only.")
    cfg = meta["config"]
    gold = {g.pmid: g.screening for g in load_gold(cfg["gold"], load_codebook(cfg["codebook"]))}
    votes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for line in (run_dir / "decisions.jsonl").read_text().splitlines():
        d = json.loads(line)
        for v in d.get("screening_votes", []):
            if v["round"] == 0 and d["pmid"] in gold:
                votes[v["reviewer_id"]].append((gold[d["pmid"]], v["labels"][0]))
    if len(votes) < 2:
        raise SystemExit("Weights need a panel of at least 2 reviewers.")
    stats = {}
    for rid, pairs in sorted(votes.items()):
        acc = sum(g == p for g, p in pairs) / len(pairs)
        stats[rid] = {"articles": len(pairs), "accuracy": round(acc, 4),
                      "kappa": round(cohen_kappa([g for g, _ in pairs], [p for _, p in pairs]), 4),
                      "raw_weight": log_odds(acc)}
    mean = sum(s["raw_weight"] for s in stats.values()) / len(stats)
    for s in stats.values():
        s["weight"] = round(s.pop("raw_weight") / mean, 4)
    return {
        "source_run": str(run_dir),
        "run_name": meta["name"],
        "codebook_version": meta.get("codebook_version"),
        "method": "log-odds of calibration screening accuracy (clipped 0.55-0.95), rescaled to mean 1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reviewers": stats,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adrd_review.eval.weights")
    ap.add_argument("run_dir")
    ap.add_argument("--out-dir", default="configs/weights")
    args = ap.parse_args(argv)
    result = reviewer_weights(Path(args.run_dir))
    out = Path(args.out_dir) / f"{result['run_name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{'reviewer':9} {'accuracy':>9} {'kappa':>7} {'weight':>7}")
    for rid, s in result["reviewers"].items():
        print(f"{rid:9} {s['accuracy']:9.3f} {s['kappa']:7.3f} {s['weight']:7.3f}")
    print(f"\nSaved: {out}\nUse it by adding to a run config:  weights_file: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
