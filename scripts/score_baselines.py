"""Score simple non-AI baselines on the calibration set.

These are the floor every agent condition must beat:
    all_include  include everything (perfect recall, no work saved)
    all_exclude  exclude everything
    random       include with probability = calibration include rate (seeded)
    keyword      include if the title/abstract names a listed instrument AND dementia;
                 instrument labels = instruments named

    python scripts/score_baselines.py

Writes predictions to data/runs/baselines/ and prints a comparison table.
Only the calibration set is scored; the test set is reserved for final runs.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adrd_review.codebook import load_codebook  # noqa: E402
from adrd_review.eval.score import evaluate  # noqa: E402
from adrd_review.gold import GoldRecord, load_gold, write_gold  # noqa: E402
from adrd_review.split import load_split  # noqa: E402
from adrd_review.tools.pubmed import load_cached  # noqa: E402

INSTRUMENT_PATTERNS = {
    "MMSE": r"\bMMSE\b|mini[- ]mental state",
    "MoCA": r"\bMoCA\b|montreal cognitive assessment",
    "Mini-Cog": r"mini[- ]?cog\b",
    "IQCODE": r"\bIQCODE\b|informant questionnaire on cognitive decline",
}
DEMENTIA = r"dementia|alzheimer"
ACCURACY = r"sensitivity|specificity|accuracy|\bROC\b|\bAUC\b|cut[- ]?off|predictive value"


def keyword_prediction(pmid: str) -> GoldRecord:
    art = load_cached(pmid)
    text = f"{art.title}\n{art.abstract}" if art else ""
    found = sorted(k for k, pat in INSTRUMENT_PATTERNS.items() if re.search(pat, text, re.I))
    dementia = bool(re.search(DEMENTIA, text, re.I))
    accuracy = bool(re.search(ACCURACY, text, re.I))
    include = bool(found) and dementia
    score = 0.5 * bool(found) + 0.25 * dementia + 0.25 * accuracy
    return GoldRecord(
        pmid=pmid,
        screening="include" if include else "exclude",
        labels={"instrument": found} if include and found else {},
        score=score,
    )


def main() -> int:
    codebook = load_codebook()
    gold = load_gold("data/gold/gold.csv", codebook)
    cal = load_split("configs/splits/placeholder.json")["calibration"]
    gold_by = {g.pmid: g for g in gold}
    rate = sum(gold_by[p].screening == "include" for p in cal) / len(cal)

    rng = random.Random(20260930)
    baselines = {
        "all_include": [GoldRecord(pmid=p, screening="include", score=1.0) for p in cal],
        "all_exclude": [GoldRecord(pmid=p, screening="exclude", score=0.0) for p in cal],
        "random": [
            GoldRecord(pmid=p, screening="include" if (x := rng.random()) < rate else "exclude", score=x)
            for p in cal
        ],
        "keyword": [keyword_prediction(p) for p in cal],
    }

    out_dir = Path("data/runs/baselines")
    header = f"{'baseline':12} {'recall':>7} {'prec':>7} {'F1':>7} {'kappa':>7} {'WSS':>7} {'WSS@95':>7} {'instr F1':>9}"
    print(f"Calibration set: {len(cal)} articles, {round(rate * len(cal))} gold includes\n")
    print(header)
    print("-" * len(header))
    for name, preds in baselines.items():
        path = out_dir / f"{name}.csv"
        write_gold(preds, path, dimensions=list(codebook.dimensions))
        r = evaluate(gold, load_gold(path, codebook), codebook, cal)
        s = r["screening"]
        instr = r["annotation"].get("instrument", {}).get("micro", {}).get("f1", 0.0)
        print(
            f"{name:12} {s['recall']:7.3f} {s['precision']:7.3f} {s['f1']:7.3f} {s['cohen_kappa']:7.3f} "
            f"{s['wss']:7.3f} {s.get('wss_at_95', float('nan')):7.3f} {instr:9.3f}"
        )
    print(f"\nPredictions saved in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
