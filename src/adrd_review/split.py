"""Freeze the calibration / test split of the gold standard.

Calibration articles are used for all tuning (prompts, codebook, thresholds).
Test articles are used once, for final results. A saved split is never
overwritten unless --force is given, so the test set cannot drift.

    python -m adrd_review.split --gold data/gold/gold.csv --out configs/splits/placeholder.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from adrd_review.codebook import DEFAULT_CODEBOOK, load_codebook
from adrd_review.gold import GoldRecord, file_sha256, load_gold

DEFAULT_SEED = 20260930


def make_split(
    records: list[GoldRecord],
    n_calibration: int = 40,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Stratified by screening decision, so both sets keep the same include rate."""
    if not 0 < n_calibration < len(records):
        raise ValueError(f"n_calibration must be between 1 and {len(records) - 1}")

    groups: dict[str, list[str]] = defaultdict(list)
    for r in records:
        groups[r.screening].append(r.pmid)

    # Largest-remainder allocation of calibration slots across groups.
    total = len(records)
    exact = {g: n_calibration * len(p) / total for g, p in groups.items()}
    quota = {g: int(x) for g, x in exact.items()}
    for g in sorted(exact, key=lambda g: exact[g] - quota[g], reverse=True):
        if sum(quota.values()) == n_calibration:
            break
        quota[g] += 1

    rng = random.Random(seed)
    calibration, test = [], []
    for g in sorted(groups):
        pmids = sorted(groups[g])
        rng.shuffle(pmids)
        calibration += pmids[: quota[g]]
        test += pmids[quota[g] :]

    return {
        "seed": seed,
        "stratified_by": "screening",
        "n_calibration": len(calibration),
        "n_test": len(test),
        "calibration": sorted(calibration, key=int),
        "test": sorted(test, key=int),
    }


def save_split(split: dict, path: str | Path, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists. The split is frozen; use --force only if you "
            "really mean to re-split (this invalidates earlier results)."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2) + "\n")


def load_split(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m adrd_review.split")
    p.add_argument("--gold", required=True, help="Gold file in the standard CSV format")
    p.add_argument("--out", required=True, help="Where to save the split JSON")
    p.add_argument("--codebook", default=str(DEFAULT_CODEBOOK))
    p.add_argument("--n-calibration", type=int, default=40)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--name", default=None, help="Label stored in the file")
    p.add_argument("--force", action="store_true", help="Overwrite an existing split")
    args = p.parse_args(argv)

    codebook = load_codebook(args.codebook)
    records = load_gold(args.gold, codebook)
    split = make_split(records, args.n_calibration, args.seed)
    split = {
        "name": args.name or Path(args.out).stem,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gold_file": args.gold,
        "gold_sha256": file_sha256(args.gold),
        "codebook_version": codebook.version,
        **split,
    }
    try:
        save_split(split, args.out, force=args.force)
    except FileExistsError as e:
        print(e, file=sys.stderr)
        return 1

    by_pmid = {r.pmid: r.screening for r in records}
    for part in ("calibration", "test"):
        inc = sum(by_pmid[x] == "include" for x in split[part])
        print(f"{part:12} {len(split[part]):4} articles  ({inc} include, {len(split[part]) - inc} exclude)")
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
