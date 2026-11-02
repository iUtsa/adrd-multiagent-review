"""Phase 1 vs phase 2: a blind human check of the papers the two runs decided differently.

Papers both runs decided the same way cannot tell them apart, so a person judges only the papers one run included
and the other excluded. On each judged paper exactly one run is right; the run right more often is the more
accurate one, and the difference is tested with an exact two-sided sign test (McNemar's test on paired decisions).

    python -m adrd_review.eval.tiebreak <phase 1 decisions.csv> <phase 2 final.csv> <out dir>            # the sheet
    python -m adrd_review.eval.tiebreak <phase 1 decisions.csv> <phase 2 final.csv> <out dir> --score <filled xlsx|csv>

Writes <out>/phase_check.xlsx (+ .csv, blind), phase_check_answer_key.csv, and phase_check_score.json when scored.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from math import comb
from pathlib import Path

from adrd_review.eval.accuracy import wilson
from adrd_review.final_results import CRITERIA, write_blind_workbook

DECIDED = ("included", "excluded")
DECISION = "human decision (include / exclude)"


def disagreements(phase1: Path, phase2: Path) -> list[dict]:
    p1 = {r["pmid"]: r for r in csv.DictReader(phase1.open())}
    out = []
    for r in csv.DictReader(phase2.open()):
        a = p1.get(r["pmid"])
        if a and a["outcome"] in DECIDED and r["outcome"] in DECIDED and a["outcome"] != r["outcome"]:
            out.append({"p1": a, "p2": r})
    return out


def _outcome(r: dict) -> str:
    return f"excluded ({r['stage']})" if r["outcome"] == "excluded" else r["outcome"]


def make_sheet(pairs: list[dict], out: Path, abstracts: dict[str, str], seed: int = 20261102) -> list[dict]:
    pairs = sorted(pairs, key=lambda x: x["p1"]["pmid"])
    random.Random(seed).shuffle(pairs)
    rows = [{"#": i, "PMID": x["p1"]["pmid"], "PubMed link": x["p1"]["link"], "title": x["p1"]["title"],
             "year": x["p1"]["year"], "abstract": abstracts.get(x["p1"]["pmid"], ""), DECISION: "",
             "human reason (rule, e.g. I1, I4, E1)": "", "notes": "",
             "phase 1 outcome": _outcome(x["p1"]), "phase 1 reason": x["p1"]["reason"],
             "phase 2 outcome": _outcome(x["p2"]), "phase 2 reason": x["p2"]["reason"],
             "phase 2 votes": " | ".join(f"{k[-1]}: {x['p2'][k]}" for k in ("reviewer A", "reviewer B", "reviewer C"))}
            for i, x in enumerate(pairs, 1)]
    out.mkdir(parents=True, exist_ok=True)
    blind = [c for c in rows[0] if not c.startswith("phase ")]
    with (out / "phase_check.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=blind, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with (out / "phase_check_answer_key.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["#", "PMID", "title"] + [c for c in rows[0] if c.startswith("phase ")],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    write_blind_workbook(rows, out / "phase_check.xlsx", "phase 1 outcome", [
        f"{len(rows)} papers, in random order: the papers where two AI screening runs reached opposite decisions "
        "(one included the paper, the other excluded it). Free full text only.",
        "For each paper, decide from the abstract and, if needed, the full paper (PubMed link) whether it belongs in "
        "the review under your criteria: include or exclude, and the main rule. 'unsure' is fine.",
        "The two runs' decisions are in the hidden columns on the right (click the + above them) and in "
        "phase_check_answer_key.csv: look only after deciding.",
        CRITERIA,
    ])
    return rows


def sign_test(k: int, n: int) -> float:
    """Exact two-sided binomial test of k successes in n against 1/2."""
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def score(filled: Path, key: Path, decided_by_both: int | None = None) -> dict:
    if filled.suffix == ".xlsx":
        from openpyxl import load_workbook

        ws = load_workbook(filled)["check (blind)"]
        head = [c.value for c in ws[1]]
        rows = [dict(zip(head, r)) for r in ws.iter_rows(min_row=2, values_only=True)]
    else:
        rows = list(csv.DictReader(filled.open()))
    ans = {str(r["#"]): r for r in csv.DictReader(key.open())}
    p1 = p2 = 0
    for r in rows:
        d = str(r.get(DECISION) or "").strip().lower()
        if d in ("include", "exclude"):
            p2_included = ans[str(r["#"])]["phase 2 outcome"] == "included"  # else phase 1 included it
            if p2_included == (d == "include"):
                p2 += 1
            else:
                p1 += 1
    n = p1 + p2
    res = {"judged": n, "unsure_or_blank": len(rows) - n, "phase 1 right": p1, "phase 2 right": p2,
           "phase 2 share right": round(p2 / n, 3) if n else None, "ci95": wilson(p2, n),
           "sign_test_p": round(sign_test(p2, n), 4),
           "better": None if not n or sign_test(p2, n) >= 0.05 else ("phase 2" if p2 > p1 else "phase 1")}
    if decided_by_both and n:
        # On the papers both runs decided alike they are equally right or wrong, so the accuracy gap over all of
        # them is the net number of wins, scaled up from the share of disagreements judged.
        res["accuracy_gap_phase2_minus_phase1"] = round((p2 - p1) / n * len(ans) / decided_by_both, 4)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adrd_review.eval.tiebreak")
    ap.add_argument("phase1")
    ap.add_argument("phase2")
    ap.add_argument("out")
    ap.add_argument("--score", default=None, help="the filled phase_check.xlsx or .csv")
    ap.add_argument("--corpus", default="data/corpus_search")
    args = ap.parse_args(argv)
    p1, p2, out = Path(args.phase1), Path(args.phase2), Path(args.out)
    if args.score:
        p2_out = {r["pmid"]: r["outcome"] for r in csv.DictReader(p2.open())}
        both = sum(1 for r in csv.DictReader(p1.open()) if r["outcome"] in DECIDED and p2_out.get(r["pmid"]) in DECIDED)
        res = score(Path(args.score), out / "phase_check_answer_key.csv", both)
        (out / "phase_check_score.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps(res, indent=2))
        return 0
    from adrd_review.tools.pubmed import fetch_articles

    pairs = disagreements(p1, p2)
    arts, _ = fetch_articles([x["p1"]["pmid"] for x in pairs], cache_dir=args.corpus)
    rows = make_sheet(pairs, out, {p: a.abstract or "" for p, a in arts.items()})
    kinds = {}
    for r in rows:
        k = f"phase 1 {r['phase 1 outcome']} / phase 2 {r['phase 2 outcome']}"
        kinds[k] = kinds.get(k, 0) + 1
    print(f"{len(rows)} papers -> {out}/phase_check.xlsx")
    print(json.dumps(kinds, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
