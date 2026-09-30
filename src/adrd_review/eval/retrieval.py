"""Retrieval check (task 9.3): how many gold-standard articles does the search find?

    python -m adrd_review.eval.retrieval                      # uses the saved search results
    python -m adrd_review.eval.retrieval --rerun              # re-runs the professor's query first

Reports recall over all gold articles and over gold includes, and for each missed
include a likely reason (published outside the query's date range, not English,
no abstract). Writes data/results/retrieval_check.json and retrieval_missed.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import yaml

from adrd_review.codebook import load_codebook
from adrd_review.gold import load_gold
from adrd_review.tools.pubmed import load_cached, search

DEFAULT_QUERY = "configs/search/prof_query.yaml"
DEFAULT_PMIDS = "data/search/prof_query_pmids.txt"


def query_limits(query: str) -> dict:
    """Date range and language restrictions written into a PubMed query."""
    out: dict = {}
    m = re.search(r'"(\d{4})/\d{2}/\d{2}"\[Date - Publication\]\s*:\s*"(\d{4})/\d{2}/\d{2}"', query)
    if m:
        out["years"] = (int(m.group(1)), int(m.group(2)))
    if "English[Language]" in query:
        out["language"] = "eng"
    return out


def likely_reason(pmid: str, limits: dict, corpus_dir: str) -> str:
    art = load_cached(pmid, corpus_dir)
    if art is None:
        return "article not downloaded"
    reasons = []
    if "years" in limits and art.year and not limits["years"][0] <= art.year <= limits["years"][1]:
        reasons.append(f"published {art.year}, outside {limits['years'][0]}-{limits['years'][1]}")
    if limits.get("language") and art.language and limits["language"] not in art.language:
        reasons.append(f"language {','.join(art.language)}")
    if not art.has_abstract:
        reasons.append("no abstract (fewer words for the search to match)")
    return "; ".join(reasons) or "search terms did not match the title/abstract/MeSH"


def check(gold_path: str, pmids_file: str, query: str, corpus_dir: str = "data/corpus") -> dict:
    gold = load_gold(gold_path, load_codebook())
    found = set(Path(pmids_file).read_text().split())
    limits = query_limits(query)
    includes = [g for g in gold if g.screening == "include"]
    missed = [g for g in includes if g.pmid not in found]
    return {
        "search_results": len(found),
        "gold_articles": len(gold),
        "gold_found": sum(g.pmid in found for g in gold),
        "recall_all": round(sum(g.pmid in found for g in gold) / len(gold), 4) if gold else None,
        "gold_includes": len(includes),
        "includes_found": len(includes) - len(missed),
        "recall_includes": round((len(includes) - len(missed)) / len(includes), 4) if includes else None,
        "query_limits": {k: list(v) if isinstance(v, tuple) else v for k, v in limits.items()},
        "missed_includes": [
            {"pmid": g.pmid, "title": (load_cached(g.pmid, corpus_dir) or type("x", (), {"title": ""})).title,
             "reason": likely_reason(g.pmid, limits, corpus_dir)}
            for g in missed
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adrd_review.eval.retrieval")
    ap.add_argument("--gold", default="data/gold/gold.csv")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--pmids", default=DEFAULT_PMIDS, help="Saved search results, one PMID per line")
    ap.add_argument("--rerun", action="store_true", help="Run the query on PubMed again first")
    ap.add_argument("--out", default="data/results")
    args = ap.parse_args(argv)

    query = yaml.safe_load(Path(args.query).read_text())["corrected"]
    if args.rerun:
        ids, count = search(query)
        Path(args.pmids).parent.mkdir(parents=True, exist_ok=True)
        Path(args.pmids).write_text("\n".join(ids) + "\n")
        print(f"Search re-run: {count} results")
    result = check(args.gold, args.pmids, query)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_check.json").write_text(json.dumps(result, indent=2) + "\n")
    with open(out / "retrieval_missed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pmid", "title", "reason"])
        w.writeheader()
        w.writerows(result["missed_includes"])

    print(f"Search results: {result['search_results']}")
    print(f"Gold articles found: {result['gold_found']} / {result['gold_articles']}  (recall {result['recall_all']})")
    print(f"Gold INCLUDES found: {result['includes_found']} / {result['gold_includes']}  (recall {result['recall_includes']})")
    reasons: dict[str, int] = {}
    for m in result["missed_includes"]:
        key = m["reason"].split(";")[0].split(",")[0] if "published" not in m["reason"] else "published outside the date range"
        reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        print("Missed includes, main reason:", reasons)
    print(f"Saved: {out / 'retrieval_check.json'}, {out / 'retrieval_missed.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
