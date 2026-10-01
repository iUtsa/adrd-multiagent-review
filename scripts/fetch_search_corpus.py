"""Collect the papers found by the professor's PubMed search: abstracts + full text.

    python scripts/fetch_search_corpus.py            # uses the saved search results
    python scripts/fetch_search_corpus.py --rerun    # run the query on PubMed again first

For every article it downloads the title/abstract (data/corpus_search/) and then looks
for the full paper: the team's PDF (data/pdfs/), PubMed Central, or an open-access PDF.
Safe to stop and re-run: everything already downloaded is skipped.

Writes:
    data/search/fulltext_report.csv   one row per article: source, size, or why none
    data/search/pdfs_needed.csv       papers with no full text: link + file name to save as
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adrd_review.tools.fulltext import get_fulltext  # noqa: E402
from adrd_review.tools.pubmed import PubMedClient, fetch_articles, search  # noqa: E402

QUERY = "configs/search/prof_query.yaml"
PMIDS = Path("data/search/prof_query_pmids.txt")
CORPUS = "data/corpus_search"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true", help="Run the query on PubMed again")
    ap.add_argument("--limit", type=int, help="Only the first N articles (for a quick try)")
    ap.add_argument("--no-oa", action="store_true", help="Skip the open-access PDF lookup (faster)")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(".env")
    except ImportError:
        pass

    if args.rerun or not PMIDS.exists():
        ids, count = search(yaml.safe_load(Path(QUERY).read_text())["corrected"])
        PMIDS.parent.mkdir(parents=True, exist_ok=True)
        PMIDS.write_text("\n".join(ids) + "\n")
        print(f"Search: {count} articles")
    pmids = PMIDS.read_text().split()
    if args.limit:
        pmids = pmids[: args.limit]

    client = PubMedClient.from_env()
    articles, report = fetch_articles(pmids, cache_dir=CORPUS, client=client)
    print(f"Abstracts: {len(articles)} of {len(pmids)} downloaded "
          f"({len(report.missing)} missing, {len(report.no_abstract)} without an abstract)")

    rows, sources = [], Counter()
    for i, pmid in enumerate(pmids, start=1):
        art = articles.get(pmid)
        if art is None:
            rows.append({"pmid": pmid, "source": "", "chars": 0, "sections": 0, "reason": "article not found in PubMed", "link": ""})
            continue
        res = get_fulltext(art, allow_download=not args.no_oa, client=client)
        sources[res.source or "none"] += 1
        rows.append({
            "pmid": pmid,
            "year": art.year or "",
            "title": art.title,
            "doi": art.doi or "",
            "source": res.source or "",
            "chars": len(res.fulltext.text) if res.found else 0,
            "sections": len(res.fulltext.sections) if res.found else 0,
            "reason": res.reason,
            "link": res.link or "",
        })
        if i % 100 == 0 or i == len(pmids):
            print(f"  full text checked: {i}/{len(pmids)}   {dict(sources)}", flush=True)

    out = Path("data/search")
    fields = ["pmid", "year", "title", "doi", "source", "chars", "sections", "reason", "link"]
    with open(out / "fulltext_report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    needed = [r for r in rows if not r["source"]]
    with open(out / "pdfs_needed.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pmid", "year", "title", "doi", "link", "why_missing", "save_as"])
        for r in needed:
            w.writerow([r["pmid"], r.get("year", ""), r.get("title", ""), r.get("doi", ""), r["link"], r["reason"], f"data/pdfs/{r['pmid']}.pdf"])

    n = len(rows)
    have = n - len(needed)
    print(f"\nFull text found for {have} of {n} articles ({have / n:.0%}): {dict(sources)}")
    print(f"No full text for {len(needed)} ({len(needed) / n:.0%}): see {out / 'pdfs_needed.csv'}")
    print(f"Report: {out / 'fulltext_report.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
