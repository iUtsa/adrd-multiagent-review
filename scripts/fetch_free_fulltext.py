"""Find and download legal free copies of papers the run could not get a full text for.

For every paper in a run's pdfs_needed.csv, OpenAlex (free, no key) lists the paper's open-access locations:
publisher open access, repository copies, author manuscripts, preprints. Each PDF link is downloaded with an
honest User-Agent and kept only if it is the right paper: it must pass the PDF quality check (tools/pdf.py) and
contain most of the title's words. Kept files go to data/pdfs_oa/<PMID>.pdf, where the full-text source ladder
(tools/fulltext.py) finds them, so the next try sends them to the reviewers. No paywall or bot protection is
bypassed: a paper that is free only on a web page that blocks scripts is listed for a person to save by hand.

    python scripts/fetch_free_fulltext.py data/runs/PROF_search_free/<timestamp>
Writes <run>/free_fulltext_found.csv: PMID, outcome, where from, and the link to open by hand when not downloaded.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adrd_review.tools import pdf as pdf_tool  # noqa: E402

OA_DIR = Path("data/pdfs_oa")
UA = {"User-Agent": "adrd-review/1.0 (academic scoping review; open-access full text for text analysis)"}
STOP = {"the", "and", "for", "with", "from", "into", "using", "based", "of", "in", "on", "to", "a", "an", "by", "as", "at"}


def title_words(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", title.lower()) if len(w) > 3 and w not in STOP}


def is_the_paper(path: Path, title: str) -> tuple[bool, str]:
    try:
        text = pdf_tool.clean(pdf_tool.read_pdf(path))
    except Exception as e:
        return False, f"unreadable PDF ({type(e).__name__})"
    ok, why = pdf_tool.quality(text)
    if not ok:
        return False, f"failed quality check: {why}"
    words = title_words(title)
    found = sum(w in text.lower() for w in words)
    if words and found / len(words) < 0.7:
        return False, f"title words found {found}/{len(words)}: probably another paper"
    return True, "ok"


def locations(pmid: str, doi: str | None) -> list[dict]:
    for ident in ([f"pmid:{pmid}"] + ([f"doi:{doi}"] if doi else [])):
        r = requests.get(f"https://api.openalex.org/works/{ident}", params={"select": "open_access,locations"},
                         headers=UA, timeout=30)
        if r.ok:
            return [loc for loc in r.json().get("locations") or [] if loc.get("is_oa")]
    return []


def one(row: dict) -> dict:
    pmid, doi, title = row["pmid"], row.get("doi") or None, row["title"]
    dest, marker = OA_DIR / f"{pmid}.pdf", OA_DIR / f"{pmid}.none"
    out = {"pmid": pmid, "title": title, "doi": doi or "", "was": row["why_missing"]}
    if dest.exists():
        return out | {"outcome": "already downloaded", "source": "", "link": ""}
    try:
        locs = locations(pmid, doi)
    except requests.RequestException as e:
        return out | {"outcome": f"lookup failed ({type(e).__name__})", "source": "", "link": row.get("link", "")}
    if not locs:
        return out | {"outcome": "no free copy listed", "source": "", "link": row.get("link", "")}
    tried = []
    for loc in sorted(locs, key=lambda l: l.get("pdf_url") is None):
        url = loc.get("pdf_url")
        src = (loc.get("source") or {}).get("display_name") or loc.get("landing_page_url") or "?"
        if not url:
            continue
        try:
            resp = requests.get(url, headers=UA, timeout=60, allow_redirects=True)
        except requests.RequestException as e:
            tried.append(f"{src}: {type(e).__name__}")
            continue
        if not (resp.ok and resp.content[:5] == b"%PDF-"):
            tried.append(f"{src}: HTTP {resp.status_code}, not a PDF")
            continue
        tmp = OA_DIR / f"{pmid}.check.pdf"
        tmp.write_bytes(resp.content)
        ok, why = is_the_paper(tmp, title)
        if ok:
            tmp.rename(dest)
            marker.unlink(missing_ok=True)  # the source ladder skips papers marked as having no free PDF
            return out | {"outcome": "downloaded", "source": src, "link": url}
        tmp.unlink()
        tried.append(f"{src}: {why}")
    free_page = next((l.get("landing_page_url") for l in locs if l.get("landing_page_url")), row.get("link", ""))
    return out | {"outcome": "free, save by hand" if free_page else "not downloaded", "source": "; ".join(tried)[:300],
                  "link": free_page}


def main(argv: list[str]) -> int:
    run = Path(argv[0])
    rows = list(csv.DictReader((run / "pdfs_needed.csv").open()))
    OA_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, rows))
    out = run / "free_fulltext_found.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pmid", "outcome", "source", "link", "doi", "title", "was"])
        w.writeheader()
        w.writerows(results)
    from collections import Counter

    print(json.dumps(Counter(r["outcome"] for r in results), indent=1))
    print(f"{len(rows)} papers checked in {time.time() - started:.0f} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
