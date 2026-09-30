"""PubMed tool: search PubMed, look up PMIDs, and download articles.

Uses NCBI E-utilities (free; no key required). Without an API key NCBI
allows 3 requests/second; with NCBI_API_KEY set it allows 10.

Command line (run from the repo root with the venv active):

    python -m adrd_review.tools.pubmed fetch --pmids 12345 67890
    python -m adrd_review.tools.pubmed fetch --pmids-file gold.xlsx --column PMID
    python -m adrd_review.tools.pubmed search "dementia screening[tiab]" --out ids.txt
    python -m adrd_review.tools.pubmed lookup --doi 10.1000/xyz123

See docs/TOOLS.md for details.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

from adrd_review.schemas import AbstractSection, Article

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_CACHE_DIR = Path("data/corpus")
BATCH_SIZE = 200  # PMIDs per efetch request


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #


class PubMedClient:
    """Thin E-utilities client with rate limiting and retries."""

    def __init__(
        self,
        api_key: str | None = None,
        email: str | None = None,
        tool: str = "adrd-review",
        timeout: float = 60,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.email = email
        self.tool = tool
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = 0.11 if api_key else 0.34
        self._last_call = 0.0
        self.session = requests.Session()

    @classmethod
    def from_env(cls) -> "PubMedClient":
        return cls(
            api_key=os.getenv("NCBI_API_KEY") or None,
            email=os.getenv("NCBI_EMAIL") or None,
        )

    def _params(self, extra: dict) -> dict:
        params = {"tool": self.tool, **extra}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        return params

    def _request(self, endpoint: str, data: dict) -> requests.Response:
        url = f"{EUTILS}/{endpoint}"
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                # POST so long PMID lists never exceed URL length limits.
                resp = self.session.post(url, data=self._params(data), timeout=self.timeout)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                return resp
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
                if attempt == self.max_retries:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def esearch(self, term: str, **params) -> dict:
        data = {"db": "pubmed", "term": term, "retmode": "json", **params}
        return self._request("esearch.fcgi", data).json()["esearchresult"]

    def efetch_xml(self, pmids: list[str]) -> str:
        data = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
        return self._request("efetch.fcgi", data).text


# --------------------------------------------------------------------------- #
# Parsing (pure functions, tested offline)
# --------------------------------------------------------------------------- #


def _text(el: ET.Element | None) -> str:
    """All text inside an element, flattening inline tags like <i> or <sup>."""
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _year(article_el: ET.Element) -> int | None:
    pub_date = article_el.find("Journal/JournalIssue/PubDate")
    if pub_date is not None:
        year = pub_date.findtext("Year")
        if year and year.isdigit():
            return int(year)
        match = re.search(r"\d{4}", pub_date.findtext("MedlineDate") or "")
        if match:
            return int(match.group())
    year = article_el.findtext("ArticleDate/Year")
    return int(year) if year and year.isdigit() else None


def _author(el: ET.Element) -> str | None:
    collective = _text(el.find("CollectiveName"))
    if collective:
        return collective
    last = _text(el.find("LastName"))
    first = _text(el.find("ForeName")) or _text(el.find("Initials"))
    return f"{last}, {first}".strip(", ") or None


def parse_pubmed_xml(xml_text: str) -> list[Article]:
    """Parse an efetch XML response into Article records.

    Book records (PubmedBookArticle) are skipped; callers report them as missing.
    """
    root = ET.fromstring(xml_text)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    articles = []
    for rec in root.findall("PubmedArticle"):
        citation = rec.find("MedlineCitation")
        art = citation.find("Article")

        sections = [
            AbstractSection(label=a.get("Label") or None, text=_text(a))
            for a in art.findall("Abstract/AbstractText")
            if _text(a)
        ]
        abstract = "\n\n".join(
            f"{s.label}: {s.text}" if s.label else s.text for s in sections
        )

        # Only this article's own IDs; ReferenceList entries also carry
        # ArticleIdList elements, which belong to the cited papers.
        ids = {
            a.get("IdType"): _text(a)
            for a in rec.findall("PubmedData/ArticleIdList/ArticleId")
        }
        doi = ids.get("doi") or next(
            (_text(e) for e in art.findall("ELocationID") if e.get("EIdType") == "doi"),
            None,
        )

        articles.append(
            Article(
                pmid=_text(citation.find("PMID")),
                title=_text(art.find("ArticleTitle")),
                abstract=abstract,
                abstract_sections=sections,
                journal=_text(art.find("Journal/Title")) or None,
                year=_year(art),
                authors=[a for a in map(_author, art.findall("AuthorList/Author")) if a],
                doi=doi or None,
                pmcid=ids.get("pmc") or None,
                language=[_text(e) for e in art.findall("Language")],
                publication_types=[
                    _text(e) for e in art.findall("PublicationTypeList/PublicationType")
                ],
                mesh_terms=[
                    _text(e)
                    for e in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")
                ],
                keywords=[_text(e) for e in citation.findall("KeywordList/Keyword")],
                fetched_at=fetched_at,
            )
        )
    return articles


def normalize_pmids(raw: list) -> list[str]:
    """Clean a PMID list: strip, drop blanks and '.0' from spreadsheets, dedupe in order.

    Raises ValueError listing any entry that is not a plain number.
    """
    seen: set[str] = set()
    out: list[str] = []
    bad: list[str] = []
    for item in raw:
        s = str(item).strip()
        if not s or s.lower() == "nan":
            continue
        s = re.sub(r"\.0$", "", s)
        if not s.isdigit():
            bad.append(s)
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    if bad:
        raise ValueError(f"Not valid PMIDs: {bad[:10]}{' ...' if len(bad) > 10 else ''}")
    return out


def read_pmids_file(path: str | Path, column: str = "pmid") -> list[str]:
    """Read PMIDs from a .txt (one per line), .csv or .xlsx file."""
    path = Path(path)
    if path.suffix.lower() == ".txt":
        return normalize_pmids(path.read_text().splitlines())

    import pandas as pd

    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    matches = [c for c in df.columns if str(c).strip().lower() == column.lower()]
    if not matches:
        raise ValueError(f"No column named {column!r} in {path.name}. Columns: {list(df.columns)}")
    return normalize_pmids(df[matches[0]].tolist())


# --------------------------------------------------------------------------- #
# High-level operations
# --------------------------------------------------------------------------- #


@dataclass
class FetchReport:
    requested: int = 0
    downloaded: list[str] = field(default_factory=list)
    from_cache: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    no_abstract: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Requested:   {self.requested}",
            f"Downloaded:  {len(self.downloaded)}",
            f"From cache:  {len(self.from_cache)}",
            f"Missing:     {len(self.missing)}" + (f"  {self.missing}" if self.missing else ""),
            f"No abstract: {len(self.no_abstract)}"
            + (f"  {self.no_abstract}" if self.no_abstract else ""),
        ]
        return "\n".join(lines)


def load_cached(pmid: str, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Article | None:
    path = Path(cache_dir) / f"{pmid}.json"
    if not path.exists():
        return None
    return Article.model_validate_json(path.read_text())


def fetch_articles(
    pmids: list,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    client: PubMedClient | None = None,
) -> tuple[dict[str, Article], FetchReport]:
    """Download articles by PMID, saving each to <cache_dir>/<pmid>.json.

    Already-saved articles are read from disk unless refresh=True.
    Returns the articles (keyed by PMID, in input order) and a report.
    """
    pmids = normalize_pmids(pmids)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = client or PubMedClient.from_env()
    report = FetchReport(requested=len(pmids))
    found: dict[str, Article] = {}

    to_download = []
    for pmid in pmids:
        cached = None if refresh else load_cached(pmid, cache_dir)
        if cached:
            found[pmid] = cached
            report.from_cache.append(pmid)
        else:
            to_download.append(pmid)

    for i in range(0, len(to_download), BATCH_SIZE):
        batch = to_download[i : i + BATCH_SIZE]
        for article in parse_pubmed_xml(client.efetch_xml(batch)):
            if article.pmid not in batch:
                continue
            (cache_dir / f"{article.pmid}.json").write_text(article.model_dump_json(indent=2))
            found[article.pmid] = article
            report.downloaded.append(article.pmid)

    report.missing = [p for p in pmids if p not in found]
    report.no_abstract = [p for p in pmids if p in found and not found[p].has_abstract]
    return {p: found[p] for p in pmids if p in found}, report


def search(
    query: str,
    mindate: str | None = None,
    maxdate: str | None = None,
    client: PubMedClient | None = None,
) -> tuple[list[str], int]:
    """Run a PubMed query. Returns (PMIDs, total count).

    Dates are YYYY/MM/DD (or YYYY) and filter on publication date.
    """
    client = client or PubMedClient.from_env()
    params = {"retmax": 10000}
    if mindate or maxdate:
        params.update(datetype="pdat", mindate=mindate or "1800", maxdate=maxdate or "3000")
    result = client.esearch(query, **params)
    count = int(result["count"])
    if count > 10000:
        raise ValueError(f"Query matches {count} articles; narrow it (limit 10,000).")
    return result["idlist"], count


def lookup_pmid(
    doi: str | None = None,
    title: str | None = None,
    client: PubMedClient | None = None,
) -> str | None:
    """Find the PMID for a DOI or an exact title. Returns None unless exactly one match."""
    if not doi and not title:
        raise ValueError("Give a DOI or a title.")
    client = client or PubMedClient.from_env()
    term = f"{doi.strip()}[doi]" if doi else f'"{title.strip().rstrip(".")}"[ti]'
    ids = client.esearch(term, retmax=2)["idlist"]
    return ids[0] if len(ids) == 1 else None


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(prog="python -m adrd_review.tools.pubmed")
    sub = parser.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="Download articles by PMID into the cache folder")
    src = f.add_mutually_exclusive_group(required=True)
    src.add_argument("--pmids", nargs="+", help="PMIDs separated by spaces")
    src.add_argument("--pmids-file", help=".txt (one per line), .csv or .xlsx")
    f.add_argument("--column", default="pmid", help="PMID column name in a .csv/.xlsx file")
    f.add_argument("--out", default=str(DEFAULT_CACHE_DIR), help="Cache folder")
    f.add_argument("--refresh", action="store_true", help="Re-download even if cached")

    s = sub.add_parser("search", help="Run a PubMed query and list matching PMIDs")
    s.add_argument("query")
    s.add_argument("--mindate", help="YYYY or YYYY/MM/DD")
    s.add_argument("--maxdate", help="YYYY or YYYY/MM/DD")
    s.add_argument("--out", help="Save PMIDs to this .txt file")

    lk = sub.add_parser("lookup", help="Find the PMID for a DOI or title")
    g = lk.add_mutually_exclusive_group(required=True)
    g.add_argument("--doi")
    g.add_argument("--title")

    args = parser.parse_args(argv)

    if args.cmd == "fetch":
        pmids = args.pmids if args.pmids else read_pmids_file(args.pmids_file, args.column)
        _, report = fetch_articles(pmids, cache_dir=args.out, refresh=args.refresh)
        print(report.summary())
        print(f"Saved in: {args.out}/")
        return 1 if report.missing else 0

    if args.cmd == "search":
        pmids, count = search(args.query, args.mindate, args.maxdate)
        print(f"{count} articles match.")
        if args.out:
            Path(args.out).write_text("\n".join(pmids) + "\n")
            print(f"PMIDs saved to {args.out}")
        else:
            print("\n".join(pmids))
        return 0

    if args.cmd == "lookup":
        pmid = lookup_pmid(doi=args.doi, title=args.title)
        print(pmid or "No single match found.")
        return 0 if pmid else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
