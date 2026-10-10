"""Get a paper's full text, trying each source in turn, and say where it came from.

    1. the team's PDF      data/pdfs/<PMID>.pdf  (what the human reviewers read; preferred when present)
    2. PubMed Central      structured XML, free   (about 61% of the professor's search results)
    3. open-access PDF     downloaded once when Europe PMC lists the article as open access
    4. none                the article is NOT excluded: it is flagged for a human and listed in
                           pdfs_needed.csv so the PDF can be fetched through the library

    python -m adrd_review.tools.fulltext 23172761 31234567     # show the source for these PMIDs
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from adrd_review.schemas import Article
from adrd_review.tools import pdf as pdf_tool
from adrd_review.tools.pmc import DEFAULT_FULLTEXT_DIR, FullText, Section, fetch_fulltext
from adrd_review.tools.pubmed import load_cached

DEFAULT_PDF_DIR = Path("data/pdfs")
OA_PDF_DIR = Path("data/pdfs_oa")
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

PRIORITY = ("method", "participant", "design", "sample", "procedure", "measure", "result", "finding", "table")


@dataclass
class FullTextResult:
    fulltext: FullText | None
    source: str | None  # "team_pdf" | "pmc" | "oa_pdf" | None
    reason: str  # why this source, or why none
    link: str | None = None  # when nothing was found: where a person can get the paper

    @property
    def found(self) -> bool:
        return self.fulltext is not None


def _from_pdf(article: Article, path: Path, source: str) -> FullTextResult | None:
    sections, status = pdf_tool.extract(path)
    if status != "ok":
        return FullTextResult(None, None, f"{source} rejected: {status}")
    ft = FullText(pmid=article.pmid, pmcid=article.pmcid or "", sections=sections,
                  fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return FullTextResult(ft, source, f"{path.name}")


def _free_links(pmid: str) -> tuple[str | None, str | None]:
    """(open-access PDF url, best link for a person) from Europe PMC.

    Only links marked 'Open access' are downloaded automatically (reuse allowed). In practice
    most are publisher pages or PDFs behind bot protection, so the second value, the best free
    link, goes into pdfs_needed.csv for a person to open.
    """
    r = requests.get(EPMC, params={"query": f"EXT_ID:{pmid} AND SRC:MED", "resultType": "core", "format": "json"}, timeout=30)
    hits = r.json().get("resultList", {}).get("result") or []
    urls = (hits[0].get("fullTextUrlList") or {}).get("fullTextUrl", []) if hits else []
    free = [u for u in urls if u.get("availability") in ("Open access", "Free")]
    oa_pdf = next((u["url"] for u in free if u["availability"] == "Open access" and u.get("documentStyle") == "pdf"), None)
    best = next((u["url"] for u in free if u.get("documentStyle") == "pdf"), None) or (free[0]["url"] if free else None)
    return oa_pdf, best


def _download_oa_pdf(article: Article, oa_dir: Path) -> tuple[Path | None, str | None]:
    """(downloaded PDF, best free link). The lookup is cached so each article is asked once."""
    dest = oa_dir / f"{article.pmid}.pdf"
    marker = oa_dir / f"{article.pmid}.none"
    if dest.exists():
        return dest, None
    if marker.exists():
        return None, marker.read_text().strip() or None
    oa_dir.mkdir(parents=True, exist_ok=True)
    oa_pdf, best = _free_links(article.pmid)
    if oa_pdf:
        try:
            resp = requests.get(oa_pdf, timeout=60, headers={"User-Agent": "adrd-review (research; text extraction)"})
            if resp.ok and resp.content[:5] == b"%PDF-":
                dest.write_bytes(resp.content)
                return dest, None
        except requests.RequestException:
            pass
    marker.write_text((best or "") + "\n")
    return None, best


def get_fulltext(
    article: Article,
    pdf_dir: str | Path = DEFAULT_PDF_DIR,
    pmc_dir: str | Path = DEFAULT_FULLTEXT_DIR,
    oa_dir: str | Path = OA_PDF_DIR,
    allow_download: bool = True,
    client=None,
) -> FullTextResult:
    """Never raises: a source that errors is skipped and the reason recorded."""
    notes: list[str] = []
    link = f"https://doi.org/{article.doi}" if article.doi else f"https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/"

    team = Path(pdf_dir) / f"{article.pmid}.pdf"
    if team.exists():
        res = _from_pdf(article, team, "team_pdf")
        if res.found:
            return res
        notes.append(res.reason)

    if article.pmcid:
        try:
            ft = fetch_fulltext(article, cache_dir=pmc_dir, client=client) if client else fetch_fulltext(article, cache_dir=pmc_dir)
            if ft is not None:
                return FullTextResult(ft, "pmc", article.pmcid)
            notes.append("PubMed Central has no downloadable body text")
        except Exception as e:
            notes.append(f"PubMed Central fetch failed: {type(e).__name__}")
    else:
        notes.append("not in PubMed Central")

    if allow_download:
        try:
            path, free_link = _download_oa_pdf(article, Path(oa_dir))
            if path is not None:
                res = _from_pdf(article, path, "oa_pdf")
                if res.found:
                    return res
                notes.append(res.reason)
            elif free_link:
                link = free_link
                notes.append("free copy exists but could not be downloaded automatically")
            else:
                notes.append("no free copy found (subscription)")
        except Exception as e:
            notes.append(f"open-access lookup failed: {type(e).__name__}")

    return FullTextResult(None, None, "; ".join(notes), link=link)


def fit_text(ft: FullText, max_chars: int) -> tuple[str, bool]:
    """The paper's text for a model: all of it when it fits, otherwise the sections that
    matter most for eligibility (methods, participants, results) first. Returns (text, truncated)."""
    def render(sections: list[Section]) -> str:
        return "\n\n".join(f"[{s.title.upper()}] {s.text}" if s.title else s.text for s in sections)

    full = render(ft.sections)
    if len(full) <= max_chars:
        return full, False
    first = [s for s in ft.sections if any(k in s.title.lower() for k in PRIORITY)]
    rest = [s for s in ft.sections if s not in first]
    out, used = [], 0
    for s in first + rest:
        piece = f"[{s.title.upper()}] {s.text}" if s.title else s.text
        room = max_chars - used
        if room < 400:
            break
        if len(piece) > room:
            cut = piece[:room]
            end = cut.rfind(". ")
            piece = (cut[: end + 1] if end > room * 0.7 else cut) + " [...]"
        out.append(piece)
        used += len(piece) + 2
    return "\n\n".join(out), True


def main(argv: list[str] | None = None) -> int:
    pmids = argv if argv is not None else sys.argv[1:]
    if not pmids:
        print("Usage: python -m adrd_review.tools.fulltext <PMID> [<PMID> ...]", file=sys.stderr)
        return 2
    for pmid in pmids:
        art = load_cached(pmid)
        if art is None:
            print(f"{pmid}: not in data/corpus (fetch it with adrd_review.tools.pubmed first)")
            continue
        res = get_fulltext(art)
        if res.found:
            print(f"{pmid}: {res.source} ({res.reason}), {len(res.fulltext.sections)} sections, {len(res.fulltext.text):,} chars")
        else:
            print(f"{pmid}: NO FULL TEXT ({res.reason})  get it here: {res.link}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
