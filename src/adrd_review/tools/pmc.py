"""PubMed Central full-text tool.

Fetches an article's full text from PubMed Central (free, no key) when the
article has a PMCID, and keeps the body text by section. Used by the pipeline
when triage finds an abstract too thin (`fulltext: when_needed` in a run config).

    python -m adrd_review.tools.pmc 23172761      # fetch by PMID (must have a PMCID)

Cached as data/fulltext/<pmid>.json. Only what PubMed Central serves is used;
articles without a PMC full text are left abstract-only.
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from adrd_review.schemas import Article
from adrd_review.tools.pubmed import PubMedClient, load_cached

DEFAULT_FULLTEXT_DIR = Path("data/fulltext")
# Back matter to leave out. "References" must be the WHOLE title: a methods subsection
# called "Reference standard" is exactly what a diagnostic-accuracy review needs.
SKIP_SECTIONS = re.compile(
    r"^(references?|reference list|bibliography|literature cited)\s*$"
    r"|^(acknowledg|funding|conflicts? of interest|competing interests?|authors?'? contributions?|"
    r"pre-publication|supplementary|abbreviations|data availability|declarations?|ethics approval|"
    r"consent for publication)",
    re.I,
)


class Section(BaseModel):
    title: str
    text: str


class FullText(BaseModel):
    pmid: str
    pmcid: str
    sections: list[Section] = Field(default_factory=list)
    license: str | None = None
    fetched_at: str | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(f"{s.title.upper()}: {s.text}" if s.title else s.text for s in self.sections)


def _text(el: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_jats(xml_text: str) -> tuple[list[Section], str | None]:
    """Body sections (title + paragraphs) and the licence from a PMC JATS XML response."""
    root = ET.fromstring(xml_text)
    article = root.find(".//article") if _local(root.tag) != "article" else root
    if article is None:
        return [], None
    license_ = None
    for el in article.iter():
        name = _local(el.tag)
        if name == "license" and el.get("license-type"):
            license_ = el.get("license-type")
        elif name == "license_ref" and _text(el):
            license_ = _text(el)
            break
    body = article.find("body")
    if body is None:
        return [], license_

    sections: list[Section] = []

    def walk(sec: ET.Element, prefix: str) -> None:
        title_el = sec.find("title")
        title = _text(title_el) if title_el is not None else ""
        full_title = f"{prefix} / {title}" if prefix and title else (title or prefix)
        if SKIP_SECTIONS.match(title or ""):
            return
        paras = [_text(p) for p in sec.findall("p") if _text(p)]
        if paras:
            sections.append(Section(title=full_title, text=" ".join(paras)))
        for child in sec.findall("sec"):
            walk(child, full_title)

    loose = [_text(p) for p in body.findall("p") if _text(p)]
    if loose:
        sections.append(Section(title="", text=" ".join(loose)))
    for sec in body.findall("sec"):
        walk(sec, "")
    return sections, license_


def fetch_fulltext(
    article: Article,
    cache_dir: str | Path = DEFAULT_FULLTEXT_DIR,
    client: PubMedClient | None = None,
) -> FullText | None:
    """Full text for an article with a PMCID, or None if PMC has no body text for it."""
    if not article.pmcid:
        return None
    cache = Path(cache_dir) / f"{article.pmid}.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        return FullText.model_validate(data) if data.get("sections") else None
    client = client or PubMedClient.from_env()
    xml = client._request(
        "efetch.fcgi", {"db": "pmc", "id": article.pmcid.removeprefix("PMC"), "retmode": "xml"}
    ).text
    sections, license_ = parse_jats(xml)
    ft = FullText(
        pmid=article.pmid,
        pmcid=article.pmcid,
        sections=sections,
        license=license_,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(ft.model_dump_json(indent=2))  # also caches "no body" so we do not ask again
    return ft if sections else None


def excerpt(ft: FullText, max_chars: int) -> str:
    """The first `max_chars` characters of the body, cut at a sentence end where possible."""
    text = ft.text
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = cut.rfind(". ")
    return (cut[: end + 1] if end > max_chars * 0.8 else cut) + " [...]"


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("Usage: python -m adrd_review.tools.pmc <PMID>", file=sys.stderr)
        return 2
    art = load_cached(args[0])
    if art is None:
        print("Article not in data/corpus; fetch it first with adrd_review.tools.pubmed fetch.", file=sys.stderr)
        return 1
    ft = fetch_fulltext(art)
    if ft is None:
        print(f"No PubMed Central full text for PMID {art.pmid} (PMCID: {art.pmcid or 'none'}).")
        return 1
    print(f"{ft.pmcid}: {len(ft.sections)} sections, {len(ft.text):,} characters, licence: {ft.license}")
    for s in ft.sections[:12]:
        print(f"  - {s.title or '(untitled)'}: {len(s.text):,} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
