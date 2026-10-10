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


# Bump when parse_jats changes what it extracts: cached full texts from an older parser are fetched again.
# 2 (2026-10-10): tables included (participants' age is usually only in Table 1).
PARSER_VERSION = 2


class FullText(BaseModel):
    pmid: str
    pmcid: str
    sections: list[Section] = Field(default_factory=list)
    license: str | None = None
    fetched_at: str | None = None
    parser_version: int = 1

    @property
    def text(self) -> str:
        return "\n\n".join(f"{s.title.upper()}: {s.text}" if s.title else s.text for s in self.sections)


FLOATS = ("table-wrap", "fig", "table-wrap-group", "fig-group")
MAX_TABLE_CHARS = 6000  # a very long (e.g. supplementary) table is cut; Table 1 demographics fit easily


def _itertext(el: ET.Element, skip: tuple[str, ...]):
    if _local(el.tag) in skip:
        return
    if el.text:
        yield el.text
    for child in el:
        yield from _itertext(child, skip)
        if child.tail:
            yield child.tail


def _text(el: ET.Element, skip: tuple[str, ...] = ()) -> str:
    return re.sub(r"\s+", " ", "".join(_itertext(el, skip) if skip else el.itertext())).strip()


def _table_text(wrap: ET.Element) -> str:
    """One table as text: label, caption, then one line per row with cells separated by ' | '.
    Tables hold what reviewers often need most (participants' age, sample size, settings)."""
    head = " ".join(x for x in (_text(wrap.find("label")) if wrap.find("label") is not None else "",
                                  _text(wrap.find("caption")) if wrap.find("caption") is not None else "") if x)
    rows = []
    for tr in wrap.iter():
        if _local(tr.tag) == "tr":
            cells = [_text(c) for c in tr if _local(c.tag) in ("td", "th")]
            if any(cells):
                rows.append(" | ".join(cells))
    foot = [_text(f) for f in wrap.iter() if _local(f.tag) == "table-wrap-foot"]
    text = "\n".join([head, *rows, *foot]).strip()
    return text[:MAX_TABLE_CHARS] + (" [table cut]" if len(text) > MAX_TABLE_CHARS else "")


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
        paras = [_text(p, FLOATS) for p in sec.findall("p") if _text(p, FLOATS)]
        if paras:
            sections.append(Section(title=full_title, text=" ".join(paras)))
        for child in sec.findall("sec"):
            walk(child, full_title)

    loose = [_text(p, FLOATS) for p in body.findall("p") if _text(p, FLOATS)]
    if loose:
        sections.append(Section(title="", text=" ".join(loose)))
    for sec in body.findall("sec"):
        walk(sec, "")
    # Tables, wherever JATS puts them (inside sections, paragraphs, or the floats group after the body).
    tables = [_table_text(w) for w in article.iter() if _local(w.tag) == "table-wrap"]
    tables = [t for t in tables if t]
    if tables and sections:
        sections.append(Section(title="Tables", text="\n\n".join(tables)))
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
        if data.get("parser_version", 1) >= PARSER_VERSION:
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
        parser_version=PARSER_VERSION,
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
