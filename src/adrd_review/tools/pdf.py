"""Read a paper's text from a PDF (the copies the human team read).

Put PDFs in data/pdfs/ named by PMID, e.g. data/pdfs/23172761.pdf.

    python -m adrd_review.tools.pdf data/pdfs/23172761.pdf     # show what was extracted

Extraction is checked before it is trusted: scanned pages, garbled columns or a
near-empty file fail the quality check and the article falls back to the next
full-text source instead of feeding nonsense to the agents.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from adrd_review.tools.pmc import Section

MIN_CHARS = 3000  # a real paper has far more body text than this
MIN_WORD_SHARE = 0.60  # share of characters that are letters or spaces

# Common section headings, used to split the text and to drop the reference list.
HEADING = re.compile(
    r"^\s*(?:\d{1,2}\.?\s+)?(abstract|introduction|background|methods?|materials and methods|"
    r"participants|study design|results?|findings|discussion|conclusions?|limitations|"
    r"references|bibliography|acknowledg(?:e)?ments?|funding|supplementary (?:material|data))\s*:?\s*$",
    re.I,
)
STOP = re.compile(r"^(references|bibliography|acknowledg|funding|supplementary)", re.I)


def read_pdf(path: str | Path) -> str:
    """All text in the PDF, page by page. Raises on unreadable or encrypted files."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)


def clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def quality(text: str) -> tuple[bool, str]:
    """Is this extraction usable? Returns (ok, reason)."""
    n = len(text)
    if n < MIN_CHARS:
        return False, f"only {n:,} characters extracted (scanned or image-only PDF?)"
    wordish = sum(c.isalpha() or c.isspace() for c in text) / n
    if wordish < MIN_WORD_SHARE:
        return False, f"only {wordish:.0%} of characters are letters or spaces (garbled extraction)"
    words = text.split()
    if words and sum(len(w) for w in words) / len(words) > 15:
        return False, "words run together (column or spacing problem)"
    return True, "ok"


def split_sections(text: str) -> list[Section]:
    """Split on recognised headings; drop references and other back matter."""
    sections: list[Section] = []
    title, buf = "", []

    def flush():
        body = " ".join(" ".join(buf).split())
        if body:
            sections.append(Section(title=title, text=body))

    for line in text.splitlines():
        m = HEADING.match(line)
        if m and len(line.strip()) < 40:
            flush()
            title, buf = m.group(1).title(), []
            if STOP.match(m.group(1)):
                title = "__STOP__"
            continue
        buf.append(line)
    flush()
    kept = []
    for s in sections:
        if s.title == "__STOP__":
            break
        kept.append(s)
    return kept


def extract(path: str | Path) -> tuple[list[Section], str]:
    """Sections from a PDF, plus a status: 'ok' or the reason it was rejected."""
    try:
        text = clean(read_pdf(path))
    except Exception as e:
        return [], f"could not read PDF: {type(e).__name__}"
    ok, reason = quality(text)
    if not ok:
        return [], reason
    sections = split_sections(text)
    if sum(len(s.text) for s in sections) < MIN_CHARS:
        sections = [Section(title="", text=" ".join(text.split()))]  # no headings recognised: keep it whole
    return sections, "ok"


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("Usage: python -m adrd_review.tools.pdf <file.pdf>", file=sys.stderr)
        return 2
    sections, status = extract(args[0])
    print(f"{args[0]}: {status}")
    for s in sections:
        print(f"  - {s.title or '(no heading)'}: {len(s.text):,} chars")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
