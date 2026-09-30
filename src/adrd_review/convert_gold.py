"""Convert the professor's gold-standard spreadsheet to the standard gold format.

    python -m adrd_review.convert_gold configs/gold_mapping.yaml

The mapping file (see configs/gold_mapping.example.yaml) names the columns and
maps the file's values to codebook labels. The converter:
- finds a PMID for rows that only have a DOI or title (PubMed lookup),
- reports every value it cannot map instead of guessing,
- writes the standard CSV and validates it against the codebook.
It stops without writing if anything is unmapped or invalid; fix the mapping and re-run.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from adrd_review.codebook import load_codebook
from adrd_review.gold import SEP, GoldError, load_gold, summarize
from adrd_review.tools.pubmed import lookup_pmid, normalize_pmids


def _cell(row, col) -> str:
    if not col or col not in row or pd.isna(row[col]):
        return ""
    v = str(row[col]).strip()
    return v.removesuffix(".0") if v.replace(".", "").isdigit() else v


def convert(mapping: dict, lookup=lookup_pmid) -> tuple[pd.DataFrame, list[str], Counter]:
    """Returns (standard-format rows, problems, unmapped values)."""
    path = Path(mapping["file"])
    df = pd.read_excel(path, sheet_name=mapping.get("sheet") or 0, dtype=str) if path.suffix in (".xlsx", ".xls") \
        else pd.read_csv(path, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    problems: list[str] = []
    unmapped: Counter = Counter()

    wanted = [mapping["id"].get(k) for k in ("pmid", "doi", "title")] + [mapping["screening"]["column"]]
    wanted += [d["column"] for d in mapping.get("dimensions", {}).values()]
    wanted += [c for r in mapping.get("reviewers", []) for k, c in r.items() if k != "id"]
    wanted += [(mapping.get("fulltext") or {}).get("column")]
    for col in [c for c in wanted if c]:
        if col not in df.columns:
            problems.append(f"column {col!r} not in file (columns: {list(df.columns)})")
    if problems:
        return pd.DataFrame(), problems, unmapped

    def decision(raw: str, values: dict, what: str) -> str:
        if not raw:
            return ""
        if raw in values:
            return values[raw]
        unmapped[(what, raw)] += 1
        return ""

    rows = []
    for i, r in df.iterrows():
        line = i + 2
        pmid = _cell(r, mapping["id"].get("pmid"))
        if not pmid:
            doi, title = _cell(r, mapping["id"].get("doi")), _cell(r, mapping["id"].get("title"))
            pmid = (lookup(doi=doi) if doi else None) or (lookup(title=title) if title else None) or ""
            if not pmid:
                problems.append(f"row {line}: no PMID and none found from DOI/title ({doi or title or 'empty'})")
                continue
        out = {"pmid": pmid}
        out["screening"] = decision(_cell(r, mapping["screening"]["column"]), mapping["screening"]["values"], "screening")
        ft = mapping.get("fulltext") or {}
        out["fulltext"] = decision(_cell(r, ft.get("column")), ft.get("values", {}), "fulltext")
        for dim, spec in mapping.get("dimensions", {}).items():
            raw = _cell(r, spec["column"])
            labels = []
            for part in [p.strip() for p in raw.split(spec.get("separator", SEP)) if p.strip()]:
                if part in spec.get("ignore", []):
                    continue
                if part in spec["values"]:
                    labels.append(spec["values"][part])
                else:
                    unmapped[(dim, part)] += 1
            out[dim] = SEP.join(dict.fromkeys(labels)) if out["screening"] == "include" else ""
        for rv in mapping.get("reviewers", []):
            for fld, col in rv.items():
                if fld == "id":
                    continue
                vals = mapping["screening"]["values"] if fld == "screening" else {}
                out[f"{fld}__{rv['id']}"] = decision(_cell(r, col), vals, f"{fld} ({rv['id']})") if vals else _cell(r, col)
        out["source"] = mapping.get("source_label", "")
        rows.append(out)

    std = pd.DataFrame(rows)
    if not std.empty:
        try:
            normalize_pmids(std["pmid"].tolist())
        except ValueError as e:
            problems.append(str(e))
    return std, problems, unmapped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adrd_review.convert_gold")
    ap.add_argument("mapping")
    ap.add_argument("--codebook", default="configs/codebook.yaml")
    args = ap.parse_args(argv)
    mapping = yaml.safe_load(Path(args.mapping).read_text())
    std, problems, unmapped = convert(mapping)
    if unmapped:
        print("Values with no mapping (add them to the mapping file):")
        for (what, raw), n in sorted(unmapped.items()):
            print(f"  {what:24} {raw!r} x{n}")
    if problems:
        print("Problems:")
        for p in problems[:30]:
            print(f"  - {p}")
    if unmapped or problems:
        print("Nothing written. Fix the mapping and run again.")
        return 1
    out = Path(mapping.get("output", "data/gold/gold.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    std.to_csv(out, index=False)
    try:
        records = load_gold(out, load_codebook(args.codebook))
    except GoldError as e:
        print(e)
        print(f"Wrote {out}, but it does not validate against the codebook (see above).")
        return 1
    print(f"Wrote {out}: {summarize(records)}")
    print("Next: python -m adrd_review.tools.pubmed fetch --pmids-file " + str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
