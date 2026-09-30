"""Build the PLACEHOLDER gold standard from the public CLEF 2017 TAR dataset.

Stand-in until the professor's gold file arrives. It uses the real title/abstract
screening decisions made by Cochrane review teams for six diagnostic-accuracy
reviews of dementia screening instruments (MMSE, MoCA, Mini-Cog, IQCODE), and
samples 268 articles, the same size as the human gold standard.

    python scripts/build_placeholder_gold.py

Outputs (git-ignored):
    data/gold/gold.csv          standard gold format (see docs/GOLD_STANDARD.md)
    data/gold/gold_meta.json    provenance, sampling settings, known limitations
    data/corpus/<pmid>.json     the downloaded articles
    data/raw/clef2017/          cached CLEF label files

Labels:
    screening  include if ANY of the six reviews included it at title/abstract stage
    fulltext   include if ANY of the six reviews included it after full-text reading
    instrument instrument(s) of the review(s) that included it (includes only)
    setting    setting of the review(s) that included it (includes only; the MoCA
               review covered mixed settings, so it adds no setting label)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adrd_review.codebook import load_codebook  # noqa: E402
from adrd_review.gold import GoldRecord, file_sha256, load_gold, summarize, write_gold  # noqa: E402
from adrd_review.tools.pubmed import fetch_articles  # noqa: E402

CLEF_REPO = "CLEF-TAR/tar"
CLEF_COMMIT = "dbc13d02bb3e2f8ebc90e62ff47f5eb591e5ca20"  # pinned for reproducibility
QRELS = {
    "testing": ("2017-TAR/testing/qrels/qrel_abs_test.txt", "2017-TAR/testing/qrels/qrel_content_test.txt"),
    "training": ("2017-TAR/training/qrels/qrel_abs_train", "2017-TAR/training/qrels/qrel_content_train"),
}

# Cochrane review -> which CLEF split holds it, and the labels it implies for included studies.
TOPICS = {
    "CD010775": {"split": "testing", "instrument": "MoCA", "setting": None},
    "CD010783": {"split": "testing", "instrument": "MMSE", "setting": "mci_cohort"},
    "CD011145": {"split": "testing", "instrument": "MMSE", "setting": "community_or_primary_care"},
    "CD010860": {"split": "testing", "instrument": "Mini-Cog", "setting": "community"},
    "CD010771": {"split": "training", "instrument": "IQCODE", "setting": "primary_care"},
    "CD010772": {"split": "testing", "instrument": "IQCODE", "setting": "secondary_care"},
}


def download(rel_path: str, cache: Path) -> Path:
    dest = cache / Path(rel_path).name
    if not dest.exists():
        url = f"https://raw.githubusercontent.com/{CLEF_REPO}/{CLEF_COMMIT}/{rel_path}"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return dest


def read_qrels(path: Path, topics: set[str]) -> dict[str, dict[str, int]]:
    """topic -> {pmid: relevance}"""
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] in topics:
            out[parts[0]][parts[2]] = int(parts[3])
    return out


def round_robin(pools: dict[str, list[str]], rng: random.Random) -> list[str]:
    """Interleave shuffled per-review lists so every review is represented."""
    queues = {t: rng.sample(p, len(p)) for t, p in sorted(pools.items())}
    order, seen = [], set()
    while any(queues.values()):
        for t in sorted(queues):
            while queues[t]:
                pmid = queues[t].pop()
                if pmid not in seen:
                    seen.add(pmid)
                    order.append(pmid)
                    break
    return order


def take_fetchable(order: list[str], n: int, corpus: Path) -> list[str]:
    """First n PMIDs in `order` that PubMed actually returns."""
    kept: list[str] = []
    pos = 0
    while len(kept) < n and pos < len(order):
        batch = order[pos : pos + (n - len(kept)) + 10]
        pos += len(batch)
        articles, _ = fetch_articles(batch, cache_dir=corpus)
        kept += [p for p in batch if p in articles][: n - len(kept)]
    if len(kept) < n:
        raise RuntimeError(f"Only {len(kept)} fetchable articles, wanted {n}")
    return kept


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-include", type=int, default=90)
    ap.add_argument("--n-exclude", type=int, default=178)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default="data/gold/gold.csv")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--codebook", default="configs/codebook.yaml")
    args = ap.parse_args(argv)

    cache = Path("data/raw/clef2017")
    cache.mkdir(parents=True, exist_ok=True)
    topic_ids = set(TOPICS)
    abs_rel: dict[str, dict[str, int]] = defaultdict(dict)
    ft_rel: dict[str, dict[str, int]] = defaultdict(dict)
    for split in {t["split"] for t in TOPICS.values()}:
        abs_file, ft_file = QRELS[split]
        for t, rel in read_qrels(download(abs_file, cache), topic_ids).items():
            abs_rel[t].update(rel)
        for t, rel in read_qrels(download(ft_file, cache), topic_ids).items():
            ft_rel[t].update(rel)

    missing_topics = topic_ids - set(abs_rel)
    if missing_topics:
        raise RuntimeError(f"No CLEF labels found for {sorted(missing_topics)}")

    included_by: dict[str, set[str]] = defaultdict(set)
    ft_included_by: dict[str, set[str]] = defaultdict(set)
    for t, rels in abs_rel.items():
        for pmid, rel in rels.items():
            if rel:
                included_by[pmid].add(t)
    for t, rels in ft_rel.items():
        for pmid, rel in rels.items():
            if rel:
                ft_included_by[pmid].add(t)

    include_pools = {t: [p for p, r in rels.items() if r] for t, rels in abs_rel.items()}
    exclude_pools = {t: [p for p in rels if p not in included_by] for t, rels in abs_rel.items()}

    rng = random.Random(args.seed)
    corpus = Path(args.corpus)
    includes = take_fetchable(round_robin(include_pools, rng), args.n_include, corpus)
    excludes = take_fetchable(round_robin(exclude_pools, rng), args.n_exclude, corpus)

    records = []
    for pmid in includes + excludes:
        topics = sorted(included_by.get(pmid, set()))
        labels = {}
        if topics:
            labels["instrument"] = sorted({TOPICS[t]["instrument"] for t in topics})
            settings = sorted({TOPICS[t]["setting"] for t in topics if TOPICS[t]["setting"]})
            if settings:
                labels["setting"] = settings
        candidate_of = sorted(t for t in TOPICS if pmid in abs_rel[t])
        records.append(
            GoldRecord(
                pmid=pmid,
                screening="include" if topics else "exclude",
                fulltext="include" if ft_included_by.get(pmid) else "exclude",
                labels=labels,
                source="CLEF2017:" + ",".join(candidate_of),
            )
        )
    records.sort(key=lambda r: int(r.pmid))

    codebook = load_codebook(args.codebook)
    write_gold(records, args.out, dimensions=list(codebook.dimensions))
    records = load_gold(args.out, codebook)  # validates against the codebook

    meta = {
        "status": "PLACEHOLDER - replace with the professor's gold standard",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"CLEF 2017 TAR, github.com/{CLEF_REPO} @ {CLEF_COMMIT} (MIT licence)",
        "reviews": {t: {k: v for k, v in cfg.items() if k != "split"} for t, cfg in TOPICS.items()},
        "sampling": {
            "seed": args.seed,
            "n_include": args.n_include,
            "n_exclude": args.n_exclude,
            "method": "round-robin across the six reviews; only PMIDs PubMed returns",
        },
        "gold_sha256": file_sha256(args.out),
        "codebook_version": codebook.version,
        "summary": summarize(records),
        "limitations": [
            "Screening labels are real Cochrane decisions, but from six separate reviews merged into one question.",
            "Annotation labels are derived from which review included the study, not coded by a person reading it: "
            "positives only (a study may evaluate other instruments too).",
            "No per-reviewer labels, so the human Fleiss' kappa cannot be computed from this file.",
            "Exclusion reasons are not available.",
        ],
    }
    Path(args.out).with_name("gold_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(json.dumps(meta["summary"], indent=2))
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
