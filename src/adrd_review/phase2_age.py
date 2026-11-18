"""Phase 2 with the strict age rule: the professor's 65+ rule applied to the majority vote's result.

The three phase 2 reviewers had the 65+ rule in their codebook, but each weighed it loosely: 48 papers that phase 1's
age check excluded on age (participants under 65) were included by phase 2's majority. The rule is strict, so
phase 2 now ends with the same dedicated age check as phase 1 (age_check.py): a model extracts only the ages, each
with an exact quote checked against the paper, and a fixed rule decides I4 at 65. On the papers the vote left
included or for human review:

    I4 not met (participants under 65)   excluded at full text (I4)
    I4 unclear (no quoted age)           an included paper goes to human review: included only when every rule
                                         is clearly met
    I4 met                               a human-review paper whose only open rule was I4 becomes included

Excluded and not-assessed papers are not touched (age cannot bring them back in). The reviewers' votes and
final.csv are unchanged; the result with the age rule is written next to them:

    python -m adrd_review.phase2_age data/runs/PHASE2_independent/<timestamp> [--config <phase 2 config>]

Writes <run>/age_check.csv, final_age65.csv and summary_age65.json (stage table, what the age rule changed, and
the comparison with phase 1).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adrd_review import age_check, phase2
from adrd_review.llm import build_llm

RULE = re.compile(r"\b([IE]\d)\b")


def _votes(run_dir: Path, ids: list[str]) -> dict[str, list[dict]]:
    """Per paper, the reviewers' non-include full-text votes as age_check reads them (labels + rule ids)."""
    out: dict[str, list[dict]] = {}
    for r in ids:
        for row in csv.DictReader((run_dir / f"reviewer_{r}.csv").open()):
            if row["verdict"] in ("excluded", "unclear"):
                out.setdefault(row["pmid"], []).append(
                    {"labels": ["exclude" if row["verdict"] == "excluded" else "unclear"],
                     "rule_ids": RULE.findall(row["reason"]) or ["?"]})  # no rule named: not open only on age
    return out


def apply(row: dict, checked: dict) -> dict:
    """The paper's outcome after the age rule (see the module docstring)."""
    i4 = checked.get("I4 at 65")
    if i4 == "not_met":
        return row | {"outcome": "excluded", "stage": "full text", "reason": "I4 Older adults (65 or over)"}
    after = checked.get("after", "")
    if after.startswith("for a person"):
        return row | {"outcome": "human review", "reason": "age unclear at 65"}
    if after.startswith("included"):
        return row | {"outcome": "included", "reason": "all rules met"}
    return row


def run(run_dir: Path, config: Path, concurrency: int = 6, llm=None, fulltext_source=None) -> dict:
    from adrd_review.schemas import Article
    from adrd_review.tools.fulltext import fit_text, get_fulltext
    from adrd_review.tools.pubmed import fetch_articles

    cfg = phase2.load_config(config)
    ids = [r.id for r in cfg.reviewers]
    final = {r["pmid"]: r for r in csv.DictReader((run_dir / "final.csv").open())}
    votes = _votes(run_dir, ids)
    todo = [p for p, r in final.items() if r["outcome"] in ("included", "human review")]
    arts, _ = fetch_articles(todo, cache_dir=cfg.corpus_dir)
    llm = llm or build_llm(age_check.DEFAULT_MODEL, cfg.cache_dir)
    fulltext_source = fulltext_source or (lambda a: get_fulltext(a, pdf_dir=cfg.pdf_dir))

    def one(pmid: str) -> dict:
        before = "included" if final[pmid]["outcome"] == "included" else "for a person"
        try:
            res = fulltext_source(arts[pmid])
            if not res.found:  # the reviewers read it, so this is a fetch failure: unchecked, never kept as is
                raise RuntimeError(f"full text not found now ({res.reason})")
            text = fit_text(res.fulltext, cfg.fulltext_max_chars)[0]
            article = Article.model_validate(arts[pmid].model_dump() | {"fulltext": text})
            state = {"pmid": pmid, "article": article.model_dump(), "eligibility_votes": votes.get(pmid, [])}
            return age_check.check(state, before, llm)
        except Exception as e:  # changes nothing; re-run to retry (answered papers are cached)
            return {"pmid": pmid, "title": final[pmid]["title"], "before": before, "after": before,
                    "I4 at 65": "failed", "reason": f"{type(e).__name__}: {str(e)[:200]}"}

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        checked = {r["pmid"]: r for r in pool.map(one, todo)}

    cols = ["pmid", "title", "before", "I4 at 65", "after", "reason", "age_criterion", "min_age", "max_age", "mean_age",
            "quotes", "note", "model"]
    with (run_dir / "age_check.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in checked.values())

    new = {p: apply(r, checked.get(p, {})) | {"before age rule": r["outcome"]} for p, r in final.items()}
    with (run_dir / "final_age65.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(new.values()))))
        w.writeheader()
        w.writerows(new.values())

    summary = {
        "stage_table": phase2.stage_table(new),
        "before_age_rule": phase2.stage_table(final),
        "age_rule_changes": dict(Counter(f"{final[p]['outcome']} -> {r['outcome']}" for p, r in new.items()
                                         if r["outcome"] != final[p]["outcome"])),
        "I4 at 65": dict(Counter(r["I4 at 65"] for r in checked.values())),
        "failed": sum(r["I4 at 65"] == "failed" for r in checked.values()),
    }
    if cfg.compare_with and Path(cfg.compare_with).exists():
        summary["compared_with_phase_1"] = phase2.compare(new, Path(cfg.compare_with))
    (run_dir / "summary_age65.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adrd_review.phase2_age")
    ap.add_argument("run_dir")
    ap.add_argument("--config", default="configs/runs/PHASE2_independent.yaml")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv()
    print(json.dumps(run(Path(args.run_dir), Path(args.config), args.concurrency), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
