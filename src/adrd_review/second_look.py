"""Second look: before the result is final, an independent stronger model re-reads every paper the panel left
for a person (full text "unclear": no rule clearly failed, but not every inclusion rule was clearly met).

It reads the same full text the panel read and is told which rules the panel could not settle, so it searches
for that evidence (participant tables, dataset descriptions, methods). It does not see the panel's votes. It
answers with the same per-rule form and the same decision rule as a full-text reviewer (agents/eligibility.py):
  include  every inclusion rule met, each with a quote that is in the paper  -> included after the second look
  exclude  a rule clearly not met / an exclusion applies, with a quote        -> excluded after the second look
  unclear  the paper does not say                                             -> stays with the professor
Quotes are checked against the paper as for every reviewer. Nothing in the run is changed: the result is a
separate table (second_look.csv) next to the run's other outputs.

    python -m adrd_review.second_look data/runs/PROF_search_free/<timestamp>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adrd_review.agents import eligibility
from adrd_review.codebook import Codebook, load_codebook
from adrd_review.eval.stages import load_states, split_by_stage
from adrd_review.llm import LLM, ModelConfig, build_llm
from adrd_review.schemas import Article

# Independent of the panel (a different model family from A, B and C), on OpenRouter (paid, ~$0.002-0.007 per paper).
# GLM-5.3 reasons before answering: with 4,096 output tokens long papers were cut off, unreadable, retried and billed
# (2026-10-21), so the answer gets 16,000 tokens. No overflow host: NVIDIA's free copy hung for minutes per paper; a
# paper whose call fails stays with the professor (re-run to retry).
DEFAULT_MODEL = ModelConfig(provider="openrouter", model="z-ai/glm-5.3", max_tokens=16000, requests_per_minute=30)
OUTCOME = {"include": "included", "exclude": "excluded", "unclear": "for the professor"}

PERSONA = (
    "the senior second reviewer. A panel of three reviewers read this paper and could not settle these rules: "
    "{rules}. Read the whole paper, including the methods, participant tables and the description of the data "
    "set, and look specifically for evidence on those rules. Mark a rule met or not met only with a quote from "
    "the paper; if the paper really does not say, mark it unclear"
)


def open_rules(state: dict, codebook: Codebook) -> list[str]:
    """Rules the panel left open: those the 'unclear' and dissenting votes cite, in codebook order."""
    cited = {r for v in state.get("eligibility_votes") or [] if v["labels"] != ["include"] for r in v.get("rule_ids") or []}
    return [r.id for r in codebook.rules if r.id in cited] or [r.id for r in codebook.screening.inclusion]


def waiting(run_dir: Path) -> list[dict]:
    """The papers the run left for a person at full text (same assignment as the stage files)."""
    return [s for s, outcome in split_by_stage(load_states(run_dir))["fulltext"] if outcome == "waiting for a person"]


def look(state: dict, codebook: Codebook, llm: LLM) -> dict:
    article = Article.model_validate(state["article"])
    rules = open_rules(state, codebook)
    names = {r.id: f"{r.id} ({r.name})" for r in codebook.rules}
    row = {"pmid": state["pmid"], "link": f"https://pubmed.ncbi.nlm.nih.gov/{state['pmid']}/", "title": article.title,
           "year": article.year or "", "panel votes": "; ".join(
               f"{v['reviewer_id']}: {v['labels'][0]}" + (f" [{', '.join(v['rule_ids'])}]" if v.get("rule_ids") else "")
               for v in state.get("eligibility_votes") or []),
           "open rules": ", ".join(rules)}
    if not article.fulltext:
        return row | {"second look": "for the professor", "decision": "", "rule marks": "", "quotes": "",
                      "reasoning": "no full text saved for this paper", "model": ""}
    vote, usage = eligibility.run(article, codebook, llm, reviewer_id="S",
                                  persona=PERSONA.format(rules=", ".join(names[r] for r in rules)))
    return row | {
        "second look": OUTCOME[vote.labels[0]],
        "decision": vote.labels[0],
        "rule marks": " · ".join(f"{k} {m}" for k, m in vote.rule_checks.items()),
        "quotes": "\n".join(f'"{e.quote}"' for e in vote.evidence if e.quote),
        "reasoning": vote.rationale,
        "model": usage.model,
    }


def run(run_dir: Path, model: ModelConfig = DEFAULT_MODEL, concurrency: int = 4, llm: LLM | None = None) -> list[dict]:
    meta = json.loads((run_dir / "run_meta.json").read_text())["config"]
    codebook = load_codebook(meta["codebook"])
    llm = llm or build_llm(model, meta.get("cache_dir", "data/cache/llm"), codebook)
    papers = waiting(run_dir)

    def one(state: dict) -> dict:
        try:
            return look(state, codebook, llm)
        except Exception as e:  # a failed call leaves the paper with the professor; re-run to retry (cached)
            return {"pmid": state["pmid"], "title": state["article"].get("title", ""), "second look": "for the professor",
                    "reasoning": f"second look failed: {type(e).__name__}: {str(e)[:200]}"}

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        rows = list(pool.map(one, papers))
    out = run_dir / "second_look.csv"
    cols = ["pmid", "link", "title", "year", "second look", "open rules", "panel votes", "decision", "rule marks",
            "quotes", "reasoning", "model"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adrd_review.second_look")
    ap.add_argument("run_dir")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv()
    rows = run(Path(args.run_dir), concurrency=args.concurrency)
    counts = Counter(r["second look"] for r in rows)
    failed = sum("second look failed" in (r.get("reasoning") or "") for r in rows)
    print(f"{len(rows)} paper(s) left for a person; after the second look: "
          + ", ".join(f"{k} {v}" for k, v in counts.most_common()) + (f" ({failed} failed, re-run to retry)" if failed else ""))
    print(f"Saved: {Path(args.run_dir) / 'second_look.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
