"""Third look: one more independent model on the papers still left for the professor after the final result.

Asked for by Utsa (2026-10-21) to settle more of those papers. The model is from a family used nowhere else in the
review (DeepSeek; the panel used gpt-oss, Qwen, Nemotron, Gemma, dots; the second look GLM; the age check Nemotron).
For each paper in <run>/final/for_professor.csv it reads the same full text and:

1. extracts the participants' age facts with quotes (the age check's form) and applies 65+ (age_check.decide);
2. marks every rule with quotes (the full-text reviewer's form; told which rules are open, not anyone's votes).

Outcome (the same evidence standard as every other step; quotes are checked against the paper):
    excluded       age clearly under 65, or a rule other than I4 is clearly not met / an exclusion applies
    included       age 65+ confirmed (a verified quote from this model, or from the earlier age check) and I1-I3
                   met and no exclusion applies
    for professor  otherwise (the paper really does not say)

    python -m adrd_review.third_look data/runs/PROF_search_free/<timestamp>
Writes <run>/third_look.csv; final_results uses it when present.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adrd_review import age_check
from adrd_review.agents import eligibility
from adrd_review.codebook import load_codebook
from adrd_review.eval.stages import load_states
from adrd_review.llm import LLM, ModelConfig, build_llm
from adrd_review.schemas import Article

DEFAULT_MODEL = ModelConfig(provider="openrouter", model="deepseek/deepseek-v4.1-flash", max_tokens=16000,
                            requests_per_minute=30)
PERSONA = (
    "an independent senior reviewer giving a third opinion. Two rounds of reviewers could not settle this paper; "
    "the open points were: {open}. Read the whole paper, including methods, participant tables and the data set "
    "description. Mark a rule met or not met only with a quote from the paper; if the paper really does not say, "
    "mark it unclear"
)


def look(state: dict, prev_age: dict, reason: str, codebook, llm: LLM) -> dict:
    article = Article.model_validate(state["article"])
    row = {"pmid": state["pmid"], "title": article.title, "before": reason}
    if not article.fulltext:
        return row | {"third look": "for the professor", "why": "no full text saved"}
    res = llm.structured(age_check.AgeAnswer, age_check.SYSTEM,
                         f"PAPER (PMID {article.pmid})\n\n{article.title}\n\n{article.fulltext}", role="third_look_age")
    i4, age_reason = age_check.decide(res.parsed, article)
    if i4 == "unclear" and prev_age.get("I4 at 65") == "met":
        i4, age_reason = "met", f"earlier age check: {prev_age.get('reason', '')}"
    vote, _ = eligibility.run(article, codebook, llm, reviewer_id="T", persona=PERSONA.format(open=reason))
    checks = vote.rule_checks
    others = [r for r in codebook.screening.inclusion if r.id != "I4"]
    failed = [r.id for r in others if checks.get(r.id) == "not_met"] + \
             [r.id for r in codebook.screening.exclusion if checks.get(r.id) == "applies"]
    names = {r.id: f"{r.id} {r.name}" for r in codebook.rules}
    if i4 == "not_met":
        outcome, why = "excluded", f"I4 Older adults (65 or over): {age_reason}"
    elif failed:
        outcome, why = "excluded", names[failed[0]]
    elif i4 == "met" and all(checks.get(r.id) == "met" for r in others):
        outcome, why = "included", f"all rules met; age: {age_reason}"
    else:
        open_ = (["I4 (age)"] if i4 != "met" else []) + [r.id for r in others if checks.get(r.id) != "met"]
        outcome, why = "for the professor", "still unclear: " + ", ".join(open_)
    a = res.parsed
    return row | {"third look": outcome, "why": why, "I4 at 65": i4,
                  "rule marks": " · ".join(f"{k} {m}" for k, m in checks.items()),
                  "age quotes": " | ".join(q for q in (a.min_age.quote, a.max_age.quote, a.mean_age.quote) if q),
                  "quotes": "\n".join(f'"{e.quote}"' for e in vote.evidence if e.quote),
                  "reasoning": vote.rationale, "model": llm.config.label}


def run(run_dir: Path, model: ModelConfig = DEFAULT_MODEL, concurrency: int = 6, llm: LLM | None = None) -> list[dict]:
    meta = json.loads((run_dir / "run_meta.json").read_text())["config"]
    codebook = load_codebook(meta["codebook"])
    llm = llm or build_llm(model, meta.get("cache_dir", "data/cache/llm"), codebook)
    states = {s["pmid"]: s for s in load_states(run_dir)}
    prev = {r["pmid"]: r for r in csv.DictReader((run_dir / "age_check.csv").open())} \
        if (run_dir / "age_check.csv").exists() else {}
    papers = list(csv.DictReader((run_dir / "final" / "for_professor.csv").open()))

    def one(p):
        try:
            return look(states[p["pmid"]], prev.get(p["pmid"], {}), p["reason"], codebook, llm)
        except Exception as e:  # a failed call leaves the paper with the professor; re-run to retry (cached)
            return {"pmid": p["pmid"], "title": p["title"], "before": p["reason"], "third look": "for the professor",
                    "why": f"third look failed: {type(e).__name__}: {str(e)[:200]}"}

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        rows = list(pool.map(one, papers))
    cols = ["pmid", "title", "before", "third look", "why", "I4 at 65", "rule marks", "age quotes", "quotes", "reasoning",
            "model"]
    with (run_dir / "third_look.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adrd_review.third_look")
    ap.add_argument("run_dir")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="only the first N papers (a cost check)")
    args = ap.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv()
    rows = run(Path(args.run_dir), concurrency=args.concurrency) if not args.limit else None
    if rows is None:  # pilot: N papers, nothing written
        run_dir = Path(args.run_dir)
        meta = json.loads((run_dir / "run_meta.json").read_text())["config"]
        cb = load_codebook(meta["codebook"])
        llm = build_llm(DEFAULT_MODEL, meta.get("cache_dir", "data/cache/llm"), cb)
        states = {s["pmid"]: s for s in load_states(run_dir)}
        prev = {r["pmid"]: r for r in csv.DictReader((run_dir / "age_check.csv").open())}
        for p in list(csv.DictReader((run_dir / "final" / "for_professor.csv").open()))[: args.limit]:
            r = look(states[p["pmid"]], prev.get(p["pmid"], {}), p["reason"], cb, llm)
            print(r["pmid"], "|", r["third look"], "|", r["why"][:120])
        return 0
    print(f"{len(rows)} papers: " + ", ".join(f"{k} {v}" for k, v in Counter(r["third look"] for r in rows).most_common()))
    failed = sum("failed" in (r.get("why") or "") for r in rows)
    if failed:
        print(f"({failed} failed: re-run to retry)")
    print(f"Saved: {Path(args.run_dir) / 'third_look.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
