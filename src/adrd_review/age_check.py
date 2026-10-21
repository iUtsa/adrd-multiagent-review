"""Age check: apply the professor's age rule (older adults = 65 or over) to every paper that reached full text.

The professor's rule is 65+ (email 2026-10-07, confirmed by Utsa 2026-10-21). The run's codebook said 60+ from a
mis-recorded message (2026-10-08). The codebook text is part of every reviewer prompt, so changing it would make
every cached answer new and re-judge all 1,855 papers; but age only decides papers at full text (a paper too young
for 60+ is too young for 65+), so the rule is re-applied there instead, by this dedicated step:

1. A model reads the full text (the same text the panel read) and extracts only the facts about age: the age
   criterion, mean/median age, youngest and oldest participant, each with an exact quote.
2. The quotes are checked against the text; a fact without a real quote is dropped.
3. A fixed rule decides I4 at 65:
     met       the study requires 65+, or the youngest participant is 65+, or the mean/median age is 65+
               (mean/median: the codebook's proposed reading, still to confirm with the professor)
     not met   the mean/median age is under 65, or the oldest participant is under 65
     unclear   no usable age with a quote (e.g. only "aged 60-85" with no mean, or nothing at all)
4. Outcome for the paper: an included paper whose I4 is 'not met' becomes excluded (I4), and one whose age is
   'unclear' at 65 goes to a person (a paper is included only when every rule is clearly met); a paper left for a
   person whose only open rule is I4 becomes included (met) or excluded (not met), else stays with the person.

Nothing in the run is changed: the result is a table, age_check.csv, reported as its own step.

    python -m adrd_review.age_check data/runs/PROF_search_free/<timestamp> [--second-look <run>/second_look.csv]
    python -m adrd_review.age_check data/runs/PROF_search_free/<timestamp> --stage abstract   # the abstract stage
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

from pydantic import BaseModel, Field

from adrd_review.agents.common import check_quotes
from adrd_review.eval.stages import load_states, split_by_stage
from adrd_review.llm import LLM, ModelConfig, build_llm
from adrd_review.schemas import Article, EvidenceSpan

THRESHOLD = 65
# A factual extraction, not a judgement: a free model is enough. Nemotron 3 Ultra, free on NVIDIA and on OpenRouter
# (1,000 free requests/day after the credit). NVIDIA slowed to ~90 s per paper (2026-10-21), so OpenRouter's free copy
# goes first and NVIDIA is the overflow; answers already cached from either host are reused.
DEFAULT_MODEL = ModelConfig(
    provider="openrouter", model="nvidia/nemotron-3-ultra-550b-a55b:free", max_tokens=4096, requests_per_minute=20,
    overflow=ModelConfig(provider="nvidia", model="nvidia/nemotron-3-ultra-550b-a55b", max_tokens=4096,
                         requests_per_minute=30),
)

SYSTEM = """\
You extract facts about the AGE of the human participants in a research paper. Do not judge anything else.

Report, each with an exact quote copied word for word from the paper (never paraphrase, never compute):
- age_criterion: the study's age eligibility rule, if stated (e.g. "aged 65 years or older", "60-85 years").
- min_age: the youngest participant's age, or the lower bound of the age criterion.
- max_age: the oldest participant's age, or the upper bound of the age criterion.
- mean_age: the mean or median age of the whole sample (if only per group, give the lowest group mean).
Use null for anything the paper does not state. Ages of a public dataset (e.g. ADNI) count only if THIS paper
states them. If the paper analyses several datasets, report the youngest sample."""


class AgeFact(BaseModel):
    value: float | None = Field(default=None, description="The number in years, or null if not stated.")
    quote: str | None = Field(default=None, description="Exact words from the paper that state it.")


class AgeAnswer(BaseModel):
    age_criterion: str | None = Field(default=None, description="The study's age eligibility rule, as quoted.")
    min_age: AgeFact = Field(default_factory=AgeFact)
    max_age: AgeFact = Field(default_factory=AgeFact)
    mean_age: AgeFact = Field(default_factory=AgeFact)
    note: str = Field(default="", description="One sentence: where the ages come from.")


def verified(fact: AgeFact, article: Article) -> float | None:
    """The value, only when its quote is really in the paper and contains the number."""
    if fact.value is None or not fact.quote:
        return None
    if check_quotes([EvidenceSpan(quote=fact.quote, source="fulltext")], article):
        return None
    number = f"{fact.value:g}"
    return fact.value if number.split(".")[0] in fact.quote else None


def decide(ans: AgeAnswer, article: Article) -> tuple[str, str]:
    """('met' | 'not_met' | 'unclear', reason) for I4 at 65."""
    lo, hi, mean = (verified(f, article) for f in (ans.min_age, ans.max_age, ans.mean_age))
    if lo is not None and lo >= THRESHOLD:
        return "met", f"youngest / age criterion {lo:g}"
    if mean is not None:
        return ("met", f"mean/median age {mean:g}") if mean >= THRESHOLD else ("not_met", f"mean/median age {mean:g}")
    if hi is not None and hi < THRESHOLD:
        return "not_met", f"oldest {hi:g}"
    return "unclear", "no usable age (with a quote) in the paper"


def targets(run_dir: Path, second_look: Path | None) -> list[tuple[dict, str]]:
    """(state, current outcome) for every paper included or left for a person at full text."""
    by = split_by_stage(load_states(run_dir))
    rows = [(s, {"passed": "included", "waiting for a person": "for a person"}[o]) for s, o in by["fulltext"]
            if o in ("passed", "waiting for a person")]
    if second_look and second_look.exists():
        sl = {r["pmid"]: r for r in csv.DictReader(second_look.open())}
        rows = [(s, {"included": "included (second look)", "excluded": "excluded (second look)"}.get(
            sl[s["pmid"]]["second look"], o) if s["pmid"] in sl and o == "for a person" else o) for s, o in rows]
    return rows


def open_only_on_age(state: dict) -> bool:
    cited = {r for v in state.get("eligibility_votes") or [] if v["labels"] != ["include"] for r in v.get("rule_ids") or []}
    return cited <= {"I4"}


def check(state: dict, outcome: str, llm: LLM) -> dict:
    article = Article.model_validate(state["article"])
    row = {"pmid": state["pmid"], "title": article.title, "before": outcome}
    if not article.fulltext:
        return row | {"I4 at 65": "unclear", "reason": "no full text saved", "after": outcome}
    res = llm.structured(AgeAnswer, SYSTEM, f"PAPER (PMID {article.pmid})\n\n{article.title}\n\n{article.fulltext}",
                         role="age_check")
    i4, reason = decide(res.parsed, article)
    after = outcome
    if outcome.startswith("included") and i4 == "not_met":
        after = "excluded (I4 under 65)"
    elif outcome.startswith("included") and i4 == "unclear":
        # included only when every rule is clearly met: age 65+ can no longer be confirmed from the paper
        after = "for a person (age unclear at 65)"
    elif outcome == "for a person" and open_only_on_age(state):
        after = {"met": "included (age 65+)", "not_met": "excluded (I4 under 65)"}.get(i4, outcome)
    a = res.parsed
    return row | {"I4 at 65": i4, "reason": reason, "after": after, "age_criterion": a.age_criterion or "",
                  "min_age": a.min_age.value, "max_age": a.max_age.value, "mean_age": a.mean_age.value,
                  "quotes": " | ".join(q for q in (a.min_age.quote, a.max_age.quote, a.mean_age.quote) if q),
                  "note": a.note, "model": res.usage.model}


# The abstract stage: the same model, on OpenRouter's paid tier first (~$0.0015 per abstract; NVIDIA's free copy was
# taking minutes per call on 2026-10-21), NVIDIA free as overflow. Only abstracts that state an age are sent at all.
ABSTRACT_MODEL = ModelConfig(
    provider="openrouter", model="nvidia/nemotron-3-ultra-550b-a55b", max_tokens=4096, requests_per_minute=30,
    overflow=ModelConfig(provider="nvidia", model="nvidia/nemotron-3-ultra-550b-a55b", max_tokens=4096,
                         requests_per_minute=30),
)
# Words and numbers that state an age. An abstract without any cannot show participants under 65 (no quoted number),
# so it passes without a model call: the same outcome the model would give ('unclear' passes at this stage).
AGE_WORDS = re.compile(r"\b(aged?|ages|age range|mean age|median age|years? old|year-old|yrs?)\b"
                       r"|\b[4-9]\d(?:\.\d+)?\s*(?:±|years|yrs|y\b)|[≥>]\s*[4-9]\d", re.I)


def abstract_targets(run_dir: Path) -> list[dict]:
    """Every paper that passed abstract screening (the age rule at the abstract stage)."""
    return [s for s, o in split_by_stage(load_states(run_dir))["abstract"] if o == "passed"]


def check_abstract(state: dict, llm: LLM) -> dict:
    """Abstract stage: exclude only when the title/abstract clearly shows participants under 65 (a quoted number);
    'met' or 'unclear' passes the paper on, as for every rule at this stage."""
    a = state["article"]
    article = Article(pmid=state["pmid"], title=a.get("title", ""), abstract=a.get("abstract", "") or "")
    row = {"pmid": state["pmid"], "title": article.title}
    if not article.abstract:
        return row | {"I4 at 65": "unclear", "reason": "no abstract", "abstract decision": "passes"}
    if not AGE_WORDS.search(f"{article.title} {article.abstract}"):
        return row | {"I4 at 65": "unclear", "reason": "the abstract states no age", "abstract decision": "passes"}
    res = llm.structured(AgeAnswer, SYSTEM.replace("research paper", "research paper's title and abstract"),
                         f"TITLE AND ABSTRACT (PMID {article.pmid})\n\n{article.title}\n\n{article.abstract}",
                         role="age_check_abstract")
    i4, reason = decide(res.parsed, article)
    p = res.parsed
    return row | {"I4 at 65": i4, "reason": reason,
                  "abstract decision": "excluded at abstract (I4 under 65)" if i4 == "not_met" else "passes",
                  "min_age": p.min_age.value, "max_age": p.max_age.value, "mean_age": p.mean_age.value,
                  "quotes": " | ".join(q for q in (p.min_age.quote, p.max_age.quote, p.mean_age.quote) if q),
                  "model": res.usage.model}


def run_abstracts(run_dir: Path, model: ModelConfig = ABSTRACT_MODEL, concurrency: int = 6,
                  llm: LLM | None = None) -> list[dict]:
    meta = json.loads((run_dir / "run_meta.json").read_text())["config"]
    llm = llm or build_llm(model, meta.get("cache_dir", "data/cache/llm"))

    def one(s):
        try:
            return check_abstract(s, llm)
        except Exception as e:
            return {"pmid": s["pmid"], "title": s["article"].get("title", ""), "I4 at 65": "failed",
                    "abstract decision": "passes", "reason": f"{type(e).__name__}: {str(e)[:200]}"}

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        rows = list(pool.map(one, abstract_targets(run_dir)))
    cols = ["pmid", "title", "I4 at 65", "abstract decision", "reason", "min_age", "max_age", "mean_age", "quotes", "model"]
    with (run_dir / "age_check_abstract.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)
    return rows


def run(run_dir: Path, second_look: Path | None = None, model: ModelConfig = DEFAULT_MODEL, concurrency: int = 6,
        llm: LLM | None = None) -> list[dict]:
    meta = json.loads((run_dir / "run_meta.json").read_text())["config"]
    llm = llm or build_llm(model, meta.get("cache_dir", "data/cache/llm"))

    def one(pair):
        s, outcome = pair
        try:
            return check(s, outcome, llm)
        except Exception as e:  # a failed call changes nothing; re-run to retry (answered papers are cached)
            return {"pmid": s["pmid"], "title": s["article"].get("title", ""), "before": outcome, "after": outcome,
                    "I4 at 65": "failed", "reason": f"{type(e).__name__}: {str(e)[:200]}"}

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        rows = list(pool.map(one, targets(run_dir, second_look)))
    cols = ["pmid", "title", "before", "I4 at 65", "after", "reason", "age_criterion", "min_age", "max_age", "mean_age",
            "quotes", "note", "model"]
    with (run_dir / "age_check.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adrd_review.age_check")
    ap.add_argument("run_dir")
    ap.add_argument("--second-look", default=None, help="second_look.csv to start from (its included/excluded count)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--stage", choices=["fulltext", "abstract"], default="fulltext",
                    help="abstract: the age rule on the abstracts of every paper that passed abstract screening")
    args = ap.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv()
    if args.stage == "abstract":
        rows = run_abstracts(Path(args.run_dir), concurrency=args.concurrency)
        print(f"{len(rows)} abstracts checked. I4 at 65:", dict(Counter(r["I4 at 65"] for r in rows)))
        print("decision:", dict(Counter(r["abstract decision"] for r in rows)))
        print(f"Saved: {Path(args.run_dir) / 'age_check_abstract.csv'}")
        return 0
    rows = run(Path(args.run_dir), Path(args.second_look) if args.second_look else None, concurrency=args.concurrency)
    print(f"{len(rows)} papers checked. I4 at 65:", dict(Counter(r["I4 at 65"] for r in rows)))
    print("before -> after:", dict(Counter((r["before"], r["after"]) for r in rows if r["before"] != r["after"])))
    print("after:", dict(Counter(r["after"] for r in rows)))
    print(f"Saved: {Path(args.run_dir) / 'age_check.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
