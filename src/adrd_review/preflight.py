"""Preflight check: find problems BEFORE a run, not halfway through it.

    python -m adrd_review.preflight configs/runs/E4_twostage_gemini.yaml
    python -m adrd_review.preflight configs/runs/E2_groq.yaml --part calibration --no-calls

Checks, each reported as PASS / WARN / FAIL:
  1. config, codebook, gold file and split (or the PMID list of an unlabelled corpus) load and agree
  2. API keys are set for every provider the config uses
  3. every model actually answers a tiny structured call (skip with --no-calls)
  4. every article is downloaded
  5. two-stage runs: where each article's full text will come from
  6. a single call fits each model's token limit (abstract and full-text sizes)
  7. estimated calls per model against daily limits, and estimated time
  8. estimated cost (configs/prices.yaml)
  9. test-set runs: the configuration is frozen
 10. enough free disk space
Exit code 1 if anything FAILs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from pydantic import BaseModel

from adrd_review.agents import eligibility, screening, title_screening
from adrd_review.agents.common import render_article
from adrd_review.codebook import load_codebook
from adrd_review.gold import load_gold
from adrd_review.graph.runner import RunConfig, load_run_config, select_pmids
from adrd_review.llm import LangChainLLM, ModelConfig, ResponseCache
from adrd_review.tools.pubmed import load_cached

KEYS = {"google": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
FULLTEXT_TOKENS = 9700  # median body of a real paper (docs/FULLTEXT_PLAN.md); 90th pct ~14,100
OUTPUT_TOKENS = 600
# Unlabelled corpora: assumed share passing each stage, for the call/cost estimate only (deliberately
# high, so the estimate is an upper bound). Labelled runs use the gold standard's include rate instead.
ASSUMED_TITLE_PASS = 0.8
ASSUMED_ABSTRACT_PASS = 0.5


class Ping(BaseModel):
    ok: bool


class Report:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def add(self, level: str, text: str):
        self.lines.append((level, text))
        print(f"[{level:4}] {text}")

    @property
    def failed(self) -> bool:
        return any(level == "FAIL" for level, _ in self.lines)


def models_by_role(cfg: RunConfig) -> dict[str, list[ModelConfig]]:
    panel = cfg.reviewer_configs()
    out = {
        **({"title": [r.model for r in panel]} if cfg.title_stage else {}),
        "triage": [cfg.roles["triage"].model],
        "extraction": [cfg.roles["extraction"].model],
        "screening": [r.model for r in panel],
        "annotation": [r.annotation_model or r.model for r in panel],
    }
    if not cfg.annotate:
        del out["extraction"], out["annotation"]
    if cfg.eligibility == "fulltext":
        out["eligibility"] = [r.model for r in panel]
    if cfg.adjudicator:
        out["adjudicator"] = [cfg.adjudicator]
    if cfg.verifier and cfg.verifier.enabled and cfg.verifier.model:
        out["verifier"] = [cfg.verifier.model]
    return out


def check(config_path: str, part: str | None = None, limit: int | None = None, calls: bool = True) -> Report:
    rep = Report()

    # 1. files load
    try:
        cfg = load_run_config(config_path)
        codebook = load_codebook(cfg.codebook)
        gold = load_gold(cfg.gold, codebook) if cfg.gold else []
        part = part or cfg.part
        pmids = select_pmids(cfg, part, gold)
        if limit:
            pmids = pmids[:limit]
        missing_gold = [p for p in pmids if p not in {g.pmid for g in gold}] if gold else []
        if missing_gold:
            rep.add("FAIL", f"{len(missing_gold)} split PMIDs are not in the gold file")
        else:
            what = f"{part} articles" if gold else f"articles from {cfg.pmids_file} (unlabelled: counts only, no scores)"
            rep.add("PASS", f"config '{cfg.name}' ({cfg.condition}), codebook {codebook.version}, {len(pmids)} {what}")
        if codebook.status == "draft":
            rep.add("WARN", f"codebook {codebook.version} is a DRAFT: results are not reportable until the professor approves it")
        proposed = [r.id for r in codebook.rules if "PROPOSED" in r.source]
        if proposed and codebook.status != "draft":
            rep.add("WARN", f"codebook {codebook.version}: rules {proposed} include PROPOSED parts not yet confirmed by the professor")
    except Exception as e:
        rep.add("FAIL", f"could not load config/codebook/gold/split: {type(e).__name__}: {e}")
        return rep
    roles = models_by_role(cfg)
    distinct: dict[str, ModelConfig] = {m.model_dump_json(): m for ms in roles.values() for m in ms}
    two_stage = cfg.eligibility == "fulltext"

    # 2. keys
    for provider in sorted({m.provider for m in distinct.values()}):
        if provider in ("fake", "ollama"):
            continue
        uses_vertex = any(m.provider == "google" and m.extra.get("vertexai") for m in distinct.values())
        if provider == "google" and uses_vertex:
            rep.add("PASS", "google: Vertex AI settings in config (uses Google Cloud login, no API key)")
        elif os.getenv(KEYS[provider]):
            rep.add("PASS", f"{provider}: {KEYS[provider]} is set")
        else:
            rep.add("FAIL", f"{provider}: {KEYS[provider]} is not set in .env")

    # 3. each model answers
    if calls:
        for m in distinct.values():
            if m.provider == "fake":
                continue
            if m.provider in KEYS and not os.getenv(KEYS[m.provider]) and not m.extra.get("vertexai"):
                rep.add("FAIL", f"{m.label}: not called (no key)")
                continue
            try:
                llm = LangChainLLM(m.model_copy(update={"max_tokens": 600}), cache=ResponseCache(None), max_attempts=1)
                llm.structured(Ping, "Reply with ok = true.", "Are you there?", role="preflight")
                rep.add("PASS", f"{m.label}: answered a structured test call")
            except Exception as e:
                rep.add("FAIL", f"{m.label}: test call failed: {str(e)[:160]}")
    else:
        rep.add("WARN", "model test calls skipped (--no-calls)")

    # 4. articles downloaded
    arts = {p: load_cached(p, cfg.corpus_dir) for p in pmids}
    absent = [p for p, a in arts.items() if a is None]
    if absent:
        rep.add("WARN", f"{len(absent)} article(s) not downloaded yet (the run will fetch them): {absent[:5]}")
    else:
        rep.add("PASS", f"all {len(pmids)} articles are downloaded ({sum(not a.has_abstract for a in arts.values())} without an abstract)")
    have = [a for a in arts.values() if a is not None]

    # 5. full-text sources
    ft_share = 1.0  # share of articles with a full-text source (the rest skip stage 2 when left out)
    if two_stage:
        pdfs = sum((Path(cfg.pdf_dir) / f"{a.pmid}.pdf").exists() for a in have)
        pmc = sum(bool(a.pmcid) and not (Path(cfg.pdf_dir) / f"{a.pmid}.pdf").exists() for a in have)
        none = len(have) - pdfs - pmc
        if cfg.missing_fulltext == "not_retrieved" and have:
            ft_share = (pdfs + pmc) / len(have)
        level = "PASS" if none == 0 else "WARN"
        rep.add(level, f"full text: {pdfs} team PDF(s) in {cfg.pdf_dir}, {pmc} via PubMed Central, {none} with neither "
                       f"(those reaching stage 2 are flagged for a human and listed in pdfs_needed.csv)")

    # 6. call size vs token limits
    sample = have[:50] or []
    art_tokens = statistics.median(len(render_article(a)) for a in sample) / 4 if sample else 600
    sys_tokens = len(screening.system_prompt(codebook)) / 4
    abstract_call = int(sys_tokens + art_tokens + OUTPUT_TOKENS)
    title_call = int(len(title_screening.system_prompt(codebook)) / 4 + 40 + OUTPUT_TOKENS)
    fulltext_call = int(len(eligibility.system_prompt(codebook)) / 4 + art_tokens + FULLTEXT_TOKENS + OUTPUT_TOKENS)
    fulltext_roles = {"eligibility", "extraction", "annotation", "adjudicator", "verifier"} if two_stage else set()
    for role, ms in roles.items():
        size = fulltext_call if role in fulltext_roles else title_call if role == "title" else abstract_call
        for m in ms:
            if m.tokens_per_minute and size > m.tokens_per_minute:
                rep.add("FAIL", f"{m.label} as {role}: one call is ~{size:,} tokens but the limit is {m.tokens_per_minute:,} tokens/min")
    if not any(level == "FAIL" and "tokens/min" in text for level, text in rep.lines):
        rep.add("PASS", f"call sizes fit every model's limits (abstract call ~{abstract_call:,} tokens"
                        + (f", full-text call ~{fulltext_call:,}" if two_stage else "") + ")")

    # 7. calls, daily limits, time
    n = len(pmids)
    gold_by = {g.pmid: g for g in gold}
    if gold:
        kept = min(1.0, 1.5 * sum(gold_by[p].screening == "include" for p in pmids) / max(1, n))  # share passing stage 1 (padded)
    else:
        kept = ASSUMED_ABSTRACT_PASS * (ASSUMED_TITLE_PASS if cfg.title_stage else 1)
        rep.add("WARN", f"unlabelled corpus: the estimate assumes {ASSUMED_TITLE_PASS:.0%} pass titles and "
                        f"{ASSUMED_ABSTRACT_PASS:.0%} of those pass abstracts (an upper bound; real counts come from the run)")
    screened = n * (ASSUMED_TITLE_PASS if cfg.title_stage else 1)  # articles reaching abstract screening
    per_role = {"title": n, "triage": screened, "screening": screened, "eligibility": n * kept * ft_share,
                "extraction": n * kept * ft_share, "annotation": n * kept * ft_share, "adjudicator": 0.4 * screened, "verifier": screened}
    calls_by: dict[str, float] = defaultdict(float)
    tokens_by: dict[str, float] = defaultdict(float)
    cfg_by: dict[str, ModelConfig] = {}
    for role, ms in roles.items():
        size = fulltext_call if role in fulltext_roles else title_call if role == "title" else abstract_call
        for m in ms:
            calls_by[m.label] += per_role[role]
            tokens_by[m.label] += per_role[role] * size
            cfg_by[m.label] = m
    minutes = 0.0
    for label, c in calls_by.items():
        m = cfg_by[label]
        if m.requests_per_day and c > m.requests_per_day:
            rep.add("FAIL", f"{label}: ~{c:,.0f} calls needed but the limit is {m.requests_per_day:,}/day (split the run over days, or use --limit)")
        if m.tokens_per_day and tokens_by[label] > m.tokens_per_day:
            rep.add("FAIL", f"{label}: ~{tokens_by[label]:,.0f} tokens needed but the limit is {m.tokens_per_day:,} tokens/day "
                            f"(use --limit, spread the model's roles over other models, or continue tomorrow: finished calls are cached)")
        elif m.tokens_per_day and tokens_by[label] > 0.5 * m.tokens_per_day:
            rep.add("WARN", f"{label}: ~{tokens_by[label]:,.0f} tokens, over half its {m.tokens_per_day:,} tokens/day limit: "
                            f"one run per day on this model, and earlier calls today count too")
        rpm = m.requests_per_minute
        if m.tokens_per_minute:
            rpm = min(rpm or 1e9, m.tokens_per_minute / (tokens_by[label] / c))
        if rpm:
            minutes = max(minutes, c / rpm)
    total_calls = sum(calls_by.values())
    rep.add("PASS" if not any("calls needed" in t or "tokens needed" in t for _, t in rep.lines) else "WARN",
            f"about {total_calls:,.0f} model calls" + (f"; roughly {minutes:,.0f} minutes at the configured rate limits" if minutes else ""))

    # 8. cost
    prices = yaml.safe_load(Path("configs/prices.yaml").read_text())["models"] if Path("configs/prices.yaml").exists() else {}
    unknown = [lab for lab in calls_by if lab not in prices and not lab.startswith("fake:")]
    cost = sum((tokens_by[lab] - calls_by[lab] * OUTPUT_TOKENS) * prices[lab]["input"] / 1e6
               + calls_by[lab] * OUTPUT_TOKENS * prices[lab]["output"] / 1e6 for lab in calls_by if lab in prices)
    if unknown:
        rep.add("WARN", f"no price listed for {unknown}; cost not estimated for them")
    rep.add("PASS", f"estimated cost for this run: about ${cost:,.2f} (rough; cached calls cost nothing)")
    paid_only = [lab for lab in calls_by if lab in prices and not prices[lab].get("free_tier", True)]
    if paid_only:
        rep.add("WARN", f"needs a paid/licensed key (no free tier): {paid_only}")

    # 9. freeze for test runs
    if part == "test":
        from adrd_review.freeze import check as freeze_check

        problems = freeze_check(config_path)
        rep.add("FAIL" if problems else "PASS", "frozen for the test set" if not problems else "not frozen: " + "; ".join(problems))

    # 10. disk
    free_gb = shutil.disk_usage(".").free / 1e9
    rep.add("PASS" if free_gb > 1 else "WARN", f"{free_gb:.1f} GB free disk space")
    return rep


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser(prog="python -m adrd_review.preflight")
    ap.add_argument("config")
    ap.add_argument("--part", choices=["calibration", "test", "all"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-calls", action="store_true", help="Skip the test call to each model")
    args = ap.parse_args(argv)
    rep = check(args.config, args.part, args.limit, calls=not args.no_calls)
    print("\n" + ("NOT READY: fix the FAIL lines above." if rep.failed else "READY to run."))
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
