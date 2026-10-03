"""Run a pipeline over a set of articles, save everything, and score it.

    python -m adrd_review.graph.runner configs/runs/E2_fake.yaml
    python -m adrd_review.graph.runner configs/runs/E3_gemini.yaml --limit 5

Each run writes to data/runs/<name>/<timestamp>/:
    predictions.csv   gold format; scored against the gold standard
    decisions.jsonl   every vote, debate round, adjudication, verification, escalation and
                      human answer, one article per line
    usage.jsonl       one line per model call (tokens, cost, seconds, cached)
    failures.json     articles that errored (predicted 'include' so none are silently dropped)
    report.json       scores, cost/time, panel agreement, escalations, debate statistics, and the
                      flow: how many articles passed each stage and why the rest were excluded
    flow.png          the flow as a PRISMA-style diagram
    run_meta.json     config, codebook/gold/split hashes, git commit, timings
With `human: queue`, articles needing a person pause; answer them with
    python -m adrd_review.human_review data/runs/<name>/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from adrd_review.agents.human import GoldHuman, QueueHuman
from adrd_review.codebook import Codebook, load_codebook
from adrd_review.consensus import panel_agreement
from adrd_review.eval.metrics import usage_summary
from adrd_review.eval.flow import flow_counts, flow_figure, format_flow
from adrd_review.eval.score import evaluate, format_report
from adrd_review.freeze import prompts_sha
from adrd_review.gold import GoldRecord, file_sha256, load_gold, write_gold
from adrd_review.graph.pipeline import PipelineOptions, Reviewer, build_pipeline, finally_included
from adrd_review.llm import LLM, ModelConfig, build_llm
from adrd_review.schemas import Article
from adrd_review.split import load_split
from adrd_review.tools.pubmed import fetch_articles


class RoleConfig(BaseModel):
    model: ModelConfig
    persona: str | None = None


class ReviewerConfig(BaseModel):
    id: str
    model: ModelConfig
    annotation_model: ModelConfig | None = None  # defaults to `model`
    persona: str | None = None
    weight: float = 1.0


class VerifierConfig(BaseModel):
    enabled: bool = True
    model: ModelConfig | None = None  # None = quote check only, no model call


class RunConfig(BaseModel):
    name: str
    condition: str = "E1"
    description: str = ""
    codebook: str = "configs/codebook.yaml"
    # A labelled run uses the gold file and split. An UNLABELLED run (e.g. the professor's 1,855 search
    # results) sets `gold: null` and `pmids_file`: it reports how many articles pass each stage, no scores.
    gold: str | None = "data/gold/gold.csv"
    split: str = "configs/splits/placeholder.json"
    part: str = "calibration"
    pmids_file: str | None = None  # one PMID per line; used instead of the split when set
    concurrency: int = 4
    cache_dir: str = "data/cache/llm"
    corpus_dir: str = "data/corpus"
    # triage and extraction always come from `roles`. Reviewers come from `panel`;
    # without a panel, roles.screening + roles.annotation form a panel of one (E1).
    roles: dict[str, RoleConfig] = Field(default_factory=dict)
    panel: list[ReviewerConfig] = Field(default_factory=list)
    # E3+: adjudicator on split fields; decisions below `escalate_below` go to a human
    adjudicator: ModelConfig | None = None
    escalate_below: float = 0.6
    verifier: VerifierConfig | None = None
    # Variant D: rounds of debate on split fields before adjudication
    debate_rounds: int = 0
    # E4: who answers escalated decisions. gold = the gold label stands in for a person;
    # queue = the run pauses until a person answers with adrd_review.human_review
    human: Literal["none", "gold", "queue"] = "none"
    # PubMed Central full text: never (abstract only, like title/abstract screening), when_needed
    # (triage says the abstract is not enough), or always. Excerpt capped at fulltext_max_chars.
    fulltext: Literal["never", "when_needed", "always"] = "never"
    fulltext_max_chars: int = 12000
    # Two-stage review (like the human process): articles kept at abstract screening get their full
    # paper (team PDF -> PubMed Central -> open-access PDF) and a strict full-text eligibility stage;
    # extraction and annotation then read the full paper. Needs a model with a large enough limit.
    eligibility: Literal["off", "fulltext"] = "off"
    # Title screening before the abstract (the professor's team: titles -> abstracts -> full text)
    title_stage: bool = False
    eligibility_max_chars: int = 80000
    pdf_dir: str = "data/pdfs"
    # Reviewer weights computed on a calibration run (python -m adrd_review.eval.weights);
    # overrides the `weight:` of each panel reviewer it lists.
    weights_file: str | None = None

    def reviewer_configs(self) -> list[ReviewerConfig]:
        if self.panel:
            return self.panel
        missing = [r for r in ("screening", "annotation") if r not in self.roles]
        if missing:
            raise ValueError(f"Config has no panel and no roles for {missing}")
        s, a = self.roles["screening"], self.roles["annotation"]
        return [ReviewerConfig(id="A", model=s.model, annotation_model=a.model, persona=s.persona)]


def load_run_config(path: str | Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(Path(path).read_text()))


# --------------------------------------------------------------------------- #
# Building the graph from a config
# --------------------------------------------------------------------------- #


def build_graph(config: RunConfig, codebook: Codebook, gold: list[GoldRecord], checkpointer=None, repeat: int = 0):
    # One LLM object per distinct model config, so reviewers sharing a model share its rate limiter.
    pool: dict[str, LLM] = {}

    def llm_for(cfg: ModelConfig) -> LLM:
        key = cfg.model_dump_json()
        if key not in pool:
            pool[key] = build_llm(cfg, config.cache_dir, codebook, namespace=f"repeat-{repeat}" if repeat else "")
        return pool[key]

    for role in ("triage", "extraction"):
        if role not in config.roles:
            raise ValueError(f"Config needs roles.{role}")
    learned = {}
    if config.weights_file:
        learned = {rid: s["weight"] for rid, s in json.loads(Path(config.weights_file).read_text())["reviewers"].items()}
    reviewers = [
        Reviewer(
            id=rc.id,
            llm=llm_for(rc.model),
            annotation_llm=llm_for(rc.annotation_model or rc.model),
            persona=rc.persona,
            weight=learned.get(rc.id, rc.weight),
        )
        for rc in config.reviewer_configs()
    ]
    if config.human == "gold" and not gold:
        raise ValueError("human: gold needs a gold standard; use none or queue for an unlabelled corpus")
    human = {"none": None, "gold": GoldHuman({g.pmid: g for g in gold}), "queue": QueueHuman()}[config.human]
    if config.human == "queue" and checkpointer is None:
        raise ValueError("human: queue needs a checkpointer")
    v = config.verifier
    options = PipelineOptions(
        adjudicator=llm_for(config.adjudicator) if config.adjudicator else None,
        escalate_below=config.escalate_below,
        verify=bool(v and v.enabled),
        verifier=llm_for(v.model) if v and v.enabled and v.model else None,
        debate_rounds=config.debate_rounds,
        human=human,
        fulltext=config.fulltext,
        fulltext_max_chars=config.fulltext_max_chars,
        title_stage=config.title_stage,
        eligibility=config.eligibility == "fulltext",
        eligibility_max_chars=config.eligibility_max_chars,
        fulltext_source=_fulltext_source(config),
    )
    graph = build_pipeline(
        codebook,
        llm_for(config.roles["triage"].model),
        llm_for(config.roles["extraction"].model),
        reviewers,
        options,
        checkpointer=checkpointer,
    )
    return graph, reviewers


def _fulltext_source(config: RunConfig):
    from adrd_review.tools.fulltext import get_fulltext

    return lambda article: get_fulltext(article, pdf_dir=config.pdf_dir)


def open_checkpointer(path: Path):
    """SQLite checkpointer for paused runs. Our record types are registered explicitly,
    so LangGraph will read them back (it blocks unregistered types from checkpoints)."""
    from pydantic import BaseModel as _Base

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver

    from adrd_review import schemas

    allowed = [
        ("adrd_review.schemas", name)
        for name, obj in vars(schemas).items()
        if isinstance(obj, type) and issubclass(obj, _Base) and obj.__module__ == "adrd_review.schemas"
    ]
    serde = JsonPlusSerializer(allowed_msgpack_modules=allowed)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False), serde=serde)


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def to_prediction(state: dict, two_stage: bool = False) -> GoldRecord:
    """Turn a finished graph state into one row of predictions.csv."""
    decision = state["screening_decision"]
    include = decision.labels == ["include"]
    if decision.include_score is not None:
        score = decision.include_score
    else:
        conf = decision.confidence if decision.confidence is not None else 0.5
        score = conf if include else 1 - conf
    # Two-stage runs also report the full-text decision. An unresolved "unclear" counts as
    # include (flagged for a human), and an article excluded at stage 1 is excluded at stage 2.
    fulltext = None
    elig = state.get("eligibility_decision")
    if two_stage:
        fulltext = "exclude" if not include or (elig is not None and elig.labels == ["exclude"]) else "include"
    labels = {}
    if finally_included(state):
        labels = {f: d.labels for f, d in (state.get("annotation_decisions") or {}).items() if d.labels}
    return GoldRecord(
        pmid=state["article"].pmid,
        screening="include" if include else "exclude",
        fulltext=fulltext,
        labels=labels,
        score=round(score, 4),
    )


def run_articles(graph, articles: list[Article], concurrency: int = 4, threaded_config: bool = False):
    """Invoke the graph on each article in parallel. Returns (states, failures, paused PMIDs)."""

    def one(article: Article):
        cfg = {"configurable": {"thread_id": article.pmid}} if threaded_config else None
        try:
            state = graph.invoke({"article": article}, cfg)
            return state, None
        except Exception as e:
            return None, {"pmid": article.pmid, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}

    states, failures, paused = [], [], []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for i, (state, fail) in enumerate(pool.map(one, articles), start=1):
            if fail:
                failures.append(fail)
                print(f"  [{i}/{len(articles)}] {fail['pmid']} FAILED: {fail['error'][:120]}", file=sys.stderr)
            elif state.get("__interrupt__"):
                paused.append(state["article"].pmid)
                print(f"  [{i}/{len(articles)}] {state['article'].pmid} waiting for a human", file=sys.stderr)
            else:
                states.append(state)
                print(f"  [{i}/{len(articles)}] {state['article'].pmid} {_progress(state)}", file=sys.stderr)
    return states, failures, paused


def _progress(state: dict) -> str:
    """One line per finished article: how far it got."""
    d = state["screening_decision"]
    if d.method == "title_excluded":
        where = "excluded at title"
    elif d.labels != ["include"]:
        where = f"excluded at abstract [{d.method}]"
    elif state.get("eligibility_decision") is not None:
        e = state["eligibility_decision"]
        where = {"include": "INCLUDED", "exclude": "excluded at full text", "unclear": "kept, a person decides"}[e.labels[0]]
        where += f" [{e.method}]"
    else:
        where = f"passed abstract [{d.method}]"
    esc = f"  ({len(state.get('escalations', []))} escalated)" if state.get("escalations") else ""
    return where + esc


def _dump(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items() if not k.startswith("__")}
    if isinstance(obj, list):
        return [_dump(v) for v in obj]
    return obj


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def debate_stats(states: list[dict], gold_by: dict[str, GoldRecord]) -> dict | None:
    """How votes moved during debate: changes accepted/rejected, and toward or away from gold."""
    rounds = [s for s in states if s.get("screening_round") or s.get("annotation_round")]
    if not rounds:
        return None
    changes = Counter()
    for s in states:
        g = gold_by.get(s["article"].pmid)
        for field_votes in (s.get("screening_votes", []), s.get("annotation_votes", [])):
            by_reviewer: dict[tuple, list] = {}
            for v in field_votes:
                by_reviewer.setdefault((v.reviewer_id, v.field), []).append(v)
            for (_rid, fld), vs in by_reviewer.items():
                vs.sort(key=lambda v: v.round)
                first, last = vs[0], vs[-1]
                changes["rejected"] += sum("rejected" in w for v in vs[1:] for w in v.warnings)
                if len(vs) == 1 or sorted(first.labels) == sorted(last.labels):
                    continue
                changes["changed"] += 1
                if g is None:
                    continue
                truth = [g.screening] if fld == "screening" else (g.labels.get(fld, []) if g.screening == "include" else [])
                before, after = sorted(first.labels) == sorted(truth), sorted(last.labels) == sorted(truth)
                changes["toward_gold" if after and not before else "away_from_gold" if before and not after else "neutral"] += 1
    return {"articles_debated": len(rounds), **dict(changes)}


def write_outputs(out: Path, config: RunConfig, codebook: Codebook, gold: list[GoldRecord], pmids: list[str],
                  states: list[dict], failures: list[dict], paused: list[str], wall: float, reviewers: list[Reviewer]) -> dict:
    two_stage = config.eligibility == "fulltext"
    preds = [to_prediction(s, two_stage) for s in states]
    # Recall guard: a failed article is predicted 'include' so it goes to a human, never silently dropped.
    preds += [
        GoldRecord(pmid=f["pmid"], screening="include", fulltext="include" if two_stage else None, score=0.5, source="FAILED")
        for f in failures
    ]
    order = {p: i for i, p in enumerate(pmids)}
    preds.sort(key=lambda r: order[r.pmid])
    write_gold(preds, out / "predictions.csv", dimensions=list(codebook.dimensions))

    with (out / "decisions.jsonl").open("w") as fh:
        for s in states:
            fh.write(json.dumps(_dump({k: v for k, v in s.items() if k != "usage"} | {"pmid": s["article"].pmid})) + "\n")
    usages = [u.model_dump() | {"pmid": s["article"].pmid} for s in states for u in s.get("usage", [])]
    with (out / "usage.jsonl").open("w") as fh:
        for u in usages:
            fh.write(json.dumps(u) + "\n")
    (out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")

    report: dict = {"pending_human": paused}
    if not paused and gold:
        report = evaluate(gold, [p for p in preds], codebook, pmids)
    elif not gold:
        report["unlabelled"] = True  # no gold standard: counts only, no scores
    report["flow"] = flow_counts(states, len(failures), codebook, {g.pmid: g for g in gold} if gold else None)
    try:
        flow_figure(report["flow"], out / "flow.png", f"{config.name}: {len(pmids)} articles")
    except Exception as e:  # a figure is never worth failing a run over
        print(f"(flow figure not drawn: {type(e).__name__}: {e})", file=sys.stderr)
    report["usage"] = usage_summary(usages, len(pmids), wall_seconds=round(wall, 2))
    report["failures"] = len(failures)
    report["panel_agreement"] = panel_agreement(states, [r.id for r in reviewers], list(codebook.dimensions))
    methods = Counter(
        f"{d.field}:{d.method}"
        for st in states
        for d in [st["screening_decision"], *([st["eligibility_decision"]] if st.get("eligibility_decision") else []),
                  *(st.get("annotation_decisions") or {}).values()]
    )
    report["decision_methods"] = dict(sorted(methods.items()))
    esc = [e for s in states for e in s.get("escalations", [])]
    report["escalations"] = {
        "total": len(esc),
        "articles": sum(1 for s in states if s.get("escalations")),
        "rate": round(sum(1 for s in states if s.get("escalations")) / max(1, len(states)), 4),
        "by_reason": dict(Counter(e.reason for e in esc)),
        "answered_by_human": sum(len(s.get("human_answers", [])) for s in states),
    }
    report["adjudications"] = sum(len(s.get("adjudications", [])) for s in states)
    if two_stage:
        reached = [s for s in states if (s.get("fulltext_info") or {}).get("stage") == 2]
        sources = Counter((s["fulltext_info"].get("source") or "none") for s in reached)
        needed = [s for s in reached if not s["fulltext_info"].get("used")]
        report["fulltext_sources"] = {
            "reached_stage_2": len(reached),
            "by_source": dict(sources),
            "truncated": sum(bool(s["fulltext_info"].get("truncated")) for s in reached),
            "pdfs_needed": len(needed),
        }
        import csv as _csv

        with (out / "pdfs_needed.csv").open("w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["pmid", "title", "doi", "link", "why_missing", "save_as"])
            for s in needed:
                a, info = s["article"], s["fulltext_info"]
                w.writerow([a.pmid, a.title, a.doi or "", info.get("link") or "", info.get("reason", ""),
                            f"{config.pdf_dir}/{a.pmid}.pdf"])
    report["debate"] = debate_stats(states, {g.pmid: g for g in gold})
    report["warnings"] = sum(
        len(v.warnings)
        for s in states
        for v in s.get("screening_votes", []) + s.get("eligibility_votes", []) + s.get("annotation_votes", [])
    ) + sum(len(a.warnings) for s in states for a in s.get("adjudications", []))
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def print_summary(report: dict, config: RunConfig, part: str, out: Path, usages_cached: int, wall: float) -> None:
    if report.get("pending_human"):
        print(f"\n{len(report['pending_human'])} article(s) are waiting for a human. Answer them with:")
        print(f"  python -m adrd_review.human_review {out}")
    elif not report.get("unlabelled"):
        print()
        print(format_report(report, f"{config.name} on {part}"))
    print()
    print(format_flow(report["flow"]))
    u = report["usage"]
    has_prices = u["total"]["cost_usd"] not in (0, 0.0, None)
    cost = f"${u['total']['cost_usd']:.4f}" if has_prices else "n/a (no prices in config)"
    print("\nCOST AND TIME")
    print(
        f"  model calls {u['calls']} ({usages_cached} from cache)   wall time {wall:.1f}s   "
        f"tokens in/out {u['total']['input_tokens']}/{u['total']['output_tokens']}   cost {cost}"
    )
    print(f"  failures {report['failures']}   evidence/label warnings {report['warnings']}")
    pa = report["panel_agreement"]
    if pa:
        print(f"\nPANEL AGREEMENT (independent votes of {', '.join(pa['reviewers'])})")
        for fld, v in pa.items():
            if fld != "reviewers":
                print(f"  {fld:17} Fleiss kappa {v['fleiss_kappa']:.3f}   unanimous {v['unanimous']}/{v['items']}")
    print("\nHOW DECISIONS WERE REACHED")
    print("  " + ", ".join(f"{k} {n}" for k, n in report["decision_methods"].items()))
    e = report["escalations"]
    print(
        f"  adjudications {report['adjudications']}   escalated to a human: {e['total']} decisions in "
        f"{e['articles']} articles ({e['rate']:.0%})   {e['by_reason']}   answered: {e['answered_by_human']}"
    )
    if report.get("debate"):
        print(f"  debate: {report['debate']}")
    fs = report.get("fulltext_sources")
    if fs:
        print("\nFULL TEXT (stage 2)")
        print(f"  articles reaching stage 2: {fs['reached_stage_2']}   source: {fs['by_source']}   shortened to fit: {fs['truncated']}")
        if fs["pdfs_needed"]:
            print(f"  {fs['pdfs_needed']} paper(s) had no full text: see {out / 'pdfs_needed.csv'} (save each PDF as shown, then re-run)")
    if any(rc.model.provider == "fake" for rc in config.reviewer_configs()):
        print("\nNOTE: this run used the FAKE keyword model. Scores check the plumbing only; they are not results.")
    if config.human == "gold":
        print("NOTE: human answers came from the gold standard (simulated reviewer): an upper bound for E4.")
    print(f"\nSaved in {out}/")


def select_pmids(config: RunConfig, part: str, gold: list[GoldRecord]) -> list[str]:
    """The articles to run: the PMID list of an unlabelled corpus, or one part of the gold split."""
    if config.pmids_file:
        lines = Path(config.pmids_file).read_text().split()
        return list(dict.fromkeys(p.strip() for p in lines if p.strip().isdigit()))
    if not gold:
        raise ValueError("A run needs a gold file (labelled) or a pmids_file (unlabelled)")
    return load_split(config.split)[part] if part != "all" else [g.pmid for g in gold]


def execute(
    config: RunConfig,
    part: str | None = None,
    limit: int | None = None,
    out_root: str = "data/runs",
    config_path: str | None = None,
    repeat: int = 0,
) -> Path:
    """repeat = 0: normal run (shares the response cache). repeat >= 1: an independent
    repeat whose model answers are cached separately, so the model is really called again."""
    part = part or config.part
    if part == "test":
        from adrd_review.freeze import check

        problems = check(config_path) if config_path else ["run the test set from a config file"]
        if problems:
            raise SystemExit("Refusing to run on the TEST set: " + "; ".join(problems))
    codebook: Codebook = load_codebook(config.codebook)
    gold = load_gold(config.gold, codebook) if config.gold else []
    pmids = select_pmids(config, part, gold)
    if limit:
        pmids = pmids[:limit]

    articles_by, fetch_report = fetch_articles(pmids, cache_dir=config.corpus_dir)
    if fetch_report.missing:
        raise RuntimeError(f"Articles not available: {fetch_report.missing}")
    articles = [articles_by[p] for p in pmids]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(out_root) / config.name / (f"{stamp}-r{repeat}" if repeat else stamp)
    out.mkdir(parents=True, exist_ok=True)
    checkpointer = open_checkpointer(out / "checkpoints.sqlite") if config.human == "queue" else None
    graph, reviewers = build_graph(config, codebook, gold, checkpointer, repeat)
    print(
        f"Run {config.name} ({config.condition}) on {part}: {len(articles)} articles, "
        f"{len(reviewers)} reviewer(s) -> {out}",
        file=sys.stderr,
    )

    started = time.monotonic()
    states, failures, paused = run_articles(graph, articles, config.concurrency, threaded_config=checkpointer is not None)
    wall = time.monotonic() - started

    meta = {
        "name": config.name,
        "condition": config.condition,
        "part": part,
        "pmids": pmids,
        "limit": limit,
        "repeat": repeat,
        "started_utc": stamp,
        "wall_seconds": round(wall, 2),
        "failures": len(failures),
        "git_commit": _git_commit(),
        "codebook_version": codebook.version,
        "codebook_sha256": codebook.sha256,
        "prompts_sha256": prompts_sha(),  # agent code + prompts: runs are only averaged if this matches
        "gold_sha256": file_sha256(config.gold) if config.gold else None,
        "pmids_file": config.pmids_file,
        "split_file": config.split,
        "config": config.model_dump(),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if paused:
        (out / "failures_pending.json").write_text(json.dumps(failures, indent=2) + "\n")
    report = write_outputs(out, config, codebook, gold, pmids, states, failures, paused, wall, reviewers)
    cached = sum(u.cached for s in states for u in s.get("usage", []))
    print_summary(report, config, part, out, cached, wall)
    return out


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser(prog="python -m adrd_review.graph.runner")
    ap.add_argument("config", help="Run config YAML, e.g. configs/runs/E2_fake.yaml")
    ap.add_argument("--part", choices=["calibration", "test", "all"], help="Overrides the config")
    ap.add_argument("--limit", type=int, help="Only the first N articles (quick checks)")
    ap.add_argument("--repeat", type=int, default=0, help="Repeat number (1, 2, 3...): calls the model again instead of using cached answers")
    args = ap.parse_args(argv)
    if args.part == "test":
        print("NOTE: running on the TEST set. Only do this for final, frozen runs.\n", file=sys.stderr)
    execute(load_run_config(args.config), part=args.part, limit=args.limit, config_path=args.config, repeat=args.repeat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
