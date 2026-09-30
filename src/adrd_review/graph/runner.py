"""Run a pipeline over a set of articles, save everything, and score it.

    python -m adrd_review.graph.runner configs/runs/E1_fake.yaml
    python -m adrd_review.graph.runner configs/runs/E1_gemini.yaml --limit 5

Each run writes to data/runs/<name>/<timestamp>/:
    predictions.csv   gold format; scored against the gold standard
    decisions.jsonl   every vote, quote, rationale and warning, one article per line
    usage.jsonl       one line per model call (tokens, cost, seconds, cached)
    failures.json     articles that errored (predicted 'include' so none are silently dropped)
    report.json       scores + cost/time summary
    run_meta.json     config, codebook/gold/split hashes, git commit, timings
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from adrd_review.codebook import Codebook, load_codebook
from adrd_review.eval.metrics import usage_summary
from adrd_review.eval.score import evaluate, format_report
from adrd_review.gold import GoldRecord, file_sha256, load_gold, write_gold
from adrd_review.graph.pipeline import ROLES, build_pipeline
from adrd_review.llm import ModelConfig, build_llm
from adrd_review.schemas import Article
from adrd_review.split import load_split
from adrd_review.tools.pubmed import fetch_articles


class RoleConfig(BaseModel):
    model: ModelConfig
    persona: str | None = None


class RunConfig(BaseModel):
    name: str
    condition: str = "E1"
    description: str = ""
    codebook: str = "configs/codebook.yaml"
    gold: str = "data/gold/gold.csv"
    split: str = "configs/splits/placeholder.json"
    part: str = "calibration"
    concurrency: int = 4
    cache_dir: str = "data/cache/llm"
    corpus_dir: str = "data/corpus"
    roles: dict[str, RoleConfig] = Field(default_factory=dict)


def load_run_config(path: str | Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def to_prediction(state: dict) -> GoldRecord:
    """Turn a finished graph state into one row of predictions.csv."""
    decision = state["screening_decision"]
    include = decision.labels == ["include"]
    conf = decision.confidence if decision.confidence is not None else 0.5
    labels = {}
    if include:
        labels = {f: d.labels for f, d in (state.get("annotation_decisions") or {}).items() if d.labels}
    return GoldRecord(
        pmid=state["article"].pmid,
        screening="include" if include else "exclude",
        labels=labels,
        score=round(conf if include else 1 - conf, 4),
    )


def run_articles(graph, articles: list[Article], concurrency: int = 4) -> tuple[list[dict], list[dict]]:
    """Invoke the graph on each article in parallel. Returns (states, failures)."""

    def one(article: Article):
        try:
            return graph.invoke({"article": article}), None
        except Exception as e:
            return None, {"pmid": article.pmid, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}

    states, failures = [], []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for i, (state, fail) in enumerate(pool.map(one, articles), start=1):
            if fail:
                failures.append(fail)
                print(f"  [{i}/{len(articles)}] {fail['pmid']} FAILED: {fail['error'][:120]}", file=sys.stderr)
            else:
                states.append(state)
                d = state["screening_decision"].labels[0]
                print(f"  [{i}/{len(articles)}] {state['article'].pmid} {d}", file=sys.stderr)
    return states, failures


def _dump(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dump(v) for v in obj]
    return obj


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def execute(config: RunConfig, part: str | None = None, limit: int | None = None, out_root: str = "data/runs") -> Path:
    part = part or config.part
    codebook: Codebook = load_codebook(config.codebook)
    gold = load_gold(config.gold, codebook)
    pmids = load_split(config.split)[part] if part != "all" else [g.pmid for g in gold]
    if limit:
        pmids = pmids[:limit]

    articles_by, report = fetch_articles(pmids, cache_dir=config.corpus_dir)
    if report.missing:
        raise RuntimeError(f"Articles not available: {report.missing}")
    articles = [articles_by[p] for p in pmids]

    llms = {role: build_llm(config.roles[role].model, config.cache_dir, codebook) for role in ROLES if role in config.roles}
    personas = {role: rc.persona for role, rc in config.roles.items() if rc.persona}
    graph = build_pipeline(llms, codebook, personas)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(out_root) / config.name / stamp
    out.mkdir(parents=True, exist_ok=True)
    print(f"Run {config.name} ({config.condition}) on {part}: {len(articles)} articles -> {out}", file=sys.stderr)

    started = time.monotonic()
    states, failures = run_articles(graph, articles, config.concurrency)
    wall = time.monotonic() - started

    preds = [to_prediction(s) for s in states]
    # Recall guard: a failed article is predicted 'include' so it goes to a human, never silently dropped.
    preds += [GoldRecord(pmid=f["pmid"], screening="include", score=0.5, source="FAILED") for f in failures]
    preds.sort(key=lambda r: pmids.index(r.pmid))
    write_gold(preds, out / "predictions.csv", dimensions=list(codebook.dimensions))

    with (out / "decisions.jsonl").open("w") as fh:
        for s in states:
            fh.write(json.dumps(_dump({k: v for k, v in s.items() if k != "usage"} | {"pmid": s["article"].pmid})) + "\n")
    usages = [u.model_dump() | {"pmid": s["article"].pmid} for s in states for u in s.get("usage", [])]
    with (out / "usage.jsonl").open("w") as fh:
        for u in usages:
            fh.write(json.dumps(u) + "\n")
    (out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")

    scored = evaluate(gold, load_gold(out / "predictions.csv", codebook), codebook, pmids)
    scored["usage"] = usage_summary(usages, len(articles), wall_seconds=round(wall, 2))
    scored["failures"] = len(failures)
    scored["warnings"] = sum(len(v.warnings) for s in states for v in s.get("screening_votes", []) + s.get("annotation_votes", []))
    (out / "report.json").write_text(json.dumps(scored, indent=2) + "\n")

    meta = {
        "name": config.name,
        "condition": config.condition,
        "part": part,
        "n_articles": len(articles),
        "limit": limit,
        "started_utc": stamp,
        "wall_seconds": round(wall, 2),
        "failures": len(failures),
        "git_commit": _git_commit(),
        "codebook_version": codebook.version,
        "codebook_sha256": codebook.sha256,
        "gold_sha256": file_sha256(config.gold),
        "split_file": config.split,
        "config": config.model_dump(),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print()
    print(format_report(scored, f"{config.name} on {part}"))
    u = scored["usage"]
    has_prices = any(x.get("cost_usd") is not None for x in usages)
    cost = f"${u['total']['cost_usd']:.4f}" if has_prices else "n/a (no prices in config)"
    cached = sum(x.get("cached", False) for x in usages)
    print("\nCOST AND TIME")
    print(
        f"  model calls {u['calls']} ({cached} from cache)   wall time {wall:.1f}s   "
        f"tokens in/out {u['total']['input_tokens']}/{u['total']['output_tokens']}   cost {cost}"
    )
    print(f"  failures {len(failures)}   evidence/label warnings {scored['warnings']}")
    if any(rc.model.provider == "fake" for rc in config.roles.values()):
        print("\nNOTE: this run used the FAKE keyword model. Scores check the plumbing only; they are not results.")
    print(f"\nSaved in {out}/")
    return out


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser(prog="python -m adrd_review.graph.runner")
    ap.add_argument("config", help="Run config YAML, e.g. configs/runs/E1_fake.yaml")
    ap.add_argument("--part", choices=["calibration", "test", "all"], help="Overrides the config")
    ap.add_argument("--limit", type=int, help="Only the first N articles (quick checks)")
    args = ap.parse_args(argv)
    if args.part == "test":
        print("NOTE: running on the TEST set. Only do this for final, frozen runs.\n", file=sys.stderr)
    execute(load_run_config(args.config), part=args.part, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
