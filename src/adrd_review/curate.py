"""Run the Codebook Curator on a finished CALIBRATION run.

Collects where the run's answers disagree with the gold standard and asks the
Curator for codebook clarifications. Writes curator_proposals.md/.json into the
run folder for the professor to accept or reject. Nothing is applied automatically.
Refuses runs on the test set: the test set must never shape the codebook.

    python -m adrd_review.curate data/runs/E3_gemini/<timestamp> --model google:gemini-3.1-pro-preview
    python -m adrd_review.curate data/runs/E3_fake/<timestamp> --model fake:keyword-rules
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adrd_review.agents import curator
from adrd_review.codebook import load_codebook
from adrd_review.gold import load_gold
from adrd_review.llm import ModelConfig, build_llm
from adrd_review.tools.pubmed import load_cached

MAX_ITEMS = 40


def collect_disagreements(run_dir: Path, gold_path: str, codebook, corpus_dir: str = "data/corpus") -> list[dict]:
    gold = {g.pmid: g for g in load_gold(gold_path, codebook)}
    items = []
    for line in (run_dir / "decisions.jsonl").read_text().splitlines():
        d = json.loads(line)
        g = gold.get(d["pmid"])
        if g is None:
            continue
        art = load_cached(d["pmid"], corpus_dir)
        base = {"pmid": d["pmid"], "title": art.title if art else "", "abstract": art.abstract if art else ""}
        agent = d["screening_decision"]["labels"]
        if agent != [g.screening]:
            votes = [v for v in d.get("screening_votes", []) if v["labels"] == agent]
            items.append(base | {
                "field": "screening",
                "gold": g.screening,
                "agent": ", ".join(agent),
                "rules": sorted({r for v in votes for r in v["rule_ids"]}),
                "reasons": [v["rationale"] for v in votes if v["rationale"]],
            })
        if g.screening == "include" and agent == ["include"]:
            for dim, dec in (d.get("annotation_decisions") or {}).items():
                truth = sorted(g.labels.get(dim, []))
                if truth and sorted(dec["labels"]) != truth:
                    votes = [v for v in d.get("annotation_votes", []) if v["field"] == dim]
                    items.append(base | {
                        "field": dim,
                        "gold": ", ".join(truth),
                        "agent": ", ".join(dec["labels"]) or "(none)",
                        "rules": [],
                        "reasons": [v["rationale"] for v in votes if v["rationale"]],
                    })
    return items[:MAX_ITEMS]


def to_markdown(answer, codebook_version: str, run_dir: Path, n_items: int) -> str:
    lines = [
        "# Codebook Curator proposals",
        "",
        f"Run: `{run_dir}` · Codebook version: `{codebook_version}` · Disagreements reviewed: {n_items}",
        "",
        "**For the professor to accept or reject.** Accepted changes go into `configs/codebook.yaml` "
        "with a new version number; then the calibration runs are repeated.",
        "",
    ]
    for i, p in enumerate(answer.proposals, start=1):
        lines += [
            f"## {i}. {p.target}",
            "",
            f"**Problem:** {p.problem}",
            "",
            f"**Suggested:** {p.suggested_text}",
            "",
            f"**Examples:** {', '.join(p.example_pmids) or '-'}",
            "",
            "Decision: [ ] accept  [ ] reject  [ ] edit",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser(prog="python -m adrd_review.curate")
    ap.add_argument("run_dir")
    ap.add_argument("--model", required=True, help="provider:model, e.g. google:gemini-3.1-pro-preview")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    if meta["part"] != "calibration":
        print(f"Refusing: this run is on the {meta['part']!r} set. The Curator may only learn from calibration runs.")
        return 1
    cfg = meta["config"]
    codebook = load_codebook(cfg["codebook"])
    items = collect_disagreements(run_dir, cfg["gold"], codebook, cfg.get("corpus_dir", "data/corpus"))
    if not items:
        print("No disagreements with the gold standard in this run: nothing to propose.")
        return 0

    provider, model = args.model.split(":", 1)
    llm = build_llm(ModelConfig(provider=provider, model=model, max_tokens=4096), cfg.get("cache_dir"), codebook)
    answer, usage = curator.run(codebook, items, llm)

    (run_dir / "curator_proposals.json").write_text(
        json.dumps({"model": llm.config.label, "items": len(items), "usage": usage.model_dump(),
                    "proposals": [p.model_dump() for p in answer.proposals]}, indent=2) + "\n"
    )
    (run_dir / "curator_proposals.md").write_text(to_markdown(answer, codebook.version, run_dir, len(items)))
    print(f"{len(answer.proposals)} proposals from {len(items)} disagreements -> {run_dir / 'curator_proposals.md'}")
    if provider == "fake":
        print("NOTE: fake model; proposals are placeholders.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
