"""Answer the decisions a run sent to a human, then finish the run.

Used with `human: queue` in a run config. The run pauses on each escalated
decision (LangGraph `interrupt`); this command shows each one, takes the
answer, resumes the pipeline from its saved checkpoint, and when nothing is
left, writes predictions.csv and report.json.

    python -m adrd_review.human_review data/runs/E4_queue_fake/<timestamp>
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path

from langgraph.types import Command

from adrd_review.agents.common import allowed_labels
from adrd_review.codebook import load_codebook
from adrd_review.gold import load_gold

AnswerFn = Callable[[dict, list[str], bool], list[str]]


def terminal_answer(item: dict, allowed: list[str], multi: bool) -> list[str]:
    """Show one escalated decision in the terminal and read the answer."""
    print("\n" + "=" * 78)
    print(f"PMID {item['pmid']}: {item['title']}")
    print(f"Decision needed: {item['field']}   (why: {item['reason']}; {item['detail']})")
    print(f"Current answer: {', '.join(item['current']) or '(none)'}")
    print("\nAbstract:\n" + textwrap.fill(item.get("abstract") or "(no abstract)", 78)[:2500])
    print("\nReviewers:")
    for v in item["votes"]:
        print(f"  {v['reviewer']} (round {v['round']}): {', '.join(v['labels']) or '(no label)'}  {v['rationale'][:160]}")
        for q in v["evidence"][:2]:
            print(f'      "{q[:150]}"')
    hint = "several allowed, comma-separated; 'none' for no label" if multi else "exactly one"
    while True:
        raw = input(f"\nYour answer [{' / '.join(allowed)}] ({hint}): ").strip()
        labels = [] if raw.lower() == "none" and multi else [x.strip() for x in raw.split(",") if x.strip()]
        bad = [x for x in labels if x not in allowed]
        if bad:
            print(f"Not valid: {bad}")
        elif not multi and len(labels) != 1:
            print("Give exactly one answer.")
        else:
            return labels


def resume_run(run_dir: str | Path, answer: AnswerFn = terminal_answer) -> dict:
    # Imported here to avoid a circular import (runner imports agents that import this package).
    from adrd_review.graph.runner import RunConfig, build_graph, open_checkpointer, print_summary, write_outputs

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    config = RunConfig.model_validate(meta["config"])
    if config.human != "queue":
        raise SystemExit("This run did not use `human: queue`; nothing to answer.")
    codebook = load_codebook(config.codebook)
    gold = load_gold(config.gold, codebook)
    graph, reviewers = build_graph(
        config, codebook, gold, open_checkpointer(run_dir / "checkpoints.sqlite"), meta.get("repeat", 0)
    )
    pmids: list[str] = meta["pmids"]

    started = time.monotonic()
    answered = 0
    for pmid in pmids:
        cfg = {"configurable": {"thread_id": pmid}}
        snap = graph.get_state(cfg)
        while snap.interrupts:
            item = snap.interrupts[0].value
            fld = item["field"]
            multi = fld not in ("screening", "eligibility") and codebook.dimensions[fld].multi_label
            allowed = [x for x in allowed_labels(codebook, fld) if x != "unclear"]
            labels = answer(item, allowed, multi)
            graph.invoke(Command(resume=labels), cfg)
            answered += 1
            snap = graph.get_state(cfg)

    failures_file = run_dir / "failures_pending.json"
    failures = json.loads(failures_file.read_text()) if failures_file.exists() else []
    failed = {f["pmid"] for f in failures}
    states = []
    for pmid in pmids:
        if pmid in failed:
            continue
        values = graph.get_state({"configurable": {"thread_id": pmid}}).values
        if values.get("screening_decision"):
            states.append(values)
    wall = meta["wall_seconds"] + (time.monotonic() - started)
    report = write_outputs(run_dir, config, codebook, gold, pmids, states, failures, [], wall, reviewers)
    report["human_answers_this_session"] = answered
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    cached = sum(u.cached for s in states for u in s.get("usage", []))
    print_summary(report, config, meta["part"], run_dir, cached, wall)
    return report


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("Usage: python -m adrd_review.human_review <run folder>", file=sys.stderr)
        return 2
    resume_run(args[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
