"""How many articles pass each stage, and why the others were excluded (a PRISMA flow).

    title screening -> abstract screening -> full text (sought / retrieved / assessed) -> included

Computed from the finished graph states of a run, so it works on any corpus, labelled or not.
With a gold standard, each stage also reports how many of the gold standard's includes it
removed ("lost"): the price of that stage in recall.

    python -m adrd_review.eval.flow data/runs/<name>/<timestamp>     # re-print a saved run's flow
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from adrd_review.codebook import Codebook

HUMAN = "human"
FLAGGED = ("no_fulltext", "unclear")
NOT_RETRIEVED = "not_retrieved"  # decision method for a paper whose full text could not be obtained and was left out


def _reason(state: dict, field: str, codebook: Codebook | None) -> str:
    """The main reason an article was excluded at one stage: the rule cited by most of the
    reviewers who excluded it (codebook order breaks ties), or 'human' if a person decided."""
    d = state.get(f"{field}_decision")
    if d is not None and d.method == HUMAN:
        return HUMAN
    votes = [v for v in state.get(f"{field}_votes", []) if v.labels == ["exclude"]]
    latest: dict[str, object] = {}
    for v in sorted(votes, key=lambda v: v.round):
        latest[v.reviewer_id] = v
    counts = Counter(r for v in latest.values() for r in v.rule_ids)
    if not counts:
        return "adjudicator" if d is not None and d.method == "adjudicator" else "unspecified"
    order = {r.id: i for i, r in enumerate(codebook.rules)} if codebook else {}
    return min(counts, key=lambda r: (-counts[r], order.get(r, 99), r))


def _named(reasons: Counter, codebook: Codebook | None) -> dict[str, int]:
    names = {r.id: f"{r.id} {r.name}" for r in codebook.rules} if codebook else {}
    names[HUMAN] = "decided by a person"
    return {names.get(k, k): n for k, n in reasons.most_common()}


def flow_counts(states: list[dict], failures: int = 0, codebook: Codebook | None = None,
                gold_by: dict | None = None) -> dict:
    """Counts per stage. `gold_by` (PMID -> GoldRecord) adds how many gold includes each stage lost."""
    gold_by = gold_by or {}

    def gold_final(pmid: str) -> bool | None:
        g = gold_by.get(pmid)
        if g is None:
            return None
        return (g.fulltext or g.screening) == "include"

    def lost(group: list[dict]) -> dict:
        if not gold_by:
            return {}
        known = [s for s in group if s["article"].pmid in gold_by]
        return {
            "gold_abstract_includes_lost": sum(gold_by[s["article"].pmid].screening == "include" for s in known),
            "gold_final_includes_lost": sum(bool(gold_final(s["article"].pmid)) for s in known),
        }

    out: dict = {"identified": len(states) + failures, "failed": failures}
    pool = list(states)

    # Stage 0: titles
    titled = [s for s in pool if s.get("title_decision") is not None]
    if titled:
        t_out = [s for s in titled if s["title_decision"].labels == ["exclude"]]
        out["title"] = {
            "screened": len(titled),
            "excluded": len(t_out),
            "passed": len(titled) - len(t_out),
            "reasons": _named(Counter(_reason(s, "title", codebook) for s in t_out), codebook),
            **lost(t_out),
        }
        pool = [s for s in pool if s not in t_out]

    # Stage 1: abstracts
    a_out = [s for s in pool if s["screening_decision"].labels != ["include"]]
    out["abstract"] = {
        "screened": len(pool),
        "excluded": len(a_out),
        "passed": len(pool) - len(a_out),
        "decided_by_human": sum(s["screening_decision"].method == HUMAN for s in pool),
        "reasons": _named(Counter(_reason(s, "screening", codebook) for s in a_out), codebook),
        **lost(a_out),
    }
    pool = [s for s in pool if s not in a_out]

    # Stage 2: full text (two-stage runs)
    reached = [s for s in pool if (s.get("fulltext_info") or {}).get("stage") == 2]
    if reached:
        retrieved = [s for s in reached if s["fulltext_info"].get("used")]
        # Not retrieved and left out (missing_fulltext: not_retrieved): PRISMA "reports not retrieved".
        dropped = [s for s in reached if s["eligibility_decision"].method == NOT_RETRIEVED]
        f_out = [s for s in reached if s["eligibility_decision"].labels == ["exclude"] and s not in dropped]
        open_ = [s for s in reached if s not in f_out and s["eligibility_decision"].labels == ["unclear"]]
        out["fulltext"] = {
            "sought": len(reached),
            "retrieved": len(retrieved),
            "by_source": dict(Counter(s["fulltext_info"].get("source") for s in retrieved)),
            "not_retrieved": len(reached) - len(retrieved),
            "assessed_by_agents": sum(bool(s.get("eligibility_votes")) for s in reached),
            "not_retrieved_left_out": len(dropped),
            "excluded": len(f_out),
            "passed": len(reached) - len(f_out) - len(dropped),
            "decided_by_human": sum(s["eligibility_decision"].method == HUMAN for s in reached),
            "still_open": {
                "no_fulltext": sum(s["eligibility_decision"].method == "no_fulltext" for s in open_),
                "unclear": sum(s["eligibility_decision"].method != "no_fulltext" for s in open_),
            },
            "reasons": _named(Counter(_reason(s, "eligibility", codebook) for s in f_out), codebook),
            **lost(f_out),
        }
        pool = [s for s in pool if s not in f_out and s not in dropped]

    final_open = sum(
        s.get("eligibility_decision") is not None and s["eligibility_decision"].labels == ["unclear"] for s in pool
    )
    out["included"] = {
        "total": len(pool) + failures,
        "confirmed": len(pool) - final_open,
        "waiting_for_human": final_open,
        "failed_kept": failures,
    }
    if gold_by:
        kept = [s for s in pool if s["article"].pmid in gold_by]
        out["included"]["gold_final_includes_kept"] = sum(bool(gold_final(s["article"].pmid)) for s in kept)
        out["included"]["gold_final_includes_total"] = sum(
            bool(gold_final(s["article"].pmid)) for s in states if s["article"].pmid in gold_by
        )
    return out


def format_flow(flow: dict) -> str:
    def reasons(d: dict) -> list[str]:
        return [f"        {n:5}  {r}" for r, n in d.get("reasons", {}).items()]

    def lost(d: dict, final_only: bool = False) -> str:
        if "gold_final_includes_lost" not in d:
            return ""
        if final_only:  # at full text, dropping abstract-stage includes is the point; only final ones count
            return f"   [gold final includes lost here: {d['gold_final_includes_lost']}]"
        return (f"   [gold includes lost here: {d['gold_abstract_includes_lost']} abstract-stage, "
                f"{d['gold_final_includes_lost']} final]")

    lines = ["ARTICLES THROUGH EACH STAGE (PRISMA flow)", f"  Records identified            {flow['identified']}"]
    if flow.get("failed"):
        lines.append(f"  Failed (kept, need a person) {flow['failed']}")
    if "title" in flow:
        t = flow["title"]
        lines.append(f"  1. Title screening           screened {t['screened']:5}   excluded {t['excluded']:5}   -> {t['passed']} passed{lost(t)}")
        lines += reasons(t)
    a = flow["abstract"]
    n = 2 if "title" in flow else 1
    human = f"   ({a['decided_by_human']} decided by a person)" if a["decided_by_human"] else ""
    lines.append(f"  {n}. Abstract screening        screened {a['screened']:5}   excluded {a['excluded']:5}   -> {a['passed']} passed{human}{lost(a)}")
    lines += reasons(a)
    if "fulltext" in flow:
        f = flow["fulltext"]
        left_out = " (left out)" if f.get("not_retrieved_left_out") else ""
        lines.append(f"  {n + 1}. Full text                 sought   {f['sought']:5}   retrieved {f['retrieved']} {f['by_source']}   not retrieved {f['not_retrieved']}{left_out}")
        lines.append(f"                                assessed {f['assessed_by_agents']:5}   excluded {f['excluded']:5}   -> {f['passed']} passed{lost(f, final_only=True)}")
        lines += reasons(f)
        so = f["still_open"]
        if so["no_fulltext"] or so["unclear"]:
            lines.append(f"        still open (kept, a person decides): {so['no_fulltext']} without full text, {so['unclear']} unclear from the paper")
    i = flow["included"]
    extra = f"   (confirmed {i['confirmed']}, waiting for a person {i['waiting_for_human'] + i['failed_kept']})" if i["waiting_for_human"] or i["failed_kept"] else ""
    lines.append(f"  INCLUDED                      {i['total']}{extra}")
    if "gold_final_includes_total" in i:
        lines.append(f"        gold-standard final includes kept: {i['gold_final_includes_kept']} of {i['gold_final_includes_total']}")
    return "\n".join(lines)


def flow_figure(flow: dict, path: str | Path, title: str = "") -> Path:
    """A PRISMA-style box diagram of the flow."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [("Records identified", flow["identified"], None)]
    if "title" in flow:
        t = flow["title"]
        stages.append((f"Titles screened\n{t['screened']}", t["passed"], ("Excluded at title", t["excluded"], t["reasons"])))
    a = flow["abstract"]
    stages.append((f"Abstracts screened\n{a['screened']}", a["passed"], ("Excluded at abstract", a["excluded"], a["reasons"])))
    if "fulltext" in flow:
        f = flow["fulltext"]
        left_out = " (left out)" if f.get("not_retrieved_left_out") else ""
        label = (f"Full texts sought {f['sought']}\nretrieved {f['retrieved']}, not retrieved {f['not_retrieved']}{left_out}"
                 f"\nassessed by agents {f['assessed_by_agents']}")
        stages.append((label, f["passed"], ("Excluded at full text", f["excluded"], f["reasons"])))

    fig, ax = plt.subplots(figsize=(9, 1.9 * (len(stages) + 1)))
    ax.set_axis_off()
    y = len(stages) + 0.5
    box = dict(boxstyle="round,pad=0.5", fc="#eef3fb", ec="#3a5a8c")
    side = dict(boxstyle="round,pad=0.5", fc="#fbefee", ec="#8c3a3a")
    for i, (label, passed, excl) in enumerate(stages):
        text = label if i else f"{label}\n{passed}"
        ax.text(0.3, y, text, ha="center", va="center", fontsize=10, bbox=box)
        if excl:
            name, n, reasons = excl
            top = "\n".join(f"{r[:38]}: {k}" for r, k in list(reasons.items())[:4])
            ax.text(0.78, y, f"{name}: {n}" + (f"\n{top}" if top else ""), ha="center", va="center", fontsize=8.5, bbox=side)
            ax.annotate("", xy=(0.6, y), xytext=(0.45, y), arrowprops=dict(arrowstyle="->", color="#555"))
        ax.annotate("", xy=(0.3, y - 0.62), xytext=(0.3, y - 0.38), arrowprops=dict(arrowstyle="->", color="#555"))
        y -= 1
    inc = flow["included"]
    wait = inc["waiting_for_human"] + inc["failed_kept"]
    ax.text(0.3, y, f"Included\n{inc['total']}" + (f"\n({wait} still need a person)" if wait else ""),
            ha="center", va="center", fontsize=11, fontweight="bold", bbox=dict(boxstyle="round,pad=0.6", fc="#e9f6ec", ec="#2f7a43"))
    ax.set_xlim(0, 1)
    ax.set_ylim(y - 0.7, len(stages) + 1.1)
    if title:
        ax.set_title(title, fontsize=11)
    path = Path(path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adrd_review.eval.flow")
    ap.add_argument("run_dir")
    args = ap.parse_args(argv)
    report = json.loads((Path(args.run_dir) / "report.json").read_text())
    if "flow" not in report:
        print("This run has no flow counts (made before they were added); re-run it to get them.", file=sys.stderr)
        return 1
    print(format_flow(report["flow"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
