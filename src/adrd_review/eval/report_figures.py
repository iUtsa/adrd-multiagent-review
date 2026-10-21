"""Result figures for a finished run (free full text only), each saved as PNG plus a CSV of the plotted numbers.

    stage_outcomes      papers per stage: passed / excluded / for the professor / retry
    exclusion_reasons   main rule behind each exclusion, per stage
    panel_agreement     Fleiss' kappa and share of unanimous votes among reviewers A, B, C, per stage
    how_decided         unanimous / majority / no majority, per stage
    reviewer_votes      how often each reviewer voted to exclude (titles, abstracts) or 'unclear' (full text)
    model_calls         answers per model and host, per stage (which model actually answered)
    openrouter_cost     recorded cost of the paid OpenRouter calls, per model
    progress            papers still open after each try of the unattended loop (from the loop's log)
    second_look         outcome of the second look at the papers left for a person (if second_look.csv exists)

Paid-access papers (no free full text) are not part of any chart; their number is printed as a side note.
Colours: the validated reference palette (blue = passed/included, orange = excluded, aqua = for the professor,
grey = retry/open); every segment carries a direct label, and every chart has its numbers in a CSV.

    python -m adrd_review.eval.report_figures data/runs/<name>/<timestamp> --out docs/diagrams/results
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from adrd_review.codebook import load_codebook
from adrd_review.eval.stages import collect, load_states

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
RAMP = ["#184f95", "#3987e5", "#86b6ef"]  # ordinal blue steps (600, 400, 250): strongest agreement darkest
STAGES = [("title", "Titles"), ("abstract", "Abstracts"), ("fulltext", "Full text (free)")]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })
    return plt


def _save(fig, out: Path, name: str, rows: list[dict]) -> Path:
    fig.savefig(out / f"{name}.png", dpi=170, bbox_inches="tight")
    if rows:
        with (out / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out / f"{name}.png"


def _hbar_labels(ax, bars, values, inside_min: float, fmt="{:,}"):
    """Count at the end of each bar (inside when the bar is wide enough, else just outside)."""
    for b, v in zip(bars, values):
        if not v:
            continue
        x = b.get_x() + b.get_width()
        inside = b.get_width() >= inside_min
        ax.text(x - (b.get_width() / 2 if inside else -inside_min * 0.08), b.get_y() + b.get_height() / 2,
                fmt.format(v), va="center", ha="center" if inside else "left", fontsize=9,
                color="white" if inside else INK)


# --------------------------------------------------------------------------- charts


def stage_outcomes(rows: dict, out: Path):
    plt = _plt()
    groups = {
        "Titles": [("Passed", sum(r["ai_outcome"] == "passed" for r in rows["title"]), BLUE),
                   ("Excluded", sum(r["ai_outcome"] == "excluded" for r in rows["title"]), ORANGE)],
        "Abstracts": [("Passed", sum(r["ai_outcome"] == "passed" for r in rows["abstract"]), BLUE),
                      ("Excluded", sum(r["ai_outcome"] == "excluded" for r in rows["abstract"]), ORANGE),
                      ("Still open", sum(r["ai_outcome"] == "still open" for r in rows["abstract"]), GREY)],
        "Full text (free)": [("Included", sum(r["ai_outcome"] == "passed" for r in rows["fulltext"]), BLUE),
                             ("Excluded", sum(r["ai_outcome"] == "excluded" for r in rows["fulltext"]), ORANGE),
                             ("For the professor", sum(r["ai_outcome"] == "waiting for a person" for r in rows["fulltext"]), AQUA),
                             ("Retry", sum(r["ai_outcome"] == "still open" for r in rows["fulltext"]), GREY)],
    }
    fig, ax = plt.subplots(figsize=(9, 3.2))
    names = list(groups)[::-1]
    widest = max(sum(n for _, n, _ in groups[g]) for g in names)
    for y, g in enumerate(names):
        left = 0
        for label, n, color in groups[g]:
            if not n:
                continue
            ax.barh(y, n, left=left, color=color, height=0.55, edgecolor=SURFACE, linewidth=2)
            if n >= widest * 0.035:
                ax.text(left + n / 2, y, f"{n:,}", ha="center", va="center", fontsize=9,
                        color="white" if color in (BLUE, ORANGE) else INK)
            left += n
        small = [f"{n:,} {label.lower()}" for label, n, _ in groups[g] if n and n < widest * 0.035]
        ax.text(left + widest * 0.01, y, f"{left:,} papers" + (f"  ({', '.join(small)})" if small else ""),
                va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(0, widest * 1.3)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, AQUA, GREY)]
    ax.legend(handles, ["Passed / included", "Excluded", "For the professor", "Still open / retry"],
              ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=9)
    ax.set_title("Papers at each stage (free full text only)")
    data = [{"stage": g, "outcome": label, "papers": n} for g in groups for label, n, _ in groups[g]]
    return _save(fig, out, "stage_outcomes", data)


def exclusion_reasons(rows: dict, codebook, out: Path):
    plt = _plt()
    order = [f"{r.id} {r.name}" for r in codebook.rules]
    counts = {k: Counter(r["main reason"] for r in rows[k] if r["ai_outcome"] == "excluded") for k, _ in STAGES}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    top = max(max(c.values(), default=0) for c in counts.values()) or 1
    data = []
    for ax, (key, title) in zip(axes, STAGES):
        vals = [counts[key].get(r, 0) for r in order]
        bars = ax.barh(range(len(order)), vals, color=ORANGE, height=0.6)
        for b, v in zip(bars, vals):
            if v:
                ax.text(b.get_width() + top * 0.02, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=9)
        ax.set_xlim(0, top * 1.25)
        ax.set_title(f"{title}: {sum(vals):,} excluded")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        data += [{"stage": title, "rule": r, "excluded": v} for r, v in zip(order, vals)]
    axes[0].set_yticks(range(len(order)), [o if len(o) < 34 else o[:32] + "…" for o in order])
    axes[0].invert_yaxis()
    fig.suptitle("Main reason for each exclusion (rule cited by most excluding reviewers)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11, y=1.03)
    return _save(fig, out, "exclusion_reasons", data)


def panel_agreement(report: dict, out: Path):
    plt = _plt()
    pa = report.get("panel_agreement") or {}
    keys = [("title", "Titles"), ("screening", "Abstracts"), ("eligibility", "Full text (free)")]
    kap = [pa.get(k, {}).get("fleiss_kappa") or 0 for k, _ in keys]
    una = [100 * pa[k]["unanimous"] / pa[k]["items"] if pa.get(k, {}).get("items") else 0 for k, _ in keys]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3))
    for ax, vals, title, fmt, lim in ((a1, kap, "Fleiss' kappa among reviewers A, B, C", "{:.2f}", 1.0),
                                      (a2, una, "Papers where all three agreed (%)", "{:.0f}%", 100)):
        bars = ax.bar([t for _, t in keys], vals, color=BLUE, width=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v - lim * 0.06, fmt.format(v), ha="center", fontsize=9,
                    color="white", fontweight="bold")
        ax.set_ylim(0, lim)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title)
    a1.axhline(0.6, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    a1.text(-0.35, 0.62, "0.6 = substantial agreement starts", fontsize=8, color=INK2, ha="left")
    data = [{"stage": t, "fleiss_kappa": round(k, 4), "unanimous_pct": round(u, 1),
             "items": pa.get(key, {}).get("items")} for (key, t), k, u in zip(keys, kap, una)]
    return _save(fig, out, "panel_agreement", data)


def how_decided(states: list[dict], out: Path):
    plt = _plt()
    methods = {"Titles": Counter(), "Abstracts": Counter(), "Full text (free)": Counter()}
    for s in states:
        t = (s.get("title_decision") or {}).get("method")
        if t:
            methods["Titles"][{"unanimous": "Unanimous", "any_include": "Kept by 1-2 of 3"}.get(t, t)] += 1
        sd = s.get("screening_decision") or {}
        if s.get("screening_votes") and sd.get("method") in ("unanimous", "majority", "tie_include"):
            methods["Abstracts"]["Unanimous" if sd["method"] == "unanimous" else "Majority (2 of 3)"] += 1
        ed = s.get("eligibility_decision") or {}
        if s.get("eligibility_votes") and ed.get("method") in ("unanimous", "majority", "no_majority"):
            methods["Full text (free)"][{"unanimous": "Unanimous", "majority": "Majority (2 of 3)",
                                         "no_majority": "No majority"}[ed["method"]]] += 1
    cats = ["Unanimous", "Majority (2 of 3)", "Kept by 1-2 of 3", "No majority"]
    colors = dict(zip(cats, [RAMP[0], RAMP[1], RAMP[1], RAMP[2]]))
    fig, ax = plt.subplots(figsize=(9, 3))
    names = list(methods)[::-1]
    data = []
    for y, g in enumerate(names):
        total = sum(methods[g].values()) or 1
        left = 0
        for c in cats:
            n = methods[g].get(c, 0)
            if not n:
                continue
            pct = 100 * n / total
            ax.barh(y, pct, left=left, color=colors[c], height=0.55, edgecolor=SURFACE, linewidth=2)
            if pct >= 6:
                ax.text(left + pct / 2, y, f"{c}\n{n:,} ({pct:.0f}%)", ha="center", va="center", fontsize=8,
                        color="white" if c == "Unanimous" else INK)
            else:
                ax.annotate(f"{c}: {n:,} ({pct:.0f}%)", (left + pct / 2, y + 0.28), xytext=(-6, 10),
                            textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK2,
                            arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.8})
            left += pct
            data.append({"stage": g, "how": c, "papers": n, "pct": round(pct, 1)})
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of papers decided at that stage")
    ax.tick_params(axis="y", length=0)
    ax.set_title("How the panel reached its decision")
    return _save(fig, out, "how_decided", data)


def reviewer_votes(states: list[dict], out: Path):
    plt = _plt()
    spec = [("title_votes", "Titles: voted exclude", "exclude"), ("screening_votes", "Abstracts: voted exclude", "exclude"),
            ("eligibility_votes", "Full text: voted unclear", "unclear")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    data = []
    for ax, (key, title, label) in zip(axes, spec):
        n, hit = Counter(), Counter()
        for s in states:
            for v in s.get(key) or []:
                n[v["reviewer_id"]] += 1
                hit[v["reviewer_id"]] += v["labels"] == [label]
        ids = sorted(n)
        pct = [100 * hit[i] / n[i] for i in ids]
        bars = ax.bar(ids, pct, color=BLUE, width=0.5)
        for b, p, i in zip(bars, pct, ids):
            ax.text(b.get_x() + b.get_width() / 2, p + 2, f"{p:.0f}%\n({hit[i]:,})", ha="center", fontsize=8)
            data.append({"stage": title, "reviewer": i, "votes": n[i], "count": hit[i], "pct": round(p, 1)})
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("% of that reviewer's votes")
    fig.suptitle("How each reviewer voted (A: neurologist, B: ML researcher, C: methodologist)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11, y=1.05)
    return _save(fig, out, "reviewer_votes", data)


def model_calls(usage: list[dict], out: Path):
    plt = _plt()
    role_name = {"title": "Titles", "screening": "Abstracts", "eligibility": "Full text"}
    counts = Counter((role_name.get(u["role"], u["role"]), u["model"]) for u in usage if u["role"] in role_name)
    items = sorted(counts.items(), key=lambda kv: (list(role_name.values()).index(kv[0][0]), -kv[1]))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(items) + 1))
    labels = [f"{st} · {m.split(':', 1)[0]} · {m.split(':', 1)[1].split('/')[-1]}" for (st, m), _ in items]
    vals = [v for _, v in items]
    paid = [not (m.split(":")[0] in ("groq", "nvidia") or m.endswith(":free")) for (_, m), _ in items]
    bars = ax.barh(range(len(items)), vals, color=BLUE, height=0.6)
    for b, is_paid in zip(bars, paid):  # texture, not a second hue: orange means "excluded" in the other charts
        if is_paid:
            b.set_facecolor(SURFACE)
            b.set_edgecolor(BLUE)
            b.set_hatch("////")
            b.set_linewidth(1.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + max(vals) * 0.01, b.get_y() + b.get_height() / 2, f"{v:,}", va="center", fontsize=8)
    ax.set_yticks(range(len(items)), labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.12)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("papers judged (one reviewer's answer for one paper; batch calls counted per paper)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, facecolor=SURFACE, edgecolor=BLUE, hatch="////")],
              ["free tier", "paid (OpenRouter)"],
              frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Which model and host answered (stage · host · model)")
    data = [{"stage": st, "model": m, "answers": v} for (st, m), v in items]
    return _save(fig, out, "model_calls", data)


def openrouter_cost(usage: list[dict], extra: dict[str, float], out: Path, billed: float | None = None):
    plt = _plt()
    cost = defaultdict(float)
    for u in usage:
        if u.get("cost_usd"):
            cost[u["model"]] += u["cost_usd"]
    for k, v in extra.items():
        cost[k] += v
    items = sorted(cost.items(), key=lambda kv: -kv[1])
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(items) + 1))
    bars = ax.barh(range(len(items)), [v for _, v in items], color=BLUE, height=0.55)
    top = max(v for _, v in items)
    for b, (_, v) in zip(bars, items):
        ax.text(b.get_width() + top * 0.01, b.get_y() + b.get_height() / 2, f"${v:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(len(items)), [k.split(":", 1)[1] for k, _ in items])
    ax.invert_yaxis()
    ax.set_xlim(0, top * 1.2)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"Cost of paid calls as recorded per call: ${sum(v for _, v in items):.2f} (all other calls free)")
    if billed is not None:
        ax.text(0, -0.22, f"OpenRouter billed ${billed:.2f} for the run: providers also bill reasoning tokens that are not "
                          "reported per call.", transform=ax.transAxes, fontsize=8.5, color=INK2)
    data = [{"model": k, "usd": round(v, 4)} for k, v in items]
    return _save(fig, out, "openrouter_cost", data)


def progress(log: Path, out: Path):
    """Papers still open after each try, read from run_until_done's log (tries whose count is known)."""
    plt = _plt()
    import matplotlib.dates as mdates

    points, start = [], None
    for line in log.read_text().splitlines():
        m = re.match(r"=== try \d+: (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}) ===", line)
        if m:
            start = m.group(1)
            continue
        m = re.match(r"=== try \d+ done (\d{2}:\d{2}): (\d+) article", line)
        if m and start:
            t = datetime.strptime(f"{start} {m.group(1)}", "%Y-%m-%d %H:%M")
            if points and t < points[-1][0]:  # finished after midnight
                t = t.replace(day=t.day + 1)
            points.append((t, int(m.group(2))))
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot([p[0] for p in points], [p[1] for p in points], color=BLUE, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    for t, n in (points[0], points[-1]):
        ax.annotate(f"{n:,}", (t, n), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylim(0, max(n for _, n in points) * 1.15)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("papers still open")
    ax.set_title("Unattended free-tier run: papers not yet finished after each try")
    data = [{"finished": t.isoformat(timespec="minutes"), "still_open": n} for t, n in points]
    return _save(fig, out, "progress", data)


def second_look(path: Path, out: Path):
    if not path.exists():
        return None
    plt = _plt()
    rows = list(csv.DictReader(path.open()))
    c = Counter(r["second look"] for r in rows)
    parts = [("Included", c.get("included", 0), BLUE), ("Excluded", c.get("excluded", 0), ORANGE),
             ("Still for the professor", c.get("for the professor", 0), AQUA)]
    fig, ax = plt.subplots(figsize=(9, 1.8))
    left = 0
    for label, n, color in parts:
        if n:
            ax.barh(0, n, left=left, color=color, height=0.5, edgecolor=SURFACE, linewidth=2)
            ax.text(left + n / 2, 0, f"{label}\n{n}", ha="center", va="center", fontsize=9,
                    color="white" if color in (BLUE, ORANGE) else INK)
            left += n
    ax.set_xlim(0, left)
    ax.set_yticks([])
    ax.set_title(f"Second look at the {len(rows)} papers the panel left for a person (GLM-5.3, same rules and quote check)")
    return _save(fig, out, "second_look", [{"outcome": l, "papers": n} for l, n, _ in parts])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adrd_review.eval.report_figures")
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="folder for the figures (default: <run_dir>/figures)")
    ap.add_argument("--log", default=None, help="run_until_done log (default: data/logs/until_done_<name>.log)")
    ap.add_argument("--billed", type=float, default=None, help="amount the provider billed, shown under the cost chart")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    codebook = load_codebook(meta["config"]["codebook"])
    rows = collect(run_dir, codebook)
    states = load_states(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    usage = [json.loads(line) for line in (run_dir / "usage.jsonl").read_text().splitlines() if line.strip()]
    log = Path(args.log) if args.log else Path("data/logs") / f"until_done_{meta['name']}.log"
    sl = run_dir / "second_look.csv"
    sl_cost = defaultdict(float)  # the second look's own calls are not in the run's usage.jsonl
    made = [stage_outcomes(rows, out), exclusion_reasons(rows, codebook, out), panel_agreement(report, out),
            how_decided(states, out), reviewer_votes(states, out), model_calls(usage, out),
            openrouter_cost(usage, sl_cost, out, args.billed), progress(log, out) if log.exists() else None, second_look(sl, out)]
    paid = sum(r["ai_outcome"] == "not retrieved" for r in rows["fulltext"])
    for p in made:
        if p:
            print(f"  {p}")
    print(f"Side note (not in any chart): {paid} paid-access paper(s) found, not assessed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
