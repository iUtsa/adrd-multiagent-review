"""Build the package for the professor from a run's final results (python -m adrd_review.final_results first).

    python scripts/make_report.py data/runs/PROF_search_free/<timestamp> [--out data/deliverables/<date>]

Writes (git-ignored, under data/): report.html (self-contained), report.pdf (via headless Chrome), the final CSVs,
the blind random-check workbook + answer key, the charts and the diagrams, and README.txt listing everything.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def img(path: Path, alt: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{data}"><figcaption>{html.escape(alt)}</figcaption></figure>'


def table(rows: list[list], head: list[str]) -> str:
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--usd", type=float, required=True, help="total paid model cost to report")
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    fin = run / "final"
    s = json.loads((fin / "summary.json").read_text())
    out = Path(args.out or f"data/deliverables/{date.today().isoformat()}_ADRD_screening_results")
    (out / "charts").mkdir(parents=True, exist_ok=True)
    (out / "diagrams").mkdir(exist_ok=True)

    for name in ("stage1_titles.csv", "stage2_abstracts.csv", "stage3_fulltext.csv", "decisions.csv", "for_professor.csv", "not_assessed.csv", "summary.json", "professor_check.csv",
                 "professor_check_answer_key.csv", "professor_check.xlsx", "final_stages.png", "final_stages.csv",
                 "final_reasons.png", "final_reasons.csv"):
        dest = out / ("charts" if name.startswith("final_") else ".") / name
        shutil.copy(fin / name, dest)
    diag = ROOT / "docs/diagrams"
    for name in ("system-overview", "consensus-by-stage", "pipeline-as-run", "model-routing", "agent-interaction-sequence",
                 "run-loop", "timeline", "langgraph-full"):
        shutil.copy(diag / f"{name}.png", out / "diagrams" / f"{name}.png")
    for name in ("panel_agreement", "reviewer_votes", "how_decided", "model_calls", "progress"):
        shutil.copy(diag / "results" / f"{name}.png", out / "charts" / f"{name}.png")

    t, a, f, side = s["title"], s["abstract"], s["fulltext_free"], s["side_note"]
    reasons = s["exclusion_reasons"]
    rules = sorted({r for d in reasons.values() for r in d}, key=lambda r: (r[0] != "I", r))
    prof = s["for_the_professor_reasons"]
    age_prof = sum(v for k, v in prof.items() if "age" in k)
    usd = args.usd
    sl = list(csv.DictReader((run / "second_look.csv").open())) if (run / "second_look.csv").exists() else []
    slc = Counter(r["second look"] for r in sl)
    sl_inc, sl_exc, sl_total = slc.get("included", 0), slc.get("excluded", 0), len(sl)
    sl_settled = sl_inc + sl_exc
    tl = list(csv.DictReader((run / "third_look.csv").open())) if (run / "third_look.csv").exists() else []
    tlc = Counter(r["third look"] for r in tl)
    third_li = (f"<li><b>Third look:</b> the papers still unsettled were re-read by one more independent model "
                f"(DeepSeek V4.1 Flash, a family used nowhere else); it settled {tlc.get('included', 0) + tlc.get('excluded', 0)} "
                f"of {len(tl)} (included {tlc.get('included', 0)}, excluded {tlc.get('excluded', 0)}); "
                f"{sum('failed' in r['why'] for r in tl)} could not be re-read because the paid credit ran out.</li>") if tl else ""

    stage_rows = [["1. Titles", f"{t['screened']:,}", f"{t['excluded']:,}", f"{t['passed']:,}", "—"],
                  ["2. Abstracts", f"{a['screened']:,}", f"{a['excluded']:,}", f"{a['passed']:,}", "—"],
                  ["3. Full text (free)", f"{f['assessed']:,}", f"{f['excluded']:,}", f"<b>{f['included']:,}</b>",
                   f"{f['for the professor']:,}" + (f" (+{f['not finished']} not finished)" if f["not finished"] else "")]]
    reason_rows = [[html.escape(r), reasons["title"].get(r, 0), reasons["abstract"].get(r, 0), reasons["full text"].get(r, 0)]
                   for r in rules] + [["<b>Total</b>", f"<b>{t['excluded']}</b>", f"<b>{a['excluded']}</b>", f"<b>{f['excluded']}</b>"]]
    prof_rows = [[html.escape(k), v] for k, v in prof.items()]

    css = """
    :root{--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6;--bg:#fcfcfb}
    body{font-family:-apple-system,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);max-width:960px;margin:32px auto;padding:0 16px;line-height:1.5;font-size:14px}
    h1{font-size:24px;margin-bottom:4px} h2{font-size:18px;margin-top:32px;border-bottom:1px solid var(--line);padding-bottom:4px} h3{font-size:15px}
    .sub{color:var(--ink2)} table{border-collapse:collapse;margin:12px 0;width:100%} th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
    th{background:#f2f1ee} td:not(:first-child){font-variant-numeric:tabular-nums}
    figure{margin:16px 0} img{max-width:100%;border:1px solid var(--line)} figcaption{color:var(--ink2);font-size:12px}
    .box{border-left:4px solid var(--accent);background:#eef4fc;padding:10px 14px;margin:12px 0}
    .warn{border-left-color:#eb6834;background:#fdf0ea} ul{padding-left:20px} code{font-size:12px}
    @media print{body{margin:0;max-width:none} h2{break-after:avoid} figure,table{break-inside:avoid}}
    """
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADRD AI/ML Screening Results</title><style>{css}</style></head><body>
<h1>AI-assisted screening of the ADRD AI/ML search: results</h1>
<p class="sub">Multi-agent review of the PubMed search ({t['screened']:,} records) · run completed {date.today().strftime('%B %d, %Y')} · prepared by Arnab Das Utsa</p>

<div class="box"><b>In short.</b> Three independent AI reviewers screened all {t['screened']:,} records in three stages,
as the human team does (titles → abstracts → full text), using your criteria. <b>{f['included']:,} papers are included</b>;
<b>{f['for the professor']:,}</b> need your decision, almost all because the paper does not state the participants' age.
Only papers with a free full text were assessed at the full-text stage.</div>

<h2>1. Papers at each stage</h2>
{table(stage_rows, ["Stage", "In", "Excluded", "Passed / included", "For your decision"])}
<p class="sub">Not counted above: {side['free_copy_to_save_by_hand']:,} papers that passed abstracts have a free copy that we
still have to save by hand (the sites block automated download), and {side['paid_access_not_assessed']:,} are available
only with a subscription, so their full text was not assessed.</p>
{img(fin / 'final_stages.png', 'Papers at each stage (older adults 65+, free full text only)')}

<h2>2. Why papers were excluded</h2>
<p>The main rule behind each exclusion (the rule cited by most of the reviewers who excluded it).</p>
{table(reason_rows, ["Rule", "Titles", "Abstracts", "Full text"])}
{img(fin / 'final_reasons.png', 'Main reason for each exclusion, by stage')}

<h2>3. Papers that need your decision ({f['for the professor']:,})</h2>
{table(prof_rows, ["Why the AI could not decide", "Papers"])}
<div class="box warn"><b>One decision settles {age_prof:,} of them:</b> if a paper does not report the participants'
age (for example it uses ADNI or NACC data without an age table), should it count as 65+? If yes, about
{f['included'] + age_prof:,} papers are included in total; if no, {f['included']:,}. The list, with the AI's reasoning,
is in <code>for_professor.csv</code>.</div>

<h2>4. How the AI decided</h2>
<ul>
<li><b>Three independent reviewers</b>, each a different AI model with a different perspective (clinical neurologist,
machine-learning researcher, research methodologist). They never see each other's answers.</li>
<li><b>Your criteria as rules:</b> I1 dementia or Alzheimer's (MCI only together with it), I2 AI or machine learning
(logistic regression counts), I3 detection, diagnosis, monitoring or prediction, I4 older adults <b>65 or over</b>;
excluded: E1 reviews, E2 abstract only (full proceedings papers count), E3 no original results, E4 not human
participants (E3 and E4 are our proposals: please confirm).</li>
<li><b>Every exclusion needs evidence:</b> a reviewer can only mark a rule as failed with an exact quote from the text,
and the program checks every quote. Without a quote the rule counts as unclear, and unclear never excludes.</li>
<li><b>Titles:</b> excluded only if all three reviewers exclude. <b>Abstracts:</b> majority of three (a 2-1 exclusion is
flagged). <b>Full text:</b> included only when all four inclusion rules are clearly met; otherwise excluded (with a quote)
or left for you.</li>
<li><b>Second look:</b> every paper the panel could not settle was re-read by a fourth, independent model, looking for the
missing evidence; it settled {sl_settled} of {sl_total} (included {sl_inc}, excluded {sl_exc}).</li>
{third_li}
<li><b>Age (65+):</b> applied at the abstract stage (excluded only when the abstract clearly shows participants under 65)
and at full text (a dedicated check reads each paper's age criterion, mean or median age and range, with quotes).</li>
</ul>
{img(ROOT / 'docs/diagrams/system-overview.png', 'How the system works')}
{img(ROOT / 'docs/diagrams/consensus-by-stage.png', 'How decisions are made at each stage')}

<h2>5. Please note</h2>
<ul>
<li><b>Accuracy is not measured yet.</b> There is no human gold standard for this search. A small earlier test on a
different dataset found the agents missed about 1 in 4 relevant papers. The random check below gives a first estimate
on this search.</li>
<li><b>Free full text only:</b> {side['paid_access_not_assessed']:,} paid-access papers were not assessed;
{side['free_copy_to_save_by_hand']:,} free ones are still to be saved by hand and will be added.</li>
<li><b>Age, strict reading:</b> a paper counts as 65+ when it requires 65+, its youngest participant is 65+, or the
mean/median age of the sample is 65+ (the mean/median part is our proposal). When a paper gives ages only per group,
the youngest group's mean was used, so a study whose younger comparison group averages under 65 is excluded. Please
tell us if you prefer a looser reading.</li>
<li>{f['not finished']} papers are not finished: one AI reviewer's service rejects them; they will be decided by hand.</li>
<li>Some free copies are preprints (arXiv, bioRxiv, medRxiv), whose text can differ slightly from the published version.</li>
<li>The models changed during the run because free services withdrew or limited them; each change and its effect is
documented. All decisions, quotes and reasoning are saved for every paper.</li>
<li>Cost: about ${usd:.2f} of paid model calls; everything else ran on free tiers (about 10 days, unattended).</li>
</ul>

<h2>6. Random check (optional)</h2>
<p><code>professor_check.xlsx</code> has 40 papers drawn at random: 10 included and 10 excluded at each stage, in random
order. It shows only the paper (title, abstract, PubMed link) and a column for your decision; the AI's decision is
hidden (and in <code>professor_check_answer_key.csv</code>). Comparing your decisions with the AI's gives a first
estimate of how often it is right.</p>

<h2>7. Files</h2>
{table([[ '<code>stage1_titles.csv</code>', f"All {t['screened']:,} records: title decision, and the reason and quotes for each exclusion"],
        ['<code>stage2_abstracts.csv</code>', f"The {a['screened']:,} papers that passed titles, with the abstract: decision, reason, quotes"],
        ['<code>stage3_fulltext.csv</code>', f"The {a['passed']:,} papers that passed abstracts: included / excluded / for your decision / not assessed, with reason and quotes"],
        ['<code>decisions.csv</code>', 'Every paper: final stage, outcome, reason, who decided, evidence (quotes)'],
        ['<code>for_professor.csv</code>', f"The {f['for the professor']:,} papers that need your decision, with the reason"],
        ['<code>not_assessed.csv</code>', 'Papers without an assessable free full text (to save by hand, or paid access)'],
        ['<code>professor_check.xlsx</code> / <code>.csv</code>', 'The blind random check (40 papers)'],
        ['<code>professor_check_answer_key.csv</code>', "The AI's decisions for the random check (open after checking)"],
        ['<code>charts/</code>, <code>diagrams/</code>', 'All charts and diagrams in this report, and more']], ["File", "Contents"])}
</body></html>"""
    (out / "report.html").write_text(doc)
    try:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={(out / 'report.pdf').resolve()}", (out / "report.html").resolve().as_uri()],
                       check=True, capture_output=True, timeout=120)
    except Exception as e:  # the HTML is the report; the PDF is a convenience
        print(f"(PDF not made: {type(e).__name__})", file=sys.stderr)
    (out / "README.txt").write_text(
        "ADRD AI/ML screening results\n\nreport.pdf / report.html  the report\n"
        "stage1_titles.csv         all records: title decision, reason, quotes\n"
        "stage2_abstracts.csv      papers that passed titles: abstract decision, reason, quotes, abstract\n"
        "stage3_fulltext.csv       papers that passed abstracts: full-text outcome, reason, quotes\n"
        "decisions.csv             every paper's final outcome, reason and evidence\n"
        "for_professor.csv         papers that need the professor's decision\n"
        "not_assessed.csv          papers without an assessable free full text\n"
        "professor_check.xlsx      blind random check (40 papers); AI answers hidden\n"
        "professor_check.csv       the same check as CSV (no AI answers)\n"
        "professor_check_answer_key.csv  the AI's answers for the check\n"
        "summary.json              all counts\ncharts/  diagrams/       figures\n")
    print(f"Package: {out}")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out)}  ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
