"""Build the simple results package from a run's final results.

    python -m adrd_review.final_results <run>        # first: final decisions, stage files, random check
    python -m adrd_review.eval.accuracy <run>        # first: reliability (accuracy.json)
    python scripts/make_report.py <run> --usd 10.03 [--out data/deliverables/<name>]

Package (git-ignored, under data/deliverables/):
    1_Report.pdf                 the report (also extras/report.html)
    2_Stage1_Titles.csv          every record: title decision, reason, quotes
    3_Stage2_Abstracts.csv       papers that passed titles: abstract decision, reason, quotes, abstract
    4_Stage3_FullText.csv        papers that passed abstracts: included / excluded / for human review / not assessed
    5_For_Human_Review.csv      papers the AI could not settle, with the reason
    6_Random_Check.xlsx          blind random check (40 papers; AI answers hidden)
    send_later/Random_Check_Answer_Key.csv   the AI's answers (send after the check)
    extras/                      all decisions, counts, reliability, charts, diagrams, the HTML report
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


def kappa_label(k: float) -> str:  # Landis & Koch
    return ("poor" if k < 0 else "slight" if k <= 0.2 else "fair" if k <= 0.4 else "moderate" if k <= 0.6
            else "substantial" if k <= 0.8 else "almost perfect")


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
    acc = json.loads((fin / "accuracy.json").read_text())
    out = Path(args.out or f"data/deliverables/{date.today().isoformat()}_ADRD_screening_results")
    if out.exists():
        shutil.rmtree(out)
    for sub in ("extras/charts", "extras/diagrams", "send_later"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    copies = {"stage1_titles.csv": "2_Stage1_Titles.csv", "stage2_abstracts.csv": "3_Stage2_Abstracts.csv",
              "stage3_fulltext.csv": "4_Stage3_FullText.csv", "for_human_review.csv": "5_For_Human_Review.csv",
              "random_check.xlsx": "6_Random_Check.xlsx",
              "random_check_answer_key.csv": "send_later/Random_Check_Answer_Key.csv",
              "random_check.csv": "extras/random_check.csv", "decisions.csv": "extras/all_decisions.csv",
              "not_assessed.csv": "extras/not_assessed.csv", "summary.json": "extras/summary.json",
              "accuracy.json": "extras/reliability.json"}
    for src, dst in copies.items():
        shutil.copy(fin / src, out / dst)
    for name in ("final_stages", "final_reasons"):
        for ext in ("png", "csv"):
            shutil.copy(fin / f"{name}.{ext}", out / "extras/charts" / f"{name}.{ext}")
    for name in ("panel_agreement", "reviewer_votes", "model_calls", "progress"):
        shutil.copy(ROOT / "docs/diagrams/results" / f"{name}.png", out / "extras/charts" / f"{name}.png")
    for name in ("system-overview", "system-overview-slide", "architecture-slide", "consensus-by-stage", "pipeline-as-run", "model-routing", "agent-interaction-sequence",
                 "run-loop", "timeline"):
        shutil.copy(ROOT / "docs/diagrams" / f"{name}.png", out / "extras/diagrams" / f"{name}.png")

    t, a, f, side = s["title"], s["abstract"], s["fulltext_free"], s["side_note"]
    reasons = s["exclusion_reasons"]
    rules = sorted({r for d in reasons.values() for r in d}, key=lambda r: (r[0] != "I", r))
    prof = s["for_the_professor_reasons"]
    age_prof = sum(v for k, v in prof.items() if "age" in k)
    rel = acc["reliability"]
    cal = acc["calibration_human_gold"]
    g = rel["grounding"]

    def looks(name: str, col: str) -> Counter:
        p = run / name
        return Counter(r[col] for r in csv.DictReader(p.open())) if p.exists() else Counter()

    sl, tl = looks("second_look.csv", "second look"), looks("third_look.csv", "third look")
    waiting = f["for human review"] + f["not finished"]

    stage_rows = [["1. Titles", f"{t['screened']:,}", f"{t['excluded']:,}", f"{t['passed']:,}", "—"],
                  ["2. Abstracts", f"{a['screened']:,}", f"{a['excluded']:,}", f"{a['passed']:,}", "—"],
                  ["3. Full text (free)", f"{f['assessed']:,}", f"{f['excluded']:,}", f"<b>{f['included']:,}</b>",
                   f"{waiting:,}"]]
    reason_rows = [[html.escape(r), reasons["title"].get(r, 0), reasons["abstract"].get(r, 0),
                    reasons["full text"].get(r, 0)] for r in rules]
    reason_rows.append(["<b>Total</b>", f"<b>{t['excluded']}</b>", f"<b>{a['excluded']}</b>", f"<b>{f['excluded']}</b>"])
    prof_rows = [[html.escape(k[0].upper() + k[1:]), v] for k, v in prof.items()]
    if f["not finished"]:
        prof_rows.append(["One AI reviewer's service could not read the paper", f["not finished"]])
    rel_rows = [[name.capitalize(), f"{r['papers']:,}", f"{r['fleiss_kappa']:.2f} ({kappa_label(r['fleiss_kappa'])})",
                 f"{r['all_three_agree']:.0%}"] for name, r in rel.items() if name != "grounding"]

    css = """
    body{font-family:-apple-system,Helvetica,Arial,sans-serif;color:#0b0b0b;background:#fcfcfb;max-width:900px;margin:32px auto;padding:0 16px;line-height:1.5;font-size:14px}
    h1{font-size:23px;margin-bottom:2px} h2{font-size:17px;margin-top:30px;border-bottom:1px solid #e4e3df;padding-bottom:4px}
    .sub{color:#52514e} table{border-collapse:collapse;margin:10px 0;width:100%} th,td{border:1px solid #e4e3df;padding:6px 8px;text-align:left;vertical-align:top}
    th{background:#f2f1ee} figure{margin:14px 0} img{max-width:100%;border:1px solid #e4e3df} figcaption{color:#52514e;font-size:12px}
    .box{border-left:4px solid #2a78d6;background:#eef4fc;padding:10px 14px;margin:12px 0} .warn{border-left-color:#eb6834;background:#fdf0ea}
    ul{padding-left:20px} code{font-size:12px}
    @media print{body{margin:0;max-width:none} h2{break-after:avoid} figure,table,.box{break-inside:avoid}}
    """
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADRD AI/ML Screening Results</title><style>{css}</style></head><body>
<h1>AI-assisted screening of the ADRD AI/ML search</h1>
<p class="sub">{t['screened']:,} PubMed records · titles → abstracts → full text · {date.today().strftime('%B %d, %Y')} · Arnab Das Utsa</p>

<div class="box"><b>Result.</b> Of {t['screened']:,} records, <b>{f['included']:,} papers are included</b> and
<b>{waiting:,} need human review</b> ({age_prof:,} of them only because the paper does not state the participants'
age). Criteria: the review criteria, with older adults = 65 or over. Full texts: free copies only.</div>

<h2>1. Papers at each stage</h2>
{table(stage_rows, ["Stage", "Screened", "Excluded", "Passed / included", "Human review"])}
<p class="sub">Not counted: {side['free_copy_to_save_by_hand']:,} free full texts still to be downloaded by hand and
{side['paid_access_not_assessed']:,} subscription-only papers (not assessed).</p>
{img(fin / 'final_stages.png', 'Papers at each stage')}

<h2>2. Why papers were excluded</h2>
{table(reason_rows, ["Main reason", "Titles", "Abstracts", "Full text"])}

<h2>3. Papers that need human review</h2>
{table(prof_rows, ["Why the AI could not decide", "Papers"])}
<div class="box warn"><b>One question settles {age_prof:,} of them:</b> if a paper does not report age (e.g. ADNI or
NACC data without an age table), should it count as 65+? Yes: about {f['included'] + age_prof:,} included in total;
no: {f['included']:,}. List: <code>5_For_Human_Review.csv</code>.</div>

<h2>4. Reliability and accuracy</h2>
<p><b>Agreement among the three AI reviewers</b> (independent votes, before any reconciliation):</p>
{table(rel_rows, ["Stage", "Papers", "Fleiss' κ", "All three agree"])}
<ul>
<li><b>Evidence:</b> {g['verified_share']:.0%} of the quotes the reviewers gave were found word for word in the paper;
the other {1 - g['verified_share']:.0%} were discarded and the rule counted as unclear, so no decision rests on an
unverified quote.</li>
<li><b>Accuracy against human decisions</b> could so far only be measured on a different dataset (Cochrane dementia
screening, 40 papers): recall {cal['recall']:.2f}, precision {cal['precision']:.2f}, specificity
{cal['specificity']:.2f}, Cohen's κ {cal['cohen_kappa']:.2f} (a keyword search: precision
{cal['keyword_baseline']['precision']:.2f}, κ {cal['keyword_baseline']['cohen_kappa']:.2f}). Context only, not an
estimate for this search.</li>
<li><b>Accuracy on this search</b> needs a human check: <code>6_Random_Check.xlsx</code> has 40 random papers (10
included, 10 excluded at each stage) with the AI's answers hidden. Once you mark include or exclude, we compute
accuracy, sensitivity, specificity, precision and Cohen's κ with 95% confidence intervals.</li>
</ul>

<h2>5. How the AI decided</h2>
<ul>
<li>Three independent AI reviewers (different models; perspectives: neurologist, machine-learning researcher,
methodologist) judged every rule: I1 dementia/Alzheimer's (MCI only with it), I2 AI/ML (logistic regression counts),
I3 detection/diagnosis/monitoring/prediction, I4 older adults 65+; excluded: E1 reviews, E2 abstract only, E3 no
original results, E4 not human (E3 and E4 are our proposals).</li>
<li>A rule can only be marked as failed with an exact quote, checked by the program. Titles: excluded only if all three
exclude. Abstracts: majority. Full text: included only when all four rules are clearly met.</li>
<li>Papers the panel could not settle were re-read by a second model (settled {sl.get('included', 0) + sl.get('excluded', 0)}),
age was then checked at 65+ in abstracts and full texts, and a third model re-read what was still open
(settled {tl.get('included', 0) + tl.get('excluded', 0)}).</li>
</ul>
{img(ROOT / 'docs/diagrams/system-overview.png', 'How the system works')}

<h2>6. Notes</h2>
<ul>
<li>Age reading is strict: when ages are given only per group, the youngest group's mean was used.</li>
<li>Some free copies are preprints. Model changes during the run are documented. Paid model calls cost
${args.usd:.2f}; everything else ran on free services.</li>
<li>Code, documentation and all diagrams:
<a href="https://github.com/iUtsa/adrd-multiagent-review">github.com/iUtsa/adrd-multiagent-review</a>.</li>
</ul>

<h2>7. Files</h2>
{table([['<code>1_Report.pdf</code>', 'This report'],
        ['<code>2_Stage1_Titles.csv</code>', f"All {t['screened']:,} records: title decision, reason, quotes"],
        ['<code>3_Stage2_Abstracts.csv</code>', f"The {a['screened']:,} that passed titles: abstract, decision, reason, quotes"],
        ['<code>4_Stage3_FullText.csv</code>', f"The {a['passed']:,} that passed abstracts: final outcome, reason, quotes"],
        ['<code>5_For_Human_Review.csv</code>', 'Papers the AI could not settle, with the reason'],
        ['<code>6_Random_Check.xlsx</code>', 'The blind random check (40 papers)']], ["File", "Contents"])}
</body></html>"""
    (out / "extras/report.html").write_text(doc)
    try:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={(out / '1_Report.pdf').resolve()}", (out / "extras/report.html").resolve().as_uri()],
                       check=True, capture_output=True, timeout=120)
    except Exception as e:  # the HTML is the report; the PDF is a convenience
        print(f"(PDF not made: {type(e).__name__})", file=sys.stderr)
    (out / "README.txt").write_text(
        "ADRD AI/ML screening results: open the files in order.\n\n"
        "1_Report.pdf              the report\n"
        "2_Stage1_Titles.csv       every record: title decision, reason, quotes\n"
        "3_Stage2_Abstracts.csv    papers that passed titles: abstract decision, reason, quotes, abstract\n"
        "4_Stage3_FullText.csv     papers that passed abstracts: final outcome, reason, quotes\n"
        "5_For_Human_Review.csv   papers the AI could not settle, with the reason\n"
        "6_Random_Check.xlsx       blind random check: mark include/exclude; the AI's answers are hidden\n"
        "send_later/               the AI's answers for the random check (send after it is done)\n"
        "extras/                   all decisions, counts, reliability, charts, diagrams, HTML report\n")
    print(f"Package: {out}")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out)}  ({max(1, p.stat().st_size // 1024)} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
