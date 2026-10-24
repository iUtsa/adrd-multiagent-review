"""Phase 2: independent reviewers. Each AI reviewer reviews every paper on its own, end to end, like a human
reviewer, and only their final verdicts are combined, by majority vote.

The professor's design for phase 2 (2026-10-24): not one model per narrow task with the votes combined at every
stage (phase 1), but individual, independent reviewers who each review everything, then majority vote wins.

For each reviewer (one model for the whole review, no specialist persona, never sees the others):
    1. Titles      every record (batches of 50): exclude only if a rule clearly fails, with a quote from the title
    2. Abstracts   only the papers THIS reviewer kept (batches of 4): the same per-rule form and quote check
    3. Full text   only the papers THIS reviewer kept, free full text only: include only when every inclusion
                   rule is clearly met, exclude with a quote, otherwise unclear
    -> its verdict: excluded at <stage> / included / unclear / not assessed (no free full text)
The evidence standard is phase 1's (agents/batch.py, agents/eligibility.py): quotes are checked against the text and a
rule marked failed without a real quote counts as unclear. The codebook is v0.4.0 (older adults 65+ in the rules).

Majority vote (3 reviewers): excluded when 2 have excluded, at the stage where the second exclusion happened;
included when 2 include; not assessed when the free full text is missing for the papers that reached it; otherwise
human review.

    python -m adrd_review.phase2 configs/runs/PHASE2_independent.yaml [--limit N]
    TRIES=240 RUNNER=adrd_review.phase2 scripts/run_until_done.sh configs/runs/PHASE2_independent.yaml
Writes data/runs/<config name>/<timestamp>/: reviewer_<id>.csv, final.csv, summary.json, failures.json, run_meta.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel

from adrd_review.agents import batch, eligibility
from adrd_review.codebook import Codebook, load_codebook
from adrd_review.eval.metrics import cohen_kappa, fleiss_kappa, gwet_ac1
from adrd_review.llm import LLM, ModelConfig, build_llm
from adrd_review.schemas import Article, Vote
from adrd_review.tools.pubmed import fetch_articles

STAGES = ("title", "abstract", "full text")
ROLE = ("an independent reviewer on a systematic review team. You review each paper yourself, applying every "
        "inclusion and exclusion criterion exactly as a careful human screener would; no one else's opinion is "
        "shown to you")


class ReviewerConfig(BaseModel):
    id: str
    model: ModelConfig


class Phase2Config(BaseModel):
    name: str
    description: str = ""
    codebook: str
    pmids_file: str
    corpus_dir: str = "data/corpus_search"
    cache_dir: str = "data/cache/llm"
    pdf_dir: str = "data/pdfs"
    title_batch: int = 50
    abstract_batch: int = 4
    fulltext_max_chars: int = 60000
    concurrency: int = 4  # per reviewer
    role: str = ROLE
    compare_with: str | None = None  # phase 1 final decisions.csv, for the comparison
    reviewers: list[ReviewerConfig]


def load_config(path: str | Path) -> Phase2Config:
    return Phase2Config.model_validate(yaml.safe_load(Path(path).read_text()))


# --------------------------------------------------------------------------- one reviewer, end to end


def _chunks(items: list, n: int) -> list[list]:
    return [items[i:i + n] for i in range(0, len(items), n)]


def _parallel(fn, jobs: list, workers: int) -> list:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(fn, jobs))


def review(reviewer: ReviewerConfig, llm: LLM, articles: list[Article], cfg: Phase2Config, codebook: Codebook,
           fulltext_source) -> tuple[dict[str, dict], list[dict]]:
    """One reviewer's whole review. Returns (record per PMID, open items). A record holds the reviewer's decision
    at each stage it reached and its verdict; open items are what a later run must retry (limits, errors)."""
    rec = {a.pmid: {"pmid": a.pmid, "title": a.title} for a in articles}
    votes: dict[str, dict[str, Vote]] = {a.pmid: {} for a in articles}
    open_items: list[dict] = []

    def stage(name, fn, items, size):
        def one(chunk):
            try:
                return chunk, fn(chunk, codebook, llm, reviewer.id, cfg.role)[0], None
            except Exception as e:  # a used-up limit or an error: these papers wait for the next run (cached)
                return chunk, None, f"{type(e).__name__}: {str(e)[:200]}"
        done, failed = 0, 0
        for chunk, out, err in _parallel(one, _chunks(items, size), cfg.concurrency):
            if err:
                failed += len(chunk)
                open_items += [{"pmid": a.pmid, "reviewer": reviewer.id, "stage": name, "error": err} for a in chunk]
                continue
            done += len(chunk)
            for a in chunk:
                votes[a.pmid][name] = out[a.pmid]
        print(f"  reviewer {reviewer.id} · {name}: {done} judged, {failed} waiting", file=sys.stderr)
        return failed == 0

    # 1. titles (all records)
    if not stage("title", batch.run_titles, articles, cfg.title_batch):
        return _verdicts(rec, votes, codebook), open_items  # abstracts wait until every title is judged
    kept = [a for a in articles if votes[a.pmid]["title"].labels == ["include"]]
    # 2. abstracts (only what this reviewer kept)
    if not stage("abstract", batch.run_abstracts, kept, cfg.abstract_batch):
        return _verdicts(rec, votes, codebook), open_items
    kept = [a for a in kept if votes[a.pmid]["abstract"].labels == ["include"]]

    # 3. full text (only what this reviewer kept; free full text only)
    from adrd_review.tools.fulltext import fit_text

    def full(a: Article):
        res = fulltext_source(a)
        if not res.found:
            return a, None, "not assessed"
        text, _ = fit_text(res.fulltext, cfg.fulltext_max_chars)
        try:
            vote, _ = eligibility.run(a.model_copy(update={"fulltext": text}), codebook, llm, reviewer.id, cfg.role)
            return a, vote, None
        except Exception as e:
            return a, None, f"{type(e).__name__}: {str(e)[:200]}"

    n_done = n_open = 0
    for a, vote, err in _parallel(full, kept, cfg.concurrency):
        if vote is not None:
            votes[a.pmid]["full text"] = vote
            n_done += 1
        elif err == "not assessed":
            rec[a.pmid]["not assessed"] = True
        else:
            open_items.append({"pmid": a.pmid, "reviewer": reviewer.id, "stage": "full text", "error": err})
            n_open += 1
    print(f"  reviewer {reviewer.id} · full text: {n_done} judged, {n_open} waiting", file=sys.stderr)
    return _verdicts(rec, votes, codebook), open_items


def _verdicts(rec: dict, votes: dict, codebook: Codebook) -> dict[str, dict]:
    names = {r.id: f"{r.id} {r.name}" for r in codebook.rules}
    for pmid, r in rec.items():
        v = votes[pmid]
        for st in STAGES:
            if st in v:
                r[f"{st} decision"] = v[st].labels[0]
        verdict, stage, src = "open", "", None
        if v.get("title") and v["title"].labels == ["exclude"]:
            verdict, stage, src = "excluded", "title", v["title"]
        elif v.get("abstract") and v["abstract"].labels == ["exclude"]:
            verdict, stage, src = "excluded", "abstract", v["abstract"]
        elif r.get("not assessed"):
            verdict, stage = "not assessed", "full text"
        elif v.get("full text"):
            src, stage = v["full text"], "full text"
            verdict = {"include": "included", "exclude": "excluded", "unclear": "unclear"}[src.labels[0]]
        r.update({"verdict": verdict, "stage": stage, "model": src.model if src else "",
                  "reason": ", ".join(names.get(x, x) for x in (src.rule_ids if src else [])),
                  "quotes": " | ".join(e.quote for e in (src.evidence if src else []) if e.quote),
                  "rationale": src.rationale if src else ""})
    return rec


# --------------------------------------------------------------------------- majority vote


def majority(verdicts: list[dict]) -> dict:
    """The panel's outcome from the reviewers' verdicts (see the module docstring)."""
    order = {s: i for i, s in enumerate(STAGES)}
    excl = sorted(((order[v["stage"]], v) for v in verdicts if v["verdict"] == "excluded"), key=lambda t: t[0])
    need = len(verdicts) // 2 + 1
    if len(excl) >= need:
        stage_i, _ = excl[need - 1]
        reasons = Counter(v["reason"].split(", ")[0] for _, v in excl[:need] if v["reason"])
        return {"outcome": "excluded", "stage": STAGES[stage_i],
                "reason": reasons.most_common(1)[0][0] if reasons else ""}
    if sum(v["verdict"] == "included" for v in verdicts) >= need:
        return {"outcome": "included", "stage": "full text", "reason": "all rules met"}
    if any(v["verdict"] == "open" for v in verdicts):
        return {"outcome": "open", "stage": "", "reason": "a reviewer has not finished"}
    if sum(v["verdict"] == "not assessed" for v in verdicts) >= 1 and \
            not any(v["verdict"] in ("included", "unclear") for v in verdicts):
        return {"outcome": "not assessed", "stage": "full text", "reason": "no free full text"}
    return {"outcome": "human review", "stage": "full text", "reason": "no majority: "
            + ", ".join(sorted(v["verdict"] for v in verdicts))}


def agreement(records: dict[str, dict[str, dict]], ids: list[str]) -> dict:
    """Fleiss' kappa, Gwet's AC1 and pairwise Cohen's kappa among the reviewers, per stage (papers all of them
    judged at that stage) and on the final verdicts (papers all of them finished)."""
    out = {}
    pmids = list(next(iter(records.values())))
    for st in (*(f"{s} decision" for s in STAGES), "verdict"):
        rows = [[records[r][p].get(st) for r in ids] for p in pmids]
        rows = [row for row in rows if all(x not in (None, "open") for x in row)]
        if len(rows) < 2:
            continue
        pairs = {f"{ids[i]}-{ids[j]}": round(cohen_kappa([r[i] for r in rows], [r[j] for r in rows]), 3)
                 for i in range(len(ids)) for j in range(i + 1, len(ids))}
        out["final verdict" if st == "verdict" else st.removesuffix(" decision")] = {
            "papers": len(rows), "fleiss_kappa": round(fleiss_kappa(rows), 3), "gwet_ac1": round(gwet_ac1(rows), 3),
            "all_agree": round(sum(len(set(r)) == 1 for r in rows) / len(rows), 3), "pairwise_cohen_kappa": pairs}
    return out


def compare(final: dict[str, dict], phase1_csv: Path) -> dict:
    """Phase 1 vs phase 2 on the papers both decided (included vs not included)."""
    p1 = {r["pmid"]: r for r in csv.DictReader(phase1_csv.open())}
    decided = ("included", "excluded")
    both = [(p1[p]["outcome"], f["outcome"]) for p, f in final.items()
            if p in p1 and p1[p]["outcome"] in decided and f["outcome"] in decided]
    table = Counter(f"phase 1 {a} / phase 2 {b}" for a, b in both)
    return {"papers_decided_by_both": len(both),
            "agreement": round(sum(a == b for a, b in both) / len(both), 3) if both else None,
            "cohen_kappa": round(cohen_kappa([a for a, _ in both], [b for _, b in both]), 3) if both else None,
            "table": dict(table)}


# --------------------------------------------------------------------------- run


def run(cfg: Phase2Config, config_path: Path, limit: int | None = None, llm_for=None, fulltext_source=None) -> Path:
    codebook = load_codebook(cfg.codebook)
    pmids = [p.strip() for p in Path(cfg.pmids_file).read_text().split() if p.strip()][: limit or None]
    articles_by, _ = fetch_articles(pmids, cache_dir=cfg.corpus_dir)
    articles = [articles_by[p] for p in pmids if p in articles_by]
    if fulltext_source is None:
        from adrd_review.tools.fulltext import get_fulltext
        fulltext_source = lambda a: get_fulltext(a, pdf_dir=cfg.pdf_dir)  # noqa: E731
    llm_for = llm_for or (lambda r: build_llm(r.model, cfg.cache_dir, codebook))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path("data/runs") / Path(config_path).stem / stamp
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"Phase 2 {cfg.name}: {len(articles)} papers, reviewers {[r.id for r in cfg.reviewers]} -> {out}", file=sys.stderr)

    def one(r: ReviewerConfig):
        return r.id, *review(r, llm_for(r), articles, cfg, codebook, fulltext_source)

    results = _parallel(one, cfg.reviewers, len(cfg.reviewers))  # reviewers work side by side, independently
    records = {rid: rec for rid, rec, _ in results}
    open_items = [o for _, _, items in results for o in items]
    ids = [r.id for r in cfg.reviewers]

    cols = ["pmid", "title", "verdict", "stage", "reason", "title decision", "abstract decision", "full text decision",
            "quotes", "rationale", "model"]
    for rid in ids:
        with (out / f"reviewer_{rid}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(records[rid].values())
    final = {}
    for a in articles:
        vs = [records[r][a.pmid] for r in ids]
        final[a.pmid] = {"pmid": a.pmid, "title": a.title, **majority(vs),
                         **{f"reviewer {r}": f"{v['verdict']}" + (f" ({v['stage']})" if v["verdict"] == "excluded" else "")
                            for r, v in zip(ids, vs)}}
    with (out / "final.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(final.values()))))
        w.writeheader()
        w.writerows(final.values())

    oc = Counter((f["outcome"], f["stage"]) for f in final.values())
    t_ex, a_ex = oc[("excluded", "title")], oc[("excluded", "abstract")]
    summary = {
        "records": len(final),
        "stage_table": {"titles": {"screened": len(final), "excluded": t_ex, "passed": len(final) - t_ex},
                        "abstracts": {"screened": len(final) - t_ex, "excluded": a_ex,
                                      "passed": len(final) - t_ex - a_ex},
                        "full text": {k: oc[(k, "full text")] for k in ("included", "excluded", "human review",
                                                                       "not assessed")},
                        "open": sum(f["outcome"] == "open" for f in final.values())},
        "per_reviewer": {r: dict(Counter(f"{v['verdict']}" + (f" at {v['stage']}" if v["verdict"] == "excluded" else "")
                                         for v in records[r].values())) for r in ids},
        "agreement": agreement(records, ids),
        "exclusion_reasons": {st: dict(Counter(f["reason"] for f in final.values()
                                               if f["outcome"] == "excluded" and f["stage"] == st).most_common())
                              for st in STAGES},
        "models": {r.id: r.model.label for r in cfg.reviewers},
        "seconds": round(time.monotonic() - started, 1),
    }
    if cfg.compare_with and Path(cfg.compare_with).exists():
        summary["compared_with_phase_1"] = compare(final, Path(cfg.compare_with))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "failures.json").write_text(json.dumps(open_items, indent=2) + "\n")  # run_until_done counts these
    (out / "run_meta.json").write_text(json.dumps({"config": cfg.model_dump(), "codebook_version": codebook.version,
                                                   "papers": len(articles), "limit": limit}, indent=2, default=str) + "\n")
    st = summary["stage_table"]
    print(f"\nTitles {st['titles']} · Abstracts {st['abstracts']} · Full text {st['full text']} · open {st['open']}",
          file=sys.stderr)
    print(f"Open items: {len(open_items)} · saved in {out}", file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adrd_review.phase2")
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None, help="only the first N papers (a pilot)")
    args = ap.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv()
    run(load_config(args.config), Path(args.config), args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
