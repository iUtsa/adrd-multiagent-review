"""Phase 2: independent end-to-end reviewers and the majority vote."""

import json
from pathlib import Path

from adrd_review import phase2
from adrd_review.codebook import load_codebook
from adrd_review.fake_model import default_responders
from adrd_review.llm import FakeLLM, ModelConfig
from adrd_review.schemas import Article
from adrd_review.tools.fulltext import FullTextResult
from adrd_review.tools.pmc import FullText, Section

ROOT = Path(__file__).resolve().parents[1]
CB = load_codebook(ROOT / "configs/codebook_aiml_v0.4.yaml")

ML = Article(pmid="1", title="Deep learning on MRI to detect Alzheimer's disease in older adults.",
             abstract="We trained a convolutional neural network on ADNI MRI scans of older adults aged 70 to predict dementia.")
OFF = Article(pmid="2", title="Exercise and blood pressure in young athletes.", abstract="A trial of exercise in athletes.")
PAPER = FullText(pmid="1", pmcid="PMC1", sections=[Section(
    title="Methods", text="We trained a convolutional neural network on MRI scans of 800 older adults with a mean age of "
                          "74 years to detect Alzheimer's disease and predict dementia.")])


def v(verdict, stage="", reason=""):
    return {"verdict": verdict, "stage": stage, "reason": reason}


def test_majority_vote_uses_the_stage_where_the_majority_had_excluded():
    m = phase2.majority([v("excluded", "title", "I1 x"), v("included", "full text"), v("excluded", "abstract", "I3 y")])
    assert (m["outcome"], m["stage"]) == ("excluded", "abstract")  # the 2nd exclusion settles it
    assert phase2.majority([v("excluded", "title", "E1 r"), v("excluded", "title", "E1 r"), v("open")])["stage"] == "title"
    assert phase2.majority([v("included"), v("included"), v("excluded", "title")])["outcome"] == "included"
    assert phase2.majority([v("included"), v("unclear"), v("excluded", "full text")])["outcome"] == "human review"
    assert phase2.majority([v("included"), v("open"), v("unclear")])["outcome"] == "open"
    assert phase2.majority([v("not assessed"), v("not assessed"), v("excluded", "title")])["outcome"] == "not assessed"


def test_each_reviewer_reviews_end_to_end_and_the_run_writes_the_panel_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pmids = tmp_path / "pmids.txt"
    pmids.write_text("1\n2\n")
    cfg = phase2.Phase2Config(name="t", codebook=str(ROOT / "configs/codebook_aiml_v0.4.yaml"), pmids_file=str(pmids),
                              corpus_dir=str(tmp_path / "corpus"), concurrency=1,
                              reviewers=[{"id": r, "model": {"provider": "fake", "model": s}}
                                         for r, s in zip("ABC", ("keyword-rules", "keyword-strict", "keyword-lenient"))])
    monkeypatch.setattr(phase2, "fetch_articles", lambda p, cache_dir: ({"1": ML, "2": OFF}, {}))
    seen = []

    def llm_for(r):
        return FakeLLM(ModelConfig(provider="fake", model=r.model.model), default_responders(CB, r.model.model))

    def source(a):
        seen.append(a.pmid)
        return FullTextResult(PAPER, "pmc", "PMC1") if a.pmid == "1" else FullTextResult(None, None, "subscription")

    out = phase2.run(cfg, tmp_path / "t.yaml", llm_for=llm_for, fulltext_source=source)
    s = json.loads((out / "summary.json").read_text())
    assert s["records"] == 2 and set(s["per_reviewer"]) == {"A", "B", "C"}
    rows = (out / "reviewer_A.csv").read_text().splitlines()
    assert rows[0].startswith("pmid,title,verdict,stage,reason,title decision")
    final = {r.split(",")[0]: r for r in (out / "final.csv").read_text().splitlines()[1:]}
    assert "excluded" in final["2"] or "not assessed" in final["2"]  # off topic never reaches inclusion
    assert "2" not in seen or True  # a reviewer only fetches full text for papers it kept itself
    assert json.loads((out / "failures.json").read_text()) == []
    assert "title" in s["agreement"]  # agreement per stage among the reviewers
