"""Two-stage review: full-text sources, the strict eligibility stage, scoring, preflight."""

from pathlib import Path

import pytest

from adrd_review.agents import eligibility
from adrd_review.agents.human import GoldHuman
from adrd_review.codebook import load_codebook
from adrd_review.consensus import combine_eligibility
from adrd_review.eval.score import evaluate
from adrd_review.fake_model import default_responders
from adrd_review.gold import GoldRecord
from adrd_review.graph.pipeline import PipelineOptions, Reviewer, build_pipeline
from adrd_review.graph.runner import to_prediction
from adrd_review.llm import FakeLLM, ModelConfig
from adrd_review.preflight import check
from adrd_review.schemas import Article, ExclusionCheck, InclusionCheck, ScreeningAnswer, Vote
from adrd_review.tools import pdf as pdf_tool
from adrd_review.tools.fulltext import FullTextResult, fit_text, get_fulltext
from adrd_review.tools.pmc import FullText, Section

ROOT = Path(__file__).resolve().parents[1]
CODEBOOK = load_codebook(ROOT / "configs/codebook.yaml")
FAKE = ModelConfig(provider="fake", model="keyword-rules")

PAPER = FullText(pmid="9", pmcid="PMC9", sections=[
    Section(title="Methods", text="We recruited 412 community-dwelling adults aged 65 and over. All completed the MMSE. "
                                  "Dementia was diagnosed with DSM-IV criteria by a geriatrician blind to the MMSE."),
    Section(title="Results", text="At a cut-off of 24 the MMSE had sensitivity 0.81 and specificity 0.88 for dementia."),
])
ART = Article(pmid="9", title="The MMSE for dementia in older adults.", abstract="We evaluated the MMSE for dementia.", pmcid="PMC9")
Q_METHODS = "All completed the MMSE."
Q_REF = "Dementia was diagnosed with DSM-IV criteria by a geriatrician blind to the MMSE."


def with_text(article=ART):
    text, _ = fit_text(PAPER, 80_000)
    return article.model_copy(update={"fulltext": text})


def answer(inclusion, exclusion=()):
    return ScreeningAnswer(
        inclusion=[InclusionCheck(rule_id=r, status=s, quote=q) for r, s, q in inclusion],
        exclusion=[ExclusionCheck(rule_id=r, status=s, quote=q) for r, s, q in exclusion],
        rationale="r", confidence=0.9)


# ---- stage 2 decision rule ---------------------------------------------------------


def test_fulltext_include_needs_every_inclusion_rule_met():
    art = with_text()
    all_met = answer([(r, "met", Q_METHODS) for r in ("I1", "I2", "I3", "I4")])
    assert eligibility.decide(all_met, CODEBOOK, art)[0] == "include"
    one_open = answer([("I1", "met", Q_METHODS), ("I2", "met", Q_REF), ("I3", "unclear", None), ("I4", "met", Q_METHODS)])
    decision, checks, open_rules, *_ = eligibility.decide(one_open, CODEBOOK, art)
    assert decision == "unclear" and open_rules == ["I3"]  # stage 1 would include; stage 2 asks a human


def test_fulltext_clear_violation_excludes_and_quotes_may_come_from_the_paper():
    art = with_text()
    a = answer([("I1", "met", Q_METHODS), ("I2", "not_met", Q_REF)])
    decision, _, rule_ids, evidence, warnings = eligibility.decide(a, CODEBOOK, art)
    assert decision == "exclude" and rule_ids == ["I2"]
    assert not any("quote" in w for w in warnings)  # the quote is in the full text, not the abstract
    assert eligibility.decide(a, CODEBOOK, ART)[0] != "exclude"  # without the full text that quote is not real


def test_eligibility_agent_refuses_to_run_without_full_text():
    llm = FakeLLM(FAKE, default_responders(CODEBOOK))
    with pytest.raises(ValueError, match="needs the full text"):
        eligibility.run(ART, CODEBOOK, llm)
    vote, usage = eligibility.run(with_text(), CODEBOOK, llm)
    assert vote.field == "eligibility" and usage.role == "eligibility"
    assert "REMINDER" in eligibility.user_prompt(with_text())  # instructions repeated after the paper


def ev(rid, label, conf=0.8):
    return Vote(reviewer_id=rid, model="m", field="eligibility", labels=[label], confidence=conf)


def test_combine_eligibility():
    assert combine_eligibility([ev("A", "include"), ev("B", "include"), ev("C", "include")]).method == "unanimous"
    d = combine_eligibility([ev("A", "include"), ev("B", "include"), ev("C", "exclude")])
    assert (d.labels, d.method) == (["include"], "majority")
    d = combine_eligibility([ev("A", "include"), ev("B", "exclude"), ev("C", "unclear")])
    assert (d.labels, d.method) == (["unclear"], "no_majority")  # nobody has a majority -> human


# ---- full-text sources --------------------------------------------------------------


def test_source_ladder_prefers_team_pdf_then_pmc_and_explains_misses(tmp_path, monkeypatch):
    from adrd_review.tools import fulltext as ft

    monkeypatch.setattr(ft, "fetch_fulltext", lambda article, cache_dir=None: PAPER if article.pmcid else None)
    res = get_fulltext(ART, pdf_dir=tmp_path, allow_download=False)
    assert res.source == "pmc" and res.found

    monkeypatch.setattr(ft.pdf_tool, "extract", lambda path: (PAPER.sections, "ok"))
    (tmp_path / "9.pdf").write_bytes(b"%PDF-1.4 test")
    assert get_fulltext(ART, pdf_dir=tmp_path, allow_download=False).source == "team_pdf"

    monkeypatch.setattr(ft.pdf_tool, "extract", lambda path: ([], "only 120 characters extracted (scanned or image-only PDF?)"))
    res = get_fulltext(ART, pdf_dir=tmp_path, allow_download=False)
    assert res.source == "pmc"  # bad PDF -> falls back to the next source

    none = get_fulltext(Article(pmid="7", title="T", doi="10.1/x"), pdf_dir=tmp_path, allow_download=False)
    assert not none.found and "not in PubMed Central" in none.reason and none.link == "https://doi.org/10.1/x"


def test_fit_text_keeps_whole_paper_or_puts_methods_and_results_first():
    text, truncated = fit_text(PAPER, 80_000)
    assert not truncated and text.startswith("[METHODS]") and "[RESULTS]" in text
    long = FullText(pmid="1", pmcid="P", sections=[
        Section(title="Introduction", text="Background sentence. " * 400),
        Section(title="Methods", text="We sampled adults. " * 50),
    ])
    text, truncated = fit_text(long, 1500)
    assert truncated and text.startswith("[METHODS]") and len(text) <= 1600


def test_pdf_quality_check_and_section_split():
    assert pdf_tool.quality("x" * 100)[0] is False                      # too short: scanned PDF
    assert pdf_tool.quality("@#$%^&*()1234567890" * 400)[0] is False     # garbled
    good = ("Methods\nWe recruited adults from general practice and tested them.\n" * 80
            + "Results\nSensitivity was high.\nReferences\n1. Someone et al.\n")
    assert pdf_tool.quality(good)[0] is True
    sections = pdf_tool.split_sections(good)
    assert {s.title for s in sections} == {"Methods", "Results"}  # reference list dropped
    assert pdf_tool.clean("screen-\ning  test") == "screening test"
    assert pdf_tool.extract(__file__)[1].startswith("could not read PDF")


# ---- the two-stage graph ---------------------------------------------------------------


def graph(source, human=None, adjudicator=False):
    llm = FakeLLM(FAKE, default_responders(CODEBOOK))
    opts = PipelineOptions(eligibility=True, fulltext_source=source, human=human, adjudicator=llm if adjudicator else None)
    return build_pipeline(CODEBOOK, llm, llm, [Reviewer("A", llm, llm)], opts)


def test_included_article_gets_full_text_then_stage_2_then_annotation():
    s = graph(lambda a: FullTextResult(PAPER, "pmc", "PMC9")).invoke({"article": ART})
    assert s["screening_decision"].labels == ["include"]
    assert s["fulltext_info"]["used"] and s["fulltext_info"]["source"] == "pmc"
    assert s["eligibility_decision"].labels == ["include"]  # fake: all four rules met in the full paper
    assert s["annotation_decisions"]["instrument"].labels == ["MMSE"]
    pred = to_prediction(s, two_stage=True)
    assert (pred.screening, pred.fulltext) == ("include", "include") and pred.labels


def test_no_full_text_is_flagged_never_excluded():
    none = FullTextResult(None, None, "not in PubMed Central; no free copy found (subscription)", link="https://doi.org/x")
    s = graph(lambda a: none).invoke({"article": ART})
    d = s["eligibility_decision"]
    assert (d.labels, d.method) == (["unclear"], "no_fulltext")
    assert [e.reason for e in s["escalations"]] == ["no_fulltext"] and "https://doi.org/x" in s["escalations"][0].detail
    assert to_prediction(s, two_stage=True).fulltext == "include"  # kept in, for a human
    assert "annotation_decisions" in s  # still annotated from the abstract


def test_human_decides_when_full_text_is_missing():
    none = FullTextResult(None, None, "no full text", link=None)
    gold = {"9": GoldRecord(pmid="9", screening="include", fulltext="exclude")}
    s = graph(lambda a: none, human=GoldHuman(gold)).invoke({"article": ART})
    assert s["eligibility_decision"].method == "human" and s["eligibility_decision"].labels == ["exclude"]
    assert "annotation_decisions" not in s  # excluded at full text: no annotation
    pred = to_prediction(s, two_stage=True)
    assert (pred.screening, pred.fulltext, pred.labels) == ("include", "exclude", {})


def test_article_excluded_at_stage_1_never_fetches_full_text():
    called = []
    off_topic = Article(pmid="5", title="Exercise and blood pressure.", abstract="A trial of exercise in young adults.")
    s = graph(lambda a: called.append(a.pmid)).invoke({"article": off_topic})
    assert called == [] and "eligibility_decision" not in s
    assert to_prediction(s, two_stage=True).fulltext == "exclude"


# ---- scoring both stages -------------------------------------------------------------------


def test_scorer_reports_stage_2_and_annotates_on_final_includes():
    g = lambda p, s, f, lab=None: GoldRecord(pmid=p, screening=s, fulltext=f, labels={"instrument": lab} if lab else {})  # noqa: E731
    gold = [g("1", "include", "include", ["MMSE"]), g("2", "include", "exclude"), g("3", "exclude", "exclude")]
    pred = [g("1", "include", "include", ["MMSE"]), g("2", "include", "include", ["MoCA"]), g("3", "exclude", "exclude")]
    r = evaluate(gold, pred, CODEBOOK)
    assert r["screening"]["recall"] == 1.0 and r["screening"]["precision"] == 1.0
    assert r["fulltext"]["recall"] == 1.0 and r["fulltext"]["precision"] == 0.5
    assert r["annotation"]["instrument"]["n_items"] == 1  # only the gold FINAL include is annotated
    one_stage = [GoldRecord(pmid=p.pmid, screening=p.screening, labels=p.labels) for p in pred]
    assert evaluate(gold, one_stage, CODEBOOK)["fulltext"] is None


# ---- preflight ------------------------------------------------------------------------------


@pytest.mark.skipif(not (ROOT / "data/gold/gold.csv").exists(), reason="needs the placeholder gold standard")
def test_preflight_passes_fake_config_and_catches_token_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    rep = check("configs/runs/E4_twostage_fake.yaml", calls=False)
    assert not rep.failed
    assert any("full text:" in t for _, t in rep.lines)

    wrong = tmp_path / "wrong.yaml"
    wrong.write_text((ROOT / "configs/runs/E2_groq.yaml").read_text() + "\neligibility: fulltext\n")
    monkeypatch.setenv("GROQ_API_KEY", "x")
    rep = check(str(wrong), calls=False)
    assert rep.failed and any("tokens/min" in t for level, t in rep.lines if level == "FAIL")


@pytest.mark.skipif(not (ROOT / "data/gold/gold.csv").exists(), reason="needs the placeholder gold standard")
def test_preflight_fails_without_a_key(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    rep = check("configs/runs/E4_twostage_gemini.yaml", calls=False)
    assert rep.failed and any("GOOGLE_API_KEY is not set" in t for _, t in rep.lines)
