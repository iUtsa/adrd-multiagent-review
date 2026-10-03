"""Three-stage review (titles -> abstracts -> full text), stage counts, unlabelled corpora, draft codebook."""

import json
from pathlib import Path

import pytest

from adrd_review.agents import title_screening
from adrd_review.codebook import load_codebook
from adrd_review.consensus import combine_title
from adrd_review.eval.flow import flow_counts, flow_figure, format_flow
from adrd_review.fake_model import default_responders
from adrd_review.gold import GoldRecord
from adrd_review.graph.pipeline import PipelineOptions, Reviewer, build_pipeline
from adrd_review.graph.runner import RunConfig, select_pmids, to_prediction
from adrd_review.llm import FakeLLM, ModelConfig
from adrd_review.schemas import Article, ExclusionCheck, InclusionCheck, ScreeningAnswer, Vote
from adrd_review.tools.fulltext import FullTextResult
from adrd_review.tools.pmc import FullText, Section

ROOT = Path(__file__).resolve().parents[1]
CODEBOOK = load_codebook(ROOT / "configs/codebook.yaml")
DRAFT = load_codebook(ROOT / "configs/codebook_aiml_draft.yaml")

ON_TOPIC = Article(pmid="1", title="The MMSE for dementia in older adults.", abstract="We evaluated the MMSE for dementia.")
OFF_TOPIC = Article(pmid="2", title="Exercise and blood pressure in young athletes.",
                    abstract="A trial of exercise. Participants also completed the MMSE for dementia.")


def answer(inclusion, exclusion=()):
    return ScreeningAnswer(
        inclusion=[InclusionCheck(rule_id=r, status=s, quote=q) for r, s, q in inclusion],
        exclusion=[ExclusionCheck(rule_id=r, status=s, quote=q) for r, s, q in exclusion],
        rationale="r", confidence=0.9)


def tv(rid, label):
    return Vote(reviewer_id=rid, model="m", field="title", labels=[label], confidence=0.8)


# ---- the title agent and its consensus -------------------------------------------------


def test_title_panel_excludes_only_when_every_reviewer_excludes():
    assert combine_title([tv("A", "exclude"), tv("B", "exclude"), tv("C", "exclude")]).labels == ["exclude"]
    d = combine_title([tv("A", "exclude"), tv("B", "include"), tv("C", "exclude")])
    assert (d.labels, d.method) == (["include"], "any_include")  # one reviewer keeping it is enough
    assert combine_title([tv("A", "exclude")]).method == "single"


def test_title_agent_sees_only_the_title_and_quotes_must_come_from_it():
    seen = {}

    def responder(_schema, system, user):
        seen["user"], seen["system"] = user, system
        return answer([("I1", "unclear", None)])

    llm = FakeLLM(ModelConfig(provider="fake", model="x"), {ScreeningAnswer: responder})
    vote, usage = title_screening.run(OFF_TOPIC, CODEBOOK, llm)
    assert "Exercise and blood pressure" in seen["user"] and "MMSE" not in seen["user"]  # no abstract
    assert "TITLES ONLY" in seen["system"] and vote.field == "title" and usage.role == "title"
    # A not_met quoting the abstract is not evidence at the title stage -> unclear -> kept.
    from_abstract = answer([("I2", "not_met", "A trial of exercise.")])
    from adrd_review.agents import screening
    assert screening.decide(from_abstract, CODEBOOK, title_screening.title_only(OFF_TOPIC))[0] == "include"
    from_title = answer([("I2", "not_met", "Exercise and blood pressure in young athletes.")])
    assert screening.decide(from_title, CODEBOOK, title_screening.title_only(OFF_TOPIC))[0] == "exclude"


# ---- the three-stage graph -----------------------------------------------------------------


PAPER = FullText(pmid="1", pmcid="PMC1", sections=[
    Section(title="Methods", text="We recruited 412 community-dwelling adults aged 65 and over. All completed the MMSE. "
                                  "Dementia was diagnosed with DSM-IV criteria by a geriatrician blind to the MMSE."),
    Section(title="Results", text="At a cut-off of 24 the MMSE had sensitivity 0.81 and specificity 0.88 for dementia."),
])


def graph(title_stage=True, n=3, codebook=CODEBOOK, source=None):
    styles = ["keyword-rules", "keyword-strict", "keyword-lenient"]
    llms = [FakeLLM(ModelConfig(provider="fake", model=s), default_responders(codebook, s)) for s in styles[:n]]
    opts = PipelineOptions(title_stage=title_stage, eligibility=True,
                           fulltext_source=source or (lambda a: FullTextResult(PAPER, "pmc", "PMC1")))
    return build_pipeline(codebook, llms[0], llms[0], [Reviewer(c, m, m) for c, m in zip("ABC", llms)], opts)


def test_off_topic_title_stops_before_the_abstract_is_read():
    assert graph().invoke({"article": OFF_TOPIC})["title_decision"].method == "any_include"  # lenient reviewer keeps it
    s = graph(n=2).invoke({"article": OFF_TOPIC})
    assert s["title_decision"].labels == ["exclude"]
    assert not s.get("screening_votes") and "triage" not in s  # the abstract was never screened
    assert s["screening_decision"].method == "title_excluded"
    assert to_prediction(s, two_stage=True).screening == "exclude"


def test_on_topic_title_goes_through_all_three_stages():
    s = graph().invoke({"article": ON_TOPIC})
    assert s["title_decision"].labels == ["include"]
    assert s["screening_votes"] and s["eligibility_votes"]
    assert s["fulltext_info"]["source"] == "pmc"


def test_title_stage_off_changes_nothing():
    s = graph(title_stage=False).invoke({"article": OFF_TOPIC})
    assert not s.get("title_votes") and "title_decision" not in s and s["screening_votes"]


# ---- flow counts --------------------------------------------------------------------------


def test_flow_counts_each_stage_with_reasons_and_gold_losses(tmp_path):
    none = FullTextResult(None, None, "subscription", link=None)
    g = graph(n=2)
    no_text = Article(pmid="3", title="The MMSE for dementia in older adults (cohort).", abstract="We evaluated the MMSE for dementia.")
    states = [g.invoke({"article": a}) for a in (ON_TOPIC, OFF_TOPIC)]
    states.append(graph(n=2, source=lambda a: none).invoke({"article": no_text}))
    gold = {"1": GoldRecord(pmid="1", screening="include", fulltext="include"),
            "2": GoldRecord(pmid="2", screening="include", fulltext="exclude")}
    f = flow_counts(states, failures=1, codebook=CODEBOOK, gold_by=gold)
    assert f["identified"] == 4
    assert (f["title"]["screened"], f["title"]["excluded"], f["title"]["passed"]) == (3, 1, 2)
    assert f["title"]["gold_abstract_includes_lost"] == 1 and f["title"]["gold_final_includes_lost"] == 0
    assert list(f["title"]["reasons"]) == ["I1 Index test"]
    assert f["abstract"]["screened"] == 2 and f["fulltext"]["sought"] == 2
    assert f["fulltext"]["not_retrieved"] == 1 and f["fulltext"]["still_open"]["no_fulltext"] == 1
    assert f["included"]["total"] == 3 and f["included"]["waiting_for_human"] == 1 and f["included"]["failed_kept"] == 1
    assert f["included"]["gold_final_includes_kept"] == 1
    text = format_flow(f)
    assert "1. Title screening" in text and "3. Full text" in text and "INCLUDED" in text
    assert flow_figure(f, tmp_path / "flow.png").stat().st_size > 1000


def test_flow_without_title_stage_or_gold():
    s = graph(title_stage=False).invoke({"article": ON_TOPIC})
    f = flow_counts([s])
    assert "title" not in f and f["abstract"]["screened"] == 1 and "gold_final_includes_lost" not in f["abstract"]
    assert "1. Abstract screening" in format_flow(f)


# ---- unlabelled corpus and the draft codebook -----------------------------------------------


def test_unlabelled_run_reads_a_pmid_list(tmp_path):
    lst = tmp_path / "pmids.txt"
    lst.write_text("111\n222\n\n111\nnot-a-pmid\n")
    cfg = RunConfig(name="x", gold=None, pmids_file=str(lst))
    assert select_pmids(cfg, "all", []) == ["111", "222"]
    with pytest.raises(ValueError, match="gold file .* or a pmids_file"):
        select_pmids(RunConfig(name="y", gold=None), "all", [])


def test_human_gold_refused_without_gold():
    from adrd_review.graph.runner import build_graph

    fake = {"model": {"provider": "fake", "model": "keyword-rules"}}
    cfg = RunConfig(name="x", gold=None, human="gold", roles={"triage": fake, "extraction": fake, "screening": fake, "annotation": fake})
    with pytest.raises(ValueError, match="needs a gold standard"):
        build_graph(cfg, CODEBOOK, [])


def test_draft_codebook_is_marked_and_names_the_real_topic():
    assert DRAFT.status == "draft" and "machine learning" in DRAFT.topic
    assert [r.id for r in DRAFT.screening.inclusion] == ["I1", "I2", "I3", "I4"]
    assert all(r.keywords for r in DRAFT.screening.inclusion)
    from adrd_review.agents import screening
    assert "scoping review of AI and machine learning" in screening.system_prompt(DRAFT)
    assert "keywords" not in screening.system_prompt(DRAFT).lower()  # stand-in hints never reach a real model
    # the placeholder codebook's prompts are unchanged (cached answers stay valid)
    assert "scoping review of dementia screening research" in screening.system_prompt(CODEBOOK)


def test_fake_model_follows_the_draft_codebook_through_three_stages():
    ml = Article(pmid="7", title="Deep learning on MRI to detect Alzheimer's disease in older adults.",
                 abstract="We trained a convolutional neural network on ADNI MRI scans of 800 older adults to predict dementia.")
    review = Article(pmid="8", title="Machine learning for dementia: a systematic review.", abstract="We reviewed 40 studies.")
    animal = Article(pmid="9", title="Gut microbes in young mice.", abstract="Mice were fed a diet.")
    g = graph(codebook=DRAFT, source=lambda a: FullTextResult(None, None, "none", link=None))
    s = g.invoke({"article": ml})
    assert s["title_decision"].labels == ["include"] and s["screening_decision"].labels == ["include"]
    assert "ADNI" in s["annotation_decisions"]["dataset_source"].labels
    assert g.invoke({"article": review})["screening_decision"].method == "title_excluded"  # E1 from the title
    assert g.invoke({"article": animal})["screening_decision"].method == "title_excluded"


@pytest.mark.skipif(not (ROOT / "data/search/prof_query_pmids.txt").exists(), reason="needs the professor's search results")
def test_preflight_estimates_the_unlabelled_search_run(monkeypatch):
    from adrd_review.preflight import check

    monkeypatch.chdir(ROOT)
    rep = check("configs/runs/PROF_search_fake.yaml", calls=False, limit=20)
    assert not rep.failed
    assert any("DRAFT" in t for _, t in rep.lines) and any("unlabelled" in t for _, t in rep.lines)
    assert json.loads(json.dumps(rep.lines))  # plain data
