"""Batch screening (many titles / several abstracts per call), deferral on failed batches, missing keys."""

from pathlib import Path

import pytest

from adrd_review.agents import batch
from adrd_review.codebook import load_codebook
from adrd_review.graph.runner import RunConfig, prepare_batches, run_articles
from adrd_review.graph.pipeline import PipelineOptions, Reviewer, build_pipeline
from adrd_review.fake_model import default_responders
from adrd_review.llm import FakeLLM, LimitReached, ModelConfig, build_llm
from adrd_review.schemas import (Article, InclusionCheck, ScreeningBatchAnswer, ScreeningBatchItem,
                                 TitleBatchAnswer, TitleHit)

ROOT = Path(__file__).resolve().parents[1]
CB = load_codebook(ROOT / "configs/codebook_aiml.yaml")
ML = Article(pmid="1", title="Deep learning on MRI to detect Alzheimer's disease in older adults.",
             abstract="We trained a convolutional neural network on MRI scans of 800 older adults to predict dementia.")
PD = Article(pmid="2", title="Retinal imaging classification for Parkinson's disease using a neural network.",
             abstract="A neural network classified Parkinson's disease from retinal images.")
REV = Article(pmid="3", title="Machine learning for dementia: a systematic review.", abstract="We reviewed 40 studies.")


def fixed(schema, answer):
    return FakeLLM(ModelConfig(provider="fake", model="x"), {schema: lambda *_: answer})


def test_title_batch_keeps_unreported_titles_and_checks_each_quote_against_its_own_title():
    ans = TitleBatchAnswer(excluded=[
        TitleHit(n=2, rule_id="I1", quote="Parkinson's disease"),
        TitleHit(n=3, rule_id="E1", quote="a systematic review"),
        TitleHit(n=1, rule_id="I1", quote="Parkinson's disease"),  # quote from another title: ignored
        TitleHit(n=9, rule_id="I1", quote="x"),  # no such title: ignored
    ])
    votes, usages = batch.run_titles([ML, PD, REV], CB, fixed(TitleBatchAnswer, ans))
    assert votes["1"].labels == ["include"] and any("not in this title" in w for w in votes["1"].warnings)
    assert (votes["2"].labels, votes["2"].rule_checks) == (["exclude"], {"I1": "not_met"})
    assert (votes["3"].labels, votes["3"].rule_checks) == (["exclude"], {"E1": "applies"})
    assert all(u.batch_size == 3 for u in usages.values())


def test_abstract_batch_decides_each_article_like_a_single_call():
    def item(n, status, quote):
        return ScreeningBatchItem(n=n, inclusion=[InclusionCheck(rule_id="I1", status=status, quote=quote)],
                                  exclusion=[], rationale="r", confidence=0.9)
    ans = ScreeningBatchAnswer(items=[item(1, "met", "predict dementia"),
                                      item(2, "not_met", "classified Parkinson's disease")])  # article 3 left out
    votes, _ = batch.run_abstracts([ML, PD, REV], CB, fixed(ScreeningBatchAnswer, ans))
    assert votes["1"].labels == ["include"] and votes["2"].labels == ["exclude"]
    assert votes["3"].labels == ["include"] and "missing from the batch" in votes["3"].warnings[0]
    # a not_met quoting ANOTHER article is not evidence
    bad = ScreeningBatchAnswer(items=[item(1, "not_met", "classified Parkinson's disease")])
    assert batch.run_abstracts([ML, PD], CB, fixed(ScreeningBatchAnswer, bad))[0]["1"].labels == ["include"]


def reviewers(llms):
    return [Reviewer(rid, m, m) for rid, m in zip("ABC", llms)]


def test_prepared_votes_feed_the_graph_and_failed_batches_defer_their_articles():
    llms = [FakeLLM(ModelConfig(provider="fake", model=s), default_responders(CB, s))
            for s in ("keyword-rules", "keyword-strict", "keyword-lenient")]
    revs = reviewers(llms)
    cfg = RunConfig(name="t", gold=None, title_stage=True, title_batch=2, abstract_batch=2)
    initial, deferred = prepare_batches(cfg, CB, [ML, PD, REV], revs, concurrency=2)
    assert not deferred and set(initial["1"]["prepared"]) == {"title", "screening"}
    assert "screening" not in initial["3"]["prepared"]  # the review was excluded at title: no abstract call
    g = build_pipeline(CB, llms[0], llms[0], revs, PipelineOptions(title_stage=True, triage=False))
    states, failures, _ = run_articles(g, [ML, PD, REV], initial=initial)
    by = {s["article"].pmid: s for s in states}
    assert by["3"]["screening_decision"].method == "title_excluded"
    assert by["1"]["screening_decision"].labels == ["include"] and "triage" not in by["1"]
    assert len(by["1"]["screening_votes"]) == 3 and not failures

    class Broke:
        config = ModelConfig(provider="groq", model="m")

        def structured(self, *a, **k):
            raise LimitReached("daily limit")

    revs[1] = Reviewer("B", Broke(), Broke())
    _, deferred = prepare_batches(cfg, CB, [ML, PD, REV], revs, concurrency=2)
    assert set(deferred) == {"1", "2", "3"} and all(v.startswith("title") for v in deferred.values())


def test_missing_key_waits_instead_of_failing_the_run(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    llm = build_llm(ModelConfig(provider="openrouter", model="x:free"), tmp_path)
    with pytest.raises(LimitReached, match="OPENROUTER_API_KEY"):
        llm.structured(TitleBatchAnswer, "s", "u", role="title")
