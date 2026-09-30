"""Agents, model factory, cache and the E1 pipeline, tested without any API key."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from adrd_review.agents import annotation, screening
from adrd_review.agents.common import check_quotes, render_article, render_screening_rules
from adrd_review.codebook import load_codebook
from adrd_review.fake_model import default_responders
from adrd_review.graph.pipeline import Reviewer, build_pipeline
from adrd_review.graph.runner import load_run_config, run_articles, to_prediction
from adrd_review.llm import FakeLLM, LangChainLLM, LLMError, ModelConfig, ResponseCache
from adrd_review.schemas import (
    AnnotationAnswer,
    Article,
    DimensionAnnotation,
    EvidenceSpan,
    ScreeningAnswer,
)

ROOT = Path(__file__).resolve().parents[1]
CODEBOOK = load_codebook(ROOT / "configs/codebook.yaml")
FAKE_CFG = ModelConfig(provider="fake", model="keyword-rules")

INCLUDE_ART = Article(
    pmid="1",
    title="Accuracy of the Mini-Mental State Examination for dementia in primary care.",
    abstract="We tested the MMSE against DSM-IV dementia diagnosis in 500 primary care patients aged 65+. "
    "Sensitivity was 0.85.",
)
EXCLUDE_ART = Article(pmid="2", title="Exercise and blood pressure in young adults.", abstract="A trial of exercise.")
NO_ABSTRACT = Article(pmid="3", title="Letter: dementia care costs.")


def fake_llm():
    return FakeLLM(FAKE_CFG, default_responders(CODEBOOK))


# ---- prompts ----------------------------------------------------------------


def test_prompts_are_built_from_the_codebook():
    rules = render_screening_rules(CODEBOOK)
    for rule in CODEBOOK.rules:
        assert f"- {rule.id} {rule.name}:" in rules
    system = screening.system_prompt(CODEBOOK, persona="clinical neurologist")
    assert "Your perspective: clinical neurologist" in system and "I4 Setting matches the instrument" in system
    ann = annotation.system_prompt(CODEBOOK)
    for dim in CODEBOOK.dimensions.values():
        for label in dim.label_ids:
            assert f"- {label}:" in ann


def test_render_article_marks_missing_abstract():
    assert "(no abstract available)" in render_article(NO_ABSTRACT)
    assert render_article(INCLUDE_ART).startswith("ARTICLE\nPMID: 1\nTitle: Accuracy")


def test_check_quotes_flags_invented_evidence():
    ok = EvidenceSpan(quote="We tested the  MMSE against DSM-IV")  # spacing/case differences are fine
    bad = EvidenceSpan(quote="The MoCA outperformed the MMSE")
    assert check_quotes([ok], INCLUDE_ART) == []
    assert len(check_quotes([bad], INCLUDE_ART)) == 1


# ---- agents post-check model output ------------------------------------------


def test_screening_drops_unknown_rule_ids_and_flags_bad_quotes():
    def responder(*_):
        return ScreeningAnswer(
            decision="include",
            rule_ids=["I1", "X9"],
            evidence=[EvidenceSpan(quote="made up sentence")],
            rationale="r",
            confidence=0.9,
        )

    vote, _ = screening.run(INCLUDE_ART, CODEBOOK, FakeLLM(FAKE_CFG, {ScreeningAnswer: responder}))
    assert vote.rule_ids == ["I1"] and vote.labels == ["include"]
    assert any("X9" in w for w in vote.warnings) and any("not found" in w for w in vote.warnings)


def test_annotation_keeps_only_codebook_labels():
    def responder(*_):
        return AnnotationAnswer(
            annotations=[DimensionAnnotation(dimension="instrument", labels=["MMSE", "SLUMS", "MMSE"])],
            confidence=0.8,
        )

    votes, _ = annotation.run(INCLUDE_ART, CODEBOOK, FakeLLM(FAKE_CFG, {AnnotationAnswer: responder}))
    by = {v.field: v for v in votes}
    assert by["instrument"].labels == ["MMSE"]
    assert any("SLUMS" in w for w in by["instrument"].warnings)
    assert by["setting"].labels == [] and "no entry" in by["setting"].warnings[0]


# ---- LangChain wrapper: structured output, retries, cache, cost -------------


class StubChat:
    """Stands in for a LangChain chat model: returns queued outputs, counts calls."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def with_structured_output(self, schema, include_raw=False):
        assert include_raw
        stub = self

        class Runnable:
            def invoke(self, messages):
                stub.calls += 1
                return stub.outputs.pop(0)

        return Runnable()


def good_output():
    parsed = ScreeningAnswer(decision="exclude", rule_ids=["I1"], evidence=[], rationale="r", confidence=0.7)
    raw = AIMessage(content="", usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200})
    return {"raw": raw, "parsed": parsed, "parsing_error": None}


def test_langchain_llm_records_usage_and_cost(tmp_path):
    cfg = ModelConfig(provider="google", model="m", price_in_per_mtok=1.0, price_out_per_mtok=4.0)
    llm = LangChainLLM(cfg, chat_model=StubChat([good_output()]), cache=ResponseCache(tmp_path))
    res = llm.structured(ScreeningAnswer, "sys", "user", role="screening")
    assert res.parsed.decision == "exclude"
    assert (res.usage.input_tokens, res.usage.output_tokens) == (1000, 200)
    assert res.usage.cost_usd == pytest.approx((1000 * 1 + 200 * 4) / 1e6)
    assert res.usage.model == "google:m" and not res.usage.cached


def test_identical_call_is_served_from_cache(tmp_path):
    cfg = ModelConfig(provider="google", model="m")
    chat = StubChat([good_output()])
    llm = LangChainLLM(cfg, chat_model=chat, cache=ResponseCache(tmp_path))
    llm.structured(ScreeningAnswer, "sys", "user", role="screening")
    again = llm.structured(ScreeningAnswer, "sys", "user", role="screening")
    assert chat.calls == 1 and again.usage.cached and again.parsed.decision == "exclude"
    # A different prompt is a different request
    chat.outputs.append(good_output())
    llm.structured(ScreeningAnswer, "sys", "other user", role="screening")
    assert chat.calls == 2


def test_parse_failure_is_retried_then_gives_up(tmp_path):
    bad = {"raw": AIMessage(content="oops"), "parsed": None, "parsing_error": "invalid json"}
    cfg = ModelConfig(provider="google", model="m")
    llm = LangChainLLM(cfg, chat_model=StubChat([bad, good_output()]), cache=ResponseCache(None))
    assert llm.structured(ScreeningAnswer, "s", "u", role="screening").parsed.decision == "exclude"
    llm = LangChainLLM(cfg, chat_model=StubChat([bad, bad, bad]), cache=ResponseCache(None))
    with pytest.raises(LLMError, match="invalid json"):
        llm.structured(ScreeningAnswer, "s", "u", role="screening")


# ---- the E1 graph ------------------------------------------------------------


def pipeline():
    llm = fake_llm()
    return build_pipeline(CODEBOOK, llm, llm, [Reviewer(id="A", llm=llm, annotation_llm=llm)])


def test_included_article_goes_through_all_four_steps():
    state = pipeline().invoke({"article": INCLUDE_ART})
    assert state["triage"].enough_information
    assert state["screening_decision"].labels == ["include"]
    assert state["screening_decision"].method == "single"
    assert "MMSE" in [f.value for f in state["extraction"].instruments]
    assert state["annotation_decisions"]["instrument"].labels == ["MMSE"]
    assert state["annotation_decisions"]["setting"].labels == ["primary_care"]
    assert sorted(u.role for u in state["usage"]) == ["annotation", "extraction", "screening", "triage"]


def test_excluded_article_stops_after_screening():
    state = pipeline().invoke({"article": EXCLUDE_ART})
    assert state["screening_decision"].labels == ["exclude"]
    assert "extraction" not in state and "annotation_decisions" not in state
    assert sorted(u.role for u in state["usage"]) == ["screening", "triage"]


def test_panel_needs_unique_reviewers():
    llm = fake_llm()
    with pytest.raises(ValueError, match="At least one reviewer"):
        build_pipeline(CODEBOOK, llm, llm, [])
    with pytest.raises(ValueError, match="unique"):
        build_pipeline(CODEBOOK, llm, llm, [Reviewer("A", llm, llm), Reviewer("A", llm, llm)])


def test_runner_collects_states_failures_and_predictions():
    graph = pipeline()

    class Boom:
        def invoke(self, state, config=None):
            if state["article"].pmid == "2":
                raise RuntimeError("provider down")
            return graph.invoke(state, config)

    states, failures, paused = run_articles(Boom(), [INCLUDE_ART, EXCLUDE_ART, NO_ABSTRACT], concurrency=2)
    assert paused == []
    assert [s["article"].pmid for s in states] == ["1", "3"]
    assert failures[0]["pmid"] == "2" and "provider down" in failures[0]["error"]
    pred = to_prediction(states[0])
    assert pred.screening == "include" and pred.labels["instrument"] == ["MMSE"] and pred.score == 0.7
    assert to_prediction(states[1]).screening == "exclude"


def test_run_configs_parse():
    for name in ("E1_fake", "E1_gemini"):
        cfg = load_run_config(ROOT / f"configs/runs/{name}.yaml")
        assert set(cfg.roles) == {"triage", "screening", "extraction", "annotation"}
        assert [r.id for r in cfg.reviewer_configs()] == ["A"]  # E1 = panel of one
    assert load_run_config(ROOT / "configs/runs/E1_gemini.yaml").roles["screening"].model.provider == "google"
    for name in ("E2_fake", "E2_gemini"):
        cfg = load_run_config(ROOT / f"configs/runs/{name}.yaml")
        reviewers = cfg.reviewer_configs()
        assert [r.id for r in reviewers] == ["A", "B", "C"] and all(r.persona for r in reviewers)
    gem = load_run_config(ROOT / "configs/runs/E2_gemini.yaml").reviewer_configs()
    assert len({r.model.model for r in gem}) == 3  # three different Gemini versions


def test_repeats_do_not_share_cached_answers(tmp_path):
    cfg = ModelConfig(provider="google", model="m")
    chat = StubChat([good_output(), good_output(), good_output()])
    r0 = LangChainLLM(cfg, chat_model=chat, cache=ResponseCache(tmp_path))
    r1 = LangChainLLM(cfg, chat_model=chat, cache=ResponseCache(tmp_path), namespace="repeat-1")
    r0.structured(ScreeningAnswer, "s", "u", role="screening")
    r0.structured(ScreeningAnswer, "s", "u", role="screening")  # cached
    r1.structured(ScreeningAnswer, "s", "u", role="screening")  # different repeat: calls the model
    assert chat.calls == 2
