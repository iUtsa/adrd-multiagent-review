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
    ExclusionCheck,
    InclusionCheck,
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


REAL_QUOTE = "We tested the MMSE against DSM-IV dementia diagnosis in 500 primary care patients aged 65+."


def screen(inclusion, exclusion=()):
    """Run the screening agent with a scripted set of rule judgments."""
    answer = ScreeningAnswer(
        inclusion=[InclusionCheck(rule_id=r, status=s, quote=q) for r, s, q in inclusion],
        exclusion=[ExclusionCheck(rule_id=r, status=s, quote=q) for r, s, q in exclusion],
        rationale="r",
        confidence=0.9,
    )
    vote, _ = screening.run(INCLUDE_ART, CODEBOOK, FakeLLM(FAKE_CFG, {ScreeningAnswer: lambda *_: answer}))
    return vote


def test_unclear_never_excludes():
    # The failure seen in the first real-model run: "not reported" must not become "exclude".
    vote = screen([("I1", "met", REAL_QUOTE), ("I2", "unclear", None), ("I3", "unclear", None), ("I4", "unclear", None)])
    assert vote.labels == ["include"] and vote.rule_ids == ["I1"]
    assert vote.rule_checks["I2"] == "unclear" and vote.confidence == 0.625  # 0.5 + 0.5 * 1/4


def test_clear_violation_with_a_real_quote_excludes():
    vote = screen([("I1", "not_met", REAL_QUOTE), ("I2", "met", REAL_QUOTE)])
    assert vote.labels == ["exclude"] and vote.rule_ids == ["I1"]
    vote = screen([("I1", "met", REAL_QUOTE)], [("E1", "applies", REAL_QUOTE)])
    assert vote.labels == ["exclude"] and vote.rule_ids == ["E1"]


def test_exclusion_without_a_real_quote_is_treated_as_unclear():
    vote = screen([("I1", "not_met", "a sentence that is not in the article")])
    assert vote.labels == ["include"] and vote.rule_checks["I1"] == "unclear"
    assert any("not_met without a real quote" in w for w in vote.warnings)
    vote = screen([("I1", "met", REAL_QUOTE)], [("E1", "applies", None)])
    assert vote.labels == ["include"] and vote.rule_checks["E1"] == "unclear"


def test_missing_and_unknown_rules_are_handled():
    vote = screen([("I1", "met", REAL_QUOTE), ("X9", "not_met", REAL_QUOTE)])
    assert vote.labels == ["include"]  # X9 is not a codebook rule; I2-I4 were not judged -> unclear
    assert any("X9" in w for w in vote.warnings) and any("I2: no judgment given" in w for w in vote.warnings)


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

    def with_structured_output(self, schema, include_raw=False, **kw):
        assert include_raw
        stub = self

        class Runnable:
            def invoke(self, messages):
                stub.calls += 1
                return stub.outputs.pop(0)

        return Runnable()


def good_output():
    parsed = ScreeningAnswer(inclusion=[InclusionCheck(rule_id="I1", status="not_met", quote="q")], rationale="r", confidence=0.7)
    raw = AIMessage(content="", usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200})
    return {"raw": raw, "parsed": parsed, "parsing_error": None}


def test_langchain_llm_records_usage_and_cost(tmp_path):
    cfg = ModelConfig(provider="google", model="m", price_in_per_mtok=1.0, price_out_per_mtok=4.0)
    llm = LangChainLLM(cfg, chat_model=StubChat([good_output()]), cache=ResponseCache(tmp_path))
    res = llm.structured(ScreeningAnswer, "sys", "user", role="screening")
    assert res.parsed.inclusion[0].status == "not_met"
    assert (res.usage.input_tokens, res.usage.output_tokens) == (1000, 200)
    assert res.usage.cost_usd == pytest.approx((1000 * 1 + 200 * 4) / 1e6)
    assert res.usage.model == "google:m" and not res.usage.cached


def test_identical_call_is_served_from_cache(tmp_path):
    cfg = ModelConfig(provider="google", model="m")
    chat = StubChat([good_output()])
    llm = LangChainLLM(cfg, chat_model=chat, cache=ResponseCache(tmp_path))
    llm.structured(ScreeningAnswer, "sys", "user", role="screening")
    again = llm.structured(ScreeningAnswer, "sys", "user", role="screening")
    assert chat.calls == 1 and again.usage.cached and again.parsed.inclusion[0].status == "not_met"
    # A different prompt is a different request
    chat.outputs.append(good_output())
    llm.structured(ScreeningAnswer, "sys", "other user", role="screening")
    assert chat.calls == 2


def test_parse_failure_is_retried_then_gives_up(tmp_path):
    bad = {"raw": AIMessage(content="oops"), "parsed": None, "parsing_error": "invalid json"}
    cfg = ModelConfig(provider="google", model="m")
    llm = LangChainLLM(cfg, chat_model=StubChat([bad, good_output()]), cache=ResponseCache(None))
    assert llm.structured(ScreeningAnswer, "s", "u", role="screening").parsed.inclusion[0].status == "not_met"
    llm = LangChainLLM(cfg, chat_model=StubChat([bad, bad, bad]), cache=ResponseCache(None), max_attempts=3)
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
        def stream(self, state, config=None, stream_mode="values"):
            if state["article"].pmid == "2":
                raise RuntimeError("provider down")
            return graph.stream(state, config, stream_mode=stream_mode)

    states, failures, paused = run_articles(Boom(), [INCLUDE_ART, EXCLUDE_ART, NO_ABSTRACT], concurrency=2)
    assert paused == []
    assert [s["article"].pmid for s in states] == ["1", "3"]
    assert failures[0]["pmid"] == "2" and "provider down" in failures[0]["error"]
    pred = to_prediction(states[0])
    assert pred.screening == "include" and pred.labels["instrument"] == ["MMSE"] and pred.score == 0.75
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


def test_triage_enough_information_is_computed_not_trusted():
    from adrd_review.agents import triage
    from adrd_review.schemas import TriageReport

    def says(**kw):
        report, _ = triage.run(INCLUDE_ART, CODEBOOK, FakeLLM(FAKE_CFG, {TriageReport: lambda *_: TriageReport(**kw)}))
        return report

    # off-topic paper: fully judgeable, nothing missing, even if the model also listed rules
    r = says(clearly_ineligible=True, missing=["I2"], enough_information=False)
    assert r.enough_information and r.missing == []
    # thin abstract: some rules cannot be judged -> not enough (this is what triggers full text)
    r = says(clearly_ineligible=False, missing=["I2", "ZZ"], enough_information=True)
    assert not r.enough_information and r.missing == ["I2"]
    # nothing missing -> enough, whatever the model put in enough_information
    assert says(clearly_ineligible=False, missing=[], enough_information=False).enough_information


def test_invalid_answer_switches_request_format(tmp_path):
    """A rejected (non-rate-limit) answer is retried with a different structured-output method."""
    used = []

    class Chat:
        def with_structured_output(self, schema, include_raw=False, **kw):
            method = kw.get("method")

            class R:
                def invoke(self, messages):
                    used.append(method)
                    if method == "function_calling":
                        raise RuntimeError("Error code: 400 - tool call validation failed")
                    return good_output()

            return R()

    cfg = ModelConfig(provider="groq", model="m", structured_method="function_calling")
    llm = LangChainLLM(cfg, chat_model=Chat(), cache=ResponseCache(None), max_attempts=3)
    import adrd_review.llm as llm_mod

    orig = llm_mod.time.sleep
    llm_mod.time.sleep = lambda s: None
    try:
        res = llm.structured(ScreeningAnswer, "s", "u", role="screening")
    finally:
        llm_mod.time.sleep = orig
    assert used == ["function_calling", None] and res.parsed.inclusion[0].status == "not_met"


def test_roles_sharing_a_model_share_one_limiter_and_token_budget():
    import adrd_review.llm as m

    m._LIMITERS.clear()
    a = m.shared_limiter(ModelConfig(provider="groq", model="x", requests_per_minute=3, tokens_per_minute=8000, max_tokens=1500))
    b = m.shared_limiter(ModelConfig(provider="groq", model="x", requests_per_minute=3, tokens_per_minute=8000, max_tokens=2048))
    c = m.shared_limiter(ModelConfig(provider="groq", model="y", requests_per_minute=3))
    assert a is b and a is not c  # same model -> same limiter, whatever the role's settings

    lim = m.RateLimiter(tokens_per_minute=8000)
    slept = []
    real_sleep, real_clock = m.time.sleep, m.time.monotonic
    clock = [1000.0]
    m.time.monotonic = lambda: clock[0]
    m.time.sleep = lambda sec: (slept.append(round(sec, 1)), clock.__setitem__(0, clock[0] + sec))
    try:
        for _ in range(3):
            lim.wait(2500)  # 7,500 tokens: fits
        assert sum(slept) == 0
        lim.wait(2500)  # would be 10,000 in the same minute: must wait for the window
        assert sum(slept) >= 60
    finally:
        m.time.sleep, m.time.monotonic = real_sleep, real_clock


def test_retry_after_is_read_from_the_provider_message():
    from adrd_review.llm import retry_after_seconds

    assert retry_after_seconds(RuntimeError("Rate limit reached... Please try again in 7.66s. Visit")) == pytest.approx(7.66)
    assert retry_after_seconds(RuntimeError("Please try again in 1m12.5s")) == pytest.approx(72.5)
    assert retry_after_seconds(RuntimeError("Please try again in 2h3m50.4s")) == pytest.approx(7430.4)
    assert retry_after_seconds(RuntimeError("something else")) is None
    # OpenRouter's daily limit gives a reset time (ms since 1970) instead of a wait
    import time
    reset_ms = int((time.time() + 5 * 3600) * 1000)
    err = RuntimeError("Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-day.', "
                       f"'metadata': {{'headers': {{'X-RateLimit-Limit': '50', 'X-RateLimit-Reset': '{reset_ms}'}}}}}}}}")
    assert retry_after_seconds(err) == pytest.approx(5 * 3600, abs=5)


def test_overflow_host_is_used_only_after_the_daily_limit_and_never_judges_twice(tmp_path):
    import time

    from adrd_review.llm import EXHAUSTED, OUT_OF_CREDIT, LimitReached, OverflowLLM

    class FailingChat(StubChat):
        def __init__(self, error):
            super().__init__([])
            self.error = error

        def with_structured_output(self, schema, include_raw=False, **kw):
            stub = self

            class Runnable:
                def invoke(self, messages):
                    stub.calls += 1
                    raise stub.error

            return Runnable()

    cache = ResponseCache(tmp_path)
    groq = ModelConfig(provider="groq", model="m")
    router = ModelConfig(provider="openrouter", model="m")
    EXHAUSTED[groq.label] = time.time() + 3600  # Groq's daily limit is used up
    try:
        backup_chat = StubChat([good_output()])
        llm = OverflowLLM(LangChainLLM(groq, chat_model=StubChat([]), cache=cache),
                          LangChainLLM(router, chat_model=backup_chat, cache=cache))
        res = llm.structured(ScreeningAnswer, "s", "u", role="screening")
        assert res.usage.model == "openrouter:m" and backup_chat.calls == 1
        # next try, Groq has quota again: the answer already given by the overflow host is reused
        del EXHAUSTED[groq.label]
        groq_chat = StubChat([good_output()])
        again = OverflowLLM(LangChainLLM(groq, chat_model=groq_chat, cache=cache),
                            LangChainLLM(router, chat_model=StubChat([]), cache=cache))
        assert again.structured(ScreeningAnswer, "s", "u", role="screening").usage.cached and groq_chat.calls == 0

        # overflow host without credit: the call is deferred (LimitReached), the run is not stopped
        EXHAUSTED[groq.label] = time.time() + 3600
        broke = LangChainLLM(router, chat_model=FailingChat(RuntimeError("Error code: 402 - insufficient_quota")),
                             cache=cache, max_attempts=1)
        broke.stops_run_without_credit = False
        with pytest.raises(LimitReached):
            OverflowLLM(LangChainLLM(groq, chat_model=StubChat([]), cache=cache), broke).structured(
                ScreeningAnswer, "s", "other", role="screening")
        assert not OUT_OF_CREDIT.is_set() and router.label in EXHAUSTED
    finally:
        EXHAUSTED.pop(groq.label, None)
        EXHAUSTED.pop(router.label, None)
        OUT_OF_CREDIT.clear()


def test_overflow_is_built_from_the_config_and_not_part_of_the_cache_key():
    from adrd_review.llm import OverflowLLM, build_llm

    plain = ModelConfig(provider="groq", model="m")
    with_overflow = plain.model_copy(update={"overflow": ModelConfig(provider="openrouter", model="m")})
    assert ResponseCache.key(plain, ScreeningAnswer, "s", "u") == ResponseCache.key(with_overflow, ScreeningAnswer, "s", "u")
    llm = build_llm(with_overflow)
    assert isinstance(llm, OverflowLLM)
    assert llm.primary.max_rate_wait == OverflowLLM.OVERFLOW_AFTER_SECONDS  # no long waits when an overflow host exists
    assert build_llm(plain).max_rate_wait is None


def test_a_busy_per_minute_budget_sends_the_call_to_the_overflow_host(tmp_path):
    from adrd_review.llm import OverflowLLM

    cache = ResponseCache(tmp_path)
    groq = ModelConfig(provider="groq", model="busy-m", tokens_per_minute=8000)
    router = ModelConfig(provider="openrouter", model="busy-m")
    groq_chat, router_chat = StubChat([good_output()]), StubChat([good_output()])
    llm = OverflowLLM(LangChainLLM(groq, chat_model=groq_chat, cache=cache),
                      LangChainLLM(router, chat_model=router_chat, cache=cache))
    assert llm.structured(ScreeningAnswer, "s", "first", role="screening").usage.model == "groq:busy-m"
    # the first call used Groq's minute budget: the next one goes to the overflow host instead of waiting
    assert llm.structured(ScreeningAnswer, "s" * 30000, "second", role="screening").usage.model == "openrouter:busy-m"
    assert (groq_chat.calls, router_chat.calls) == (1, 1)
