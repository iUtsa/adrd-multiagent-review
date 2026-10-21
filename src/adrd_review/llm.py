"""Model factory + response cache. Every agent calls models through `LLM.structured`.

    llm = build_llm(ModelConfig(provider="google", model="gemini-..."), cache_dir="data/cache/llm")
    result = llm.structured(ScreeningAnswer, system_prompt, user_prompt, role="screening")
    result.parsed   # a validated ScreeningAnswer
    result.usage    # tokens, cost, seconds, cached?

Providers: google (AI Studio key or Vertex AI), anthropic, openai, openrouter, groq,
ollama (all through LangChain), and
`fake`, a keyword-rule stand-in used for tests and for running the pipeline
without an API key. Fake answers are plumbing checks, never results.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar

import yaml
from pydantic import BaseModel

from adrd_review.schemas import Usage

T = TypeVar("T", bound=BaseModel)

Provider = Literal["google", "anthropic", "openai", "openrouter", "groq", "ollama", "fake", "nvidia", "mistral"]


class ModelConfig(BaseModel):
    provider: Provider
    model: str
    # None = do not send it (newer Claude and OpenAI models reject a temperature setting)
    temperature: float | None = 0.0
    max_tokens: int | None = 2048
    requests_per_minute: float | None = None  # stay under free-tier limits
    tokens_per_minute: int | None = None  # provider limit, if any (preflight checks calls fit)
    requests_per_day: int | None = None  # provider limit, if any
    tokens_per_day: int | None = None  # provider limit, if any (Groq free tier: 200,000; not shown in its headers)
    price_in_per_mtok: float | None = None  # USD per million input tokens (optional)
    price_out_per_mtok: float | None = None
    # How LangChain asks for structured output: None = provider default, or
    # "function_calling" / "json_schema" / "json_mode" (e.g. for OpenAI-compatible hosts).
    structured_method: str | None = None
    # Anthropic prompt caching: the instructions and answer form, identical for every article a reviewer
    # sees, are cached by the provider and billed at a fraction of the input price on repeat calls.
    prompt_cache: bool = False
    # Extra constructor options passed straight to the LangChain model, e.g. for Gemini
    # on Vertex AI: {vertexai: true, project: my-gcp-project, location: us-central1},
    # or {thinking_budget: 0} to turn off Gemini's thinking step.
    extra: dict[str, Any] = {}
    # The same model at another host, called only once this one's daily limit is used up (e.g. Groq's free
    # tier -> OpenRouter, paid per token). An answer already cached from either host is reused, so a request is
    # never judged twice. Not part of the cache key.
    overflow: ModelConfig | None = None

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class LLMResult(Generic[T]):
    parsed: T
    usage: Usage


class LLMError(RuntimeError):
    pass


class OutOfCredit(LLMError):
    """The provider account has no credit left. Retrying cannot help, so the whole run stops."""


# Set when any call finds the account out of credit; the runner then stops starting new articles.
OUT_OF_CREDIT = threading.Event()
# Models whose daily (or other long) limit is used up: label -> time it resets. Calls to them fail at
# once instead of each waiting and retrying, so a free-tier run moves on and is simply resumed later.
EXHAUSTED: dict[str, float] = {}


class LimitReached(LLMError):
    """A model's daily limit is used up; calls fail fast until it resets (re-run then: the cache resumes)."""
_CREDIT_SIGNS = ("credit balance is too low", "insufficient_quota", "credit_balance_exhausted", "no credits remaining")


def is_out_of_credit(error: BaseException) -> bool:
    text = str(error).lower()
    return any(s in text for s in _CREDIT_SIGNS)


_PRICES: dict | None = None


def listed_price(label: str) -> dict | None:
    """USD per million tokens from configs/prices.yaml, when the model config gives none."""
    global _PRICES
    if _PRICES is None:
        path = Path("configs/prices.yaml")
        _PRICES = yaml.safe_load(path.read_text()).get("models", {}) if path.exists() else {}
    return _PRICES.get(label)


class LLM(Protocol):
    config: ModelConfig

    def structured(self, schema: type[T], system: str, user: str, role: str) -> LLMResult[T]: ...


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


class ResponseCache:
    """One JSON file per request, keyed by a hash of everything that affects the answer."""

    def __init__(self, directory: str | Path | None):
        self.dir = Path(directory) if directory else None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(config: ModelConfig, schema: type[BaseModel], system: str, user: str, namespace: str = "") -> str:
        payload = {
            "namespace": namespace,  # e.g. 'repeat-2': repeats never share answers
            "provider": config.provider,
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "structured_method": config.structured_method,
            "extra": config.extra,
            "schema": schema.model_json_schema(),
            "system": system,
            "user": user,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        if not self.dir:
            return None
        path = self.dir / key[:2] / f"{key}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, key: str, value: dict) -> None:
        if not self.dir:
            return
        path = self.dir / key[:2] / f"{key}.json"
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value))
        tmp.replace(path)


class RateLimiter:
    """Thread-safe limiter for one model: spacing between calls AND a tokens-per-minute budget.

    Providers such as Groq limit tokens per minute, not just requests. Before each call the
    caller says roughly how many tokens it will use; the limiter waits until that fits in the
    last 60 seconds' budget. After the call the real count replaces the estimate.
    """

    def __init__(self, per_minute: float | None = None, tokens_per_minute: int | None = None):
        self.interval = 60.0 / per_minute if per_minute else 0.0
        self.tpm = tokens_per_minute
        self._lock = threading.Lock()
        self._next = 0.0
        self._used: list[tuple[float, float]] = []  # (time, tokens) in the last minute

    def tighten(self, per_minute: float | None, tokens_per_minute: int | None) -> None:
        """Several roles can share a model: keep the strictest limits any of them configured."""
        if per_minute:
            self.interval = max(self.interval, 60.0 / per_minute)
        if tokens_per_minute:
            self.tpm = min(self.tpm, tokens_per_minute) if self.tpm else tokens_per_minute

    def wait(self, tokens: float = 0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._used = [(t, n) for t, n in self._used if now - t < 60]
                spent = sum(n for _, n in self._used)
                if self.tpm and self._used and spent + tokens > self.tpm:
                    pause = 60 - (now - self._used[0][0]) + 0.5  # until the oldest call leaves the window
                else:
                    start = max(now, self._next)
                    self._next = start + self.interval
                    self._used.append((start, tokens))
                    pause = start - now
                    break
            time.sleep(max(0.1, pause))
        time.sleep(max(0.0, pause))

    def expected_wait(self, tokens: float = 0) -> float:
        """Seconds wait() would pause for a call of this size now (nothing is reserved)."""
        with self._lock:
            now = time.monotonic()
            used = [(t, n) for t, n in self._used if now - t < 60]
            if self.tpm and used and sum(n for _, n in used) + tokens > self.tpm:
                return 60 - (now - used[0][0])
            return max(0.0, self._next - now)

    def record(self, estimated: float, actual: float) -> None:
        """Correct the budget once the real token count is known."""
        with self._lock:
            self._used.append((time.monotonic(), actual - estimated))


_LIMITERS: dict[str, RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def shared_limiter(config: "ModelConfig") -> RateLimiter:
    """One limiter per provider+model, shared by every role that uses that model."""
    with _LIMITERS_LOCK:
        lim = _LIMITERS.get(config.label)
        if lim is None:
            lim = _LIMITERS[config.label] = RateLimiter(config.requests_per_minute, config.tokens_per_minute)
        else:
            lim.tighten(config.requests_per_minute, config.tokens_per_minute)
        return lim


def retry_after_seconds(error: Exception) -> float | None:
    """Seconds a provider asked us to wait, if it said: 'Please try again in 1m12.5s' (Groq), or a reset time
    'X-RateLimit-Reset': '1792368000000' in ms since 1970 (OpenRouter's free-models-per-day limit)."""
    import re

    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", str(error))
    if m:
        return int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3))
    m = re.search(r"X-RateLimit-Reset['\"]?\s*:\s*['\"]?(\d{10,13})", str(error), re.IGNORECASE)
    if m:
        reset = int(m.group(1)) / (1000 if len(m.group(1)) == 13 else 1)
        return max(0.0, reset - time.time())
    return None


# --------------------------------------------------------------------------- #
# Real models through LangChain
# --------------------------------------------------------------------------- #


def make_chat_model(config: ModelConfig):
    """Build the LangChain chat model for a config (imports only what is needed)."""
    p = config.provider
    kw = {"model": config.model, **config.extra}
    if config.temperature is not None:
        kw["temperature"] = config.temperature
    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**kw, max_tokens=config.max_tokens or 2048)
    if p == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(**kw, max_output_tokens=config.max_tokens)
    if p in ("openai", "openrouter", "groq", "nvidia", "mistral"):
        from langchain_openai import ChatOpenAI

        hosts = {
            "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
            "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
            "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
            "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
        }
        if p in hosts:
            url, env = hosts[p]
            kw = {"base_url": url, "api_key": os.getenv(env), **kw}
        # No hidden retries inside the client (our loop retries, switches host, or defers): with the client's
        # default 2 retries of up to 10 min each, one hung request blocked a run for ~30 min (2026-10-20).
        return ChatOpenAI(**{"max_retries": 0, **kw}, max_tokens=config.max_tokens)
    if p == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(**kw, num_predict=config.max_tokens)
    raise ValueError(f"Provider {p!r} has no LangChain model")


class LangChainLLM:
    def __init__(
        self,
        config: ModelConfig,
        chat_model=None,
        cache: ResponseCache | None = None,
        max_attempts: int = 5,
        namespace: str = "",
    ):
        self.stops_run_without_credit = True  # False for an overflow host: no credit there only defers its calls
        self.max_rate_wait: float | None = None  # longest rate-limit wait before giving up (None: env, default 600 s)
        self.namespace = namespace
        self.config = config
        self.chat = chat_model if chat_model is not None else make_chat_model(config)
        self.cache = cache or ResponseCache(None)
        self.max_attempts = max_attempts
        self.limiter = shared_limiter(config)

    def _cost(self, tokens_in: int, tokens_out: int, cache_read: int = 0, cache_write: int = 0) -> float | None:
        """`tokens_in` includes cached tokens (LangChain's count); those are re-priced at the cache rates."""
        c = self.config
        if c.price_in_per_mtok is not None and c.price_out_per_mtok is not None:
            p = {"input": c.price_in_per_mtok, "output": c.price_out_per_mtok}
        else:
            p = listed_price(c.label)
            if p is None:
                return None
        fresh = max(0, tokens_in - cache_read - cache_write)
        read = p.get("cache_read", 0.1 * p["input"])
        write = p.get("cache_write", 1.25 * p["input"])
        return (fresh * p["input"] + cache_read * read + cache_write * write + tokens_out * p["output"]) / 1e6

    def estimate(self, system: str, user: str) -> float:
        return (len(system) + len(user)) / 4 + (self.config.max_tokens or 1000) / 2

    def structured(self, schema: type[T], system: str, user: str, role: str) -> LLMResult[T]:
        key = ResponseCache.key(self.config, schema, system, user, self.namespace)
        hit = self.cache.get(key)
        if hit:
            usage = Usage(**hit["usage"])
            usage.role, usage.cached, usage.seconds = role, True, 0.0
            return LLMResult(schema.model_validate(hit["parsed"]), usage)

        from langchain_core.messages import HumanMessage, SystemMessage

        # Ways to ask for structured output, in order of preference. If the model's answer is
        # rejected as invalid (not a rate limit), the next attempt uses the next way: repeating
        # the identical request at temperature 0 tends to repeat the identical mistake.
        methods = list(dict.fromkeys([self.config.structured_method, None, "function_calling", "json_schema"]))
        which = 0

        def runnable_for(i: int):
            m = methods[i % len(methods)]
            return self.chat.with_structured_output(schema, include_raw=True, **({"method": m} if m else {}))

        if self.config.provider == "anthropic" and self.config.prompt_cache:
            sys_msg = SystemMessage(content=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}])
        else:
            sys_msg = SystemMessage(content=system)
        messages = [sys_msg, HumanMessage(content=user)]
        # Rough size of this call, for the tokens-per-minute budget (corrected after the call).
        estimate = self.estimate(system, user)
        last_error: Any = None
        until = EXHAUSTED.get(self.config.label)
        if until and time.time() < until:
            raise LimitReached(f"{self.config.label}: daily limit used up; resets in ~{(until - time.time()) / 3600:.1f} h")
        for attempt in range(1, self.max_attempts + 1):
            self.limiter.wait(estimate)
            start = time.monotonic()
            try:
                out = runnable_for(which).invoke(messages)
            except Exception as e:  # network, rate limit, provider error, invalid tool call
                last_error = e
                if is_out_of_credit(e):
                    if self.stops_run_without_credit:
                        OUT_OF_CREDIT.set()
                    raise OutOfCredit(f"{self.config.label}: the account is out of credit ({e})") from e
                if attempt == self.max_attempts:
                    break
                # Per-minute token limits (e.g. Groq free tier: 8,000 tokens/min) need a real pause.
                rate_limited = "429" in str(e) or "rate limit" in str(e).lower() or "quota" in str(e).lower()
                if rate_limited:
                    asked = retry_after_seconds(e)
                    # A long wait means a daily limit. By default give up on this call (the run can be
                    # resumed later from the cache). ADRD_MAX_RATE_WAIT_SECONDS raises the patience for
                    # unattended runs that should simply wait for the provider's budget to refill.
                    patience = self.max_rate_wait if self.max_rate_wait is not None else float(
                        os.getenv("ADRD_MAX_RATE_WAIT_SECONDS", "600"))
                    if asked is not None and asked > patience:
                        EXHAUSTED[self.config.label] = time.time() + asked
                        raise LimitReached(f"{self.config.label}: limit reached; resets in ~{asked / 3600:.1f} h ({e})") from e
                    time.sleep(asked + 1 if asked is not None else 15 * attempt)
                else:
                    which += 1
                    time.sleep(min(30, 2**attempt))
                continue
            seconds = time.monotonic() - start
            parsed = out.get("parsed")
            if parsed is None:
                last_error = out.get("parsing_error") or "model returned no structured output"
                which += 1
                continue
            meta = getattr(out.get("raw"), "usage_metadata", None) or {}
            tin, tout = int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
            details = meta.get("input_token_details") or {}
            cache_read, cache_write = int(details.get("cache_read") or 0), int(details.get("cache_creation") or 0)
            if tin or tout:
                self.limiter.record(estimate, tin + tout)
            usage = Usage(
                role=role,
                model=self.config.label,
                input_tokens=tin,
                output_tokens=tout,
                cost_usd=self._cost(tin, tout, cache_read, cache_write),
                prompt_cache_read_tokens=cache_read,
                seconds=round(seconds, 3),
            )
            self.cache.put(key, {"parsed": parsed.model_dump(), "usage": usage.model_dump()})
            return LLMResult(parsed, usage)
        raise LLMError(f"{self.config.label} failed after {self.max_attempts} attempts: {last_error}")


# --------------------------------------------------------------------------- #
# Fake model (no key needed)
# --------------------------------------------------------------------------- #

Responder = Callable[[type[BaseModel], str, str], BaseModel]


class FakeLLM:
    """Answers with a function per output schema. Used for tests and dry runs."""

    def __init__(self, config: ModelConfig, responders: dict[type[BaseModel], Responder]):
        self.config = config
        self.responders = responders

    def structured(self, schema: type[T], system: str, user: str, role: str) -> LLMResult[T]:
        if schema not in self.responders:
            raise LLMError(f"FakeLLM has no responder for {schema.__name__}")
        parsed = schema.model_validate(self.responders[schema](schema, system, user).model_dump())
        tokens_in = (len(system) + len(user)) // 4
        return LLMResult(parsed, Usage(role=role, model=self.config.label, input_tokens=tokens_in, output_tokens=50))


def _from_cache(llm, schema, system: str, user: str, role: str) -> LLMResult | None:
    cache = getattr(llm, "cache", None)
    if cache is None:
        return None
    hit = cache.get(ResponseCache.key(llm.config, schema, system, user, getattr(llm, "namespace", "")))
    if hit is None:
        return None
    usage = Usage.model_validate(hit["usage"])
    usage.role, usage.cached, usage.seconds = role, True, 0.0
    return LLMResult(schema.model_validate(hit["parsed"]), usage)


class OverflowLLM:
    """A model with a second host for the same model (ModelConfig.overflow): the second is called only when the
    first fails (its limit is used up, or it keeps asking to wait). Cached answers from either are reused first, so nothing is judged twice. If the
    second host fails too (no credit, its own limit), the call is deferred like any used-up limit."""

    OVERFLOW_AFTER_SECONDS = 20  # a longer rate-limit wait at the first host goes to the overflow host instead

    def __init__(self, primary, backup):
        self.primary, self.backup, self.config = primary, backup, primary.config
        if hasattr(primary, "max_rate_wait"):
            primary.max_rate_wait = self.OVERFLOW_AFTER_SECONDS

    def structured(self, schema: type[T], system: str, user: str, role: str) -> LLMResult[T]:
        for llm in (self.primary, self.backup):
            if (hit := _from_cache(llm, schema, system, user, role)) is not None:
                return hit
        # The first host's per-minute budget is busy (e.g. Groq's free 8,000 tokens/min fits about one abstract
        # batch): rather than queue, this call goes to the overflow host, so both hosts work side by side.
        limiter = getattr(self.primary, "limiter", None)
        if limiter is not None and EXHAUSTED.get(self.backup.config.label, 0) < time.time() and \
                limiter.expected_wait(self.primary.estimate(system, user)) > self.OVERFLOW_AFTER_SECONDS:
            try:
                return self.backup.structured(schema, system, user, role)
            except LLMError as e:
                if is_out_of_credit(e) or "402" in str(e):
                    EXHAUSTED.setdefault(self.backup.config.label, time.time() + 3600)
        try:
            return self.primary.structured(schema, system, user, role)
        except LLMError as first:  # limit used up, or repeated short rate limits / errors: same model, other host
            try:
                return self.backup.structured(schema, system, user, role)
            except LLMError as e:
                if is_out_of_credit(e) or "402" in str(e):  # no credit at the overflow host: stop trying it for now
                    EXHAUSTED.setdefault(self.backup.config.label, time.time() + 3600)
                raise LimitReached(f"{first}; overflow {self.backup.config.label} failed too: {str(e)[:200]}") from e


KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY",
           "groq": "GROQ_API_KEY", "nvidia": "NVIDIA_API_KEY", "mistral": "MISTRAL_API_KEY"}


class _CacheOnlyLLM:
    """Answers from the response cache when it can (earlier runs made with the key); otherwise delegates."""

    def __init__(self, config: ModelConfig, cache: "ResponseCache", namespace: str, fallback):
        self.config, self.cache, self.namespace, self.fallback = config, cache, namespace, fallback

    def structured(self, schema, system, user, role):
        hit = self.cache.get(ResponseCache.key(self.config, schema, system, user, self.namespace))
        if hit is not None:
            usage = Usage.model_validate(hit["usage"])
            usage.role, usage.cached, usage.seconds = role, True, 0.0
            return LLMResult(schema.model_validate(hit["parsed"]), usage)
        return self.fallback.structured(schema, system, user, role)


class MissingKeyLLM:
    """Stands in for a model whose API key is not set yet. Every call fails with LimitReached, so only the
    articles that need this model wait; the rest of the run goes ahead, and a re-run with the key resumes."""

    def __init__(self, config: ModelConfig, env: str):
        self.config, self.env = config, env

    def structured(self, schema, system, user, role):
        raise LimitReached(f"{self.config.label}: no {self.env} in .env yet; this step waits for it")


def build_llm(config: ModelConfig, cache_dir: str | Path | None = None, codebook=None, namespace: str = "") -> LLM:
    """`namespace` separates cache entries, e.g. per repeat, so repeats call the model again."""
    if config.overflow is not None:
        backup = build_llm(config.overflow, cache_dir, codebook, namespace)
        if isinstance(backup, LangChainLLM):
            backup.stops_run_without_credit = False
        return OverflowLLM(build_llm(config.model_copy(update={"overflow": None}), cache_dir, codebook, namespace), backup)
    env = KEY_ENV.get(config.provider)
    if env and not os.getenv(env) and not config.extra.get("api_key"):
        cache = ResponseCache(cache_dir)
        return _CacheOnlyLLM(config, cache, namespace, MissingKeyLLM(config, env))
    if config.provider == "fake":
        from adrd_review.fake_model import default_responders

        return FakeLLM(config, default_responders(codebook, style=config.model))
    return LangChainLLM(config, cache=ResponseCache(cache_dir), namespace=namespace)
