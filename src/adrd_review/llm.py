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

Provider = Literal["google", "anthropic", "openai", "openrouter", "groq", "ollama", "fake"]


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
    """Seconds a provider asked us to wait ('Please try again in 1m12.5s'), if it said."""
    import re

    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", str(error))
    if not m:
        return None
    return int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3))


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
    if p in ("openai", "openrouter", "groq"):
        from langchain_openai import ChatOpenAI

        hosts = {
            "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
            "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
        }
        if p in hosts:
            url, env = hosts[p]
            kw = {"base_url": url, "api_key": os.getenv(env), **kw}
        return ChatOpenAI(**kw, max_tokens=config.max_tokens)
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
        estimate = (len(system) + len(user)) / 4 + (self.config.max_tokens or 1000) / 2
        last_error: Any = None
        for attempt in range(1, self.max_attempts + 1):
            self.limiter.wait(estimate)
            start = time.monotonic()
            try:
                out = runnable_for(which).invoke(messages)
            except Exception as e:  # network, rate limit, provider error, invalid tool call
                last_error = e
                if is_out_of_credit(e):
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
                    patience = float(os.getenv("ADRD_MAX_RATE_WAIT_SECONDS", "600"))
                    if asked is not None and asked > patience:
                        break
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


def build_llm(config: ModelConfig, cache_dir: str | Path | None = None, codebook=None, namespace: str = "") -> LLM:
    """`namespace` separates cache entries, e.g. per repeat, so repeats call the model again."""
    if config.provider == "fake":
        from adrd_review.fake_model import default_responders

        return FakeLLM(config, default_responders(codebook, style=config.model))
    return LangChainLLM(config, cache=ResponseCache(cache_dir), namespace=namespace)
