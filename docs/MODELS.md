# Model plan

**Constraint (2026-09-30): the professor has provided a Gemini licence.** Everything in the plan runs on Gemini. One free non-Google provider (Groq) is optional and only adds an experiment; nothing depends on it.

## Current plan (2026-10-10): Claude, with OpenAI when it has credit

Gemini is on hold. Utsa added Anthropic and OpenAI keys (both in `.env`, never committed).

| Role | `PROF_search_claude.yaml` | `PROF_search_mixed.yaml` (when OpenAI has credit) | $ per M tokens in / out |
|---|---|---|---|
| Reviewer A (clinical) | Claude Sonnet 5.5 | Claude Sonnet 5.5 | 2 / 10 |
| Reviewer B (machine learning) | Claude Sonnet 4.6 | **OpenAI gpt-6-sol** (second family) | 3 / 15 · 2 / 10 |
| Reviewer C (methods) | Claude Haiku 5.5 | Claude Haiku 5.5 | 0.10 / 0.50 |
| Adjudicator (split decisions only) | Claude Opus 5.5 (not on the panel) | same | 4 / 20 |
| Triage, verifier | Claude Haiku 5.5 | same | 0.10 / 0.50 |

- **Three different models**, so their errors are less correlated than one model run three times; the mixed config adds a second model family, which the evidence favours. Groq stays out: its free daily token limit (200,000 per model) covers fewer than 100 papers a day.
- **These models reject a temperature setting** (`temperature: null` in the configs).
- **Prompt caching** (`prompt_cache: true`): each reviewer's instructions and answer form (~2,500 tokens) are identical for every paper and are billed at 5–10% of the input price after the first call.
- **Account limits** (from response headers): 10,000 requests and 10 million input tokens per minute, so runs are not throttled.

**Pilot (2026-10-10), 100 random search results** (`--sample 100`): 65 completed before the Anthropic account ran out of credit (the other 35 were kept, not dropped). Cost **$6.37 for 65 papers, about $0.10 per paper** (before prompt caching), 4 minutes. Full run estimate: about $180 at that rate; prompt caching should cut the title and abstract stages substantially (to be measured).

## Which model does what (Gemini plan, on hold)

| Role | Model | Why |
|---|---|---|
| Triage | `gemini-3.5-flash-lite` | Simple check on every article; cheapest |
| Reviewer A | `gemini-3.8-flash` + clinical perspective | Newest Flash |
| Reviewer B | `gemini-3.7-flash` + epidemiology perspective | A different Gemini version adds some variety |
| Reviewer C | `gemini-3.5-flash` + methods perspective | Third version |
| Extraction | `gemini-3.8-flash` | Careful reading with quotes |
| Adjudicator (Day 6) | `gemini-3.1-pro-preview` | The strongest Gemini model, and **not on the panel**, so it has no "own vote" to favour |
| Verifier rule check (Day 6) | `gemini-3.5-flash-lite` | Cheap yes/no check |
| Codebook Curator (Day 7) | `gemini-3.1-pro-preview` | Runs a few times only |
| Development / tests | `fake` keyword model | No key, no cost |

Model IDs come from Google's model list (checked 2026-09-30). Preview models can change without notice; the model ID used is saved with every run, and final runs must use pinned versions.

## What changes from the original plan, and why it still works

| Original plan | With Gemini only | Effect on the study |
|---|---|---|
| 3 reviewers from 3 different companies | 3 Gemini versions × 3 perspectives | Less variety. Prior work found the biggest panel gains from different model families. We report this as a limitation and **measure it** with the optional mixed panel below. |
| Adjudicator from a 4th company | Gemini Pro (not used by any reviewer) | Still stronger than the reviewers and independent of their votes. Possible same-family bias is noted. |
| Everything else (triage, extraction, consensus, verifier, human review, curator, debate test, scoring) | Unchanged | None |

## Groq: second and third model families (key added 2026-09-30)

A Groq free-tier key is in `.env` (`GROQ_API_KEY`). Groq's general-purpose models, checked 2026-09-30:

| Model | Family | Structured output | Free limits |
|---|---|---|---|
| `openai/gpt-oss-120b` | OpenAI open-weights (large) | default works | 1,000 requests/day, 8,000 tokens/min |
| `openai/gpt-oss-20b` | OpenAI open-weights (small) | default works | same |
| `qwen/qwen3.8-27b` | Alibaba Qwen | needs `structured_method: function_calling` | same |

There is also a **daily token limit of 200,000 per model** that the response headers do not show (found on 2026-10-01 when a run hit it). A 3-reviewer run on 40 articles uses 125,000–210,000 tokens per model, so the free tier supports about **one 40-article run per day**, and a model with two roles can exceed it; the preflight check reports this, and an interrupted run continues the next day from the cache.

**The per-minute token limit is the other tight one**: a screening call uses ~2,400 tokens (gpt-oss spends ~800 on reasoning), so each model manages ~3 calls/min. Configs set `requests_per_minute: 3`, and rate-limit errors now back off 15/30/45/60 s (5 attempts). Fine for calibration runs; the full test set with repeats would exceed 1,000 requests/day per model.

| Config | Needs | Panel |
|---|---|---|
| `E1_groq.yaml` | Groq | one reviewer, gpt-oss-120b |
| `E2_groq.yaml` | Groq | gpt-oss-120b + qwen3.8-27b + gpt-oss-20b (runs today, no Gemini) |
| `E2_mixed.yaml` | Gemini + Groq | **three families**: Gemini 3.8 Flash + Qwen 3.8 + gpt-oss-120b; compare with `E2_gemini` |

First real runs (2026-09-30): in a 3-article check all three models correctly excluded a lymphoma paper whose abstract mentions the chemotherapy regimen "MOCA", the keyword trap that fooled the keyword baseline.

## Other free providers considered

To test whether variety between model families matters, which is a question the paper can answer, add one non-Google reviewer:

| Provider | Free limits (Sept 2026) | Fit |
|---|---|---|
| **Groq** (recommended) | 30 requests/min, 1,000/day, no credit card | Enough for a reviewer on 268 articles per day |
| OpenRouter free models | 20/min, but only 50/day without buying credit | Too few |
| GitHub Models | 50–150/day depending on model | Too few for full runs |

Sources: [OpenRouter free LLM API comparison](https://openrouter.ai/blog/tutorials/free-llm-apis-compared/), [KDnuggets free LLM API providers](https://www.kdnuggets.com/5-free-llm-api-providers-you-can-use-in-2026).

Groq was chosen and is set up (above). Comparing `E2_gemini` with `E2_mixed` is the "same family vs different families" experiment.

## Two kinds of Gemini access

The code supports both; which one the licence is decides the setup:

| If the licence is… | Setup |
|---|---|
| An **API key** from Google AI Studio (starts with `AIza`) | Put it in `.env` as `GOOGLE_API_KEY=...` |
| A **Google Cloud / Vertex AI** project | Run `gcloud auth application-default login` once, then add `extra: {vertexai: true, project: <project-id>, location: us-central1}` to each model in the run config |
| A **Gemini app subscription** (gemini.google.com, Gemini for Workspace) | This does **not** include API access. Ask for an AI Studio key or a Vertex AI project, or use the free AI Studio tier |

## Free tier and prices

Prices are in `configs/prices.yaml` (checked 2026-09-30). **`gemini-3.1-pro-preview` has no free tier**: on a free key, set the E3/E4/D adjudicator to `gemini-3.8-flash` (each Gemini config says so at the adjudicator line). All Flash and Flash-Lite models have a free tier. Estimated test-run cost: about $30–40 ([ANALYSIS.md](ANALYSIS.md#cost-estimate-task-84)).

## Budget notes

- Calls per article: triage 1, screening 1 per reviewer, extraction 1 (includes only), annotation 1 per reviewer (includes only). A 3-reviewer run on 40 articles is about 250 calls; on 228 test articles about 1,400.
- Every answer is cached, so re-running an unchanged configuration costs nothing.
- Add `price_in_per_mtok` / `price_out_per_mtok` to a model config to get dollar costs in the report.
- Rate limits are set per model (`requests_per_minute`, default 10 in the configs); raise them if the licence allows.
