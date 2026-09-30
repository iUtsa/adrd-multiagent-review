# Model plan

**Constraint (2026-09-30): the professor has provided a Gemini licence.** Everything in the plan runs on Gemini. One free non-Google provider (Groq) is optional and only adds an experiment; nothing depends on it.

## Which model does what

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

## Optional: a second model family for free (recommended)

To test whether variety between model families matters, which is a question the paper can answer, add one non-Google reviewer:

| Provider | Free limits (Sept 2026) | Fit |
|---|---|---|
| **Groq** (recommended) | 30 requests/min, 1,000/day, no credit card | Enough for a reviewer on 268 articles per day |
| OpenRouter free models | 20/min, but only 50/day without buying credit | Too few |
| GitHub Models | 50–150/day depending on model | Too few for full runs |

Sources: [OpenRouter free LLM API comparison](https://openrouter.ai/blog/tutorials/free-llm-apis-compared/), [KDnuggets free LLM API providers](https://www.kdnuggets.com/5-free-llm-api-providers-you-can-use-in-2026).

With a Groq key (`GROQ_API_KEY` in `.env`), an `E2_mixed` config replaces reviewer C with an open-weights model on Groq. The code already supports it (`provider: groq`). Comparing `E2_gemini` with `E2_mixed` becomes the "same family vs different families" experiment.

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
