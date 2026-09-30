# Agents and the pipeline

How the agents are built (LangChain), wired together (LangGraph), and run. Current stage: **E1**, one reviewer agent. Days 5–8 add the 3-reviewer panel, adjudicator, verifier and debate test on top of the same pieces.

---

## The E1 pipeline

```
START -> triage -> screen --include--> extract -> annotate -> END
                         \--exclude--> END
```

Code: `src/adrd_review/graph/pipeline.py`. Excluded articles stop after screening, so they cost 2 model calls instead of 4.

## The agents

Each agent is a small module in `src/adrd_review/agents/`: a system prompt, the article as the user message, and a fixed answer form (Pydantic schema) the model must fill in. **All rule and label text in the prompts is generated from `configs/codebook.yaml`**, so swapping in the professor's codebook updates every prompt with no code changes.

| Agent | File | Model returns | What the code then checks |
|---|---|---|---|
| Triage | `triage.py` | `TriageReport`: is the abstract enough? which rules can't be judged? | Rule IDs exist in the codebook |
| Screening reviewer | `screening.py` | `ScreeningAnswer`: include/exclude, rule IDs, 1–3 exact quotes, rationale, confidence | Unknown rule IDs dropped; every quote searched for in the article |
| Extraction | `extraction.py` | `ExtractionRecord`: instruments, target condition, reference standard, design, setting, population, sample size, country, each with a quote | — |
| Annotation reviewer | `annotation.py` | `AnnotationAnswer`: labels per codebook dimension, quotes, rationale | Labels not in the codebook dropped; single-label dimensions keep one; quotes checked |

Anything dropped or not found is kept as a **warning** on the vote (in `decisions.jsonl`) and counted in the report. Nothing is silently fixed.

Shared prompt pieces (`agents/common.py`): how an article is shown to the model (title, journal/year, publication type, non-English language, MeSH terms, abstract or "(no abstract available)"), how rules and labels are listed, and the evidence rules (quote word for word, never paraphrase, say "not reported" instead of guessing).

Answer forms: `src/adrd_review/schemas.py`. The field descriptions there are sent to the model as instructions.

## Models: the factory and the cache

`src/adrd_review/llm.py`. Every agent calls models through one method, `llm.structured(schema, system, user, role)`, which returns the validated answer plus a usage record (tokens, cost, seconds, cached or not).

| Provider | LangChain class | Key in `.env` |
|---|---|---|
| `google` | `ChatGoogleGenerativeAI` | `GOOGLE_API_KEY` |
| `anthropic` | `ChatAnthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `ChatOpenAI` | `OPENAI_API_KEY` |
| `openrouter` (open-weights models) | `ChatOpenAI` via openrouter.ai | `OPENROUTER_API_KEY` |
| `ollama` (local, on standby) | `ChatOllama` | none |
| `fake` | keyword rules, no AI | none |

Built-in behaviour:
- **Structured output.** Uses LangChain's `with_structured_output`, so the model must return the answer form; a malformed answer is retried (3 attempts in total).
- **Response cache** (`data/cache/llm/`). The key is a hash of provider, model, temperature, max tokens, answer form, system prompt and article. An identical request is answered from disk for free. Any change to the prompt, codebook or model makes a new request.
- **Rate limit** per model (`requests_per_minute`), to stay under free-tier limits.
- **Retries** on network or provider errors, with increasing waits.
- **Cost** is computed when the config gives prices (USD per million tokens); otherwise tokens are still recorded.

### The fake model

`src/adrd_review/fake_model.py`. Answers with the same keyword rules as the keyword baseline. It exists so the pipeline can be built, tested and run with **no API key**. Its scores are a plumbing check, never a result, and the runner says so.

## Run configs

`configs/runs/*.yaml` choose a model per role:

```yaml
name: E1_gemini
condition: E1
part: calibration          # calibration | test | all
concurrency: 2             # articles processed at once
roles:
  triage:     {model: {provider: google, model: gemini-3.5-flash-lite, requests_per_minute: 10}}
  screening:  {model: {provider: google, model: gemini-3.8-flash, requests_per_minute: 10}}
  extraction: {model: {provider: google, model: gemini-3.8-flash, requests_per_minute: 10}}
  annotation: {model: {provider: google, model: gemini-3.8-flash, requests_per_minute: 10}}
```

A role can also have a `persona` (used from Day 5 for the reviewer panel).

| Config | Needs | Use |
|---|---|---|
| `E1_fake.yaml` | nothing | plumbing check |
| `E1_gemini.yaml` | `GOOGLE_API_KEY` | first real E1 run |

## Running

```bash
source .venv/bin/activate
python -m adrd_review.graph.runner configs/runs/E1_fake.yaml              # 40 calibration articles
python -m adrd_review.graph.runner configs/runs/E1_gemini.yaml --limit 3  # quick real check
python -m adrd_review.graph.runner configs/runs/E1_gemini.yaml            # full calibration run
```

Each run writes `data/runs/<name>/<timestamp>/`:

| File | Contents |
|---|---|
| `predictions.csv` | One row per article, gold format; what the scorer reads |
| `decisions.jsonl` | Everything per article: triage, every vote with quotes, rule IDs, rationale, confidence, warnings; extraction |
| `usage.jsonl` | One line per model call: role, model, tokens, cost, seconds, cached |
| `failures.json` | Articles that errored, with the error |
| `report.json` | Scores, cost/time summary, failure and warning counts |
| `run_meta.json` | Config, codebook version + hash, gold hash, split file, git commit, timings |

**Failed articles are predicted `include`** (score 0.5) so a crash can never silently drop a study; they are listed in `failures.json` and counted in the report.

## Checks so far

- 59 tests pass without any key (`tests/test_agents_pipeline.py`): prompts built from the codebook, invented quotes flagged, unknown rules/labels dropped with warnings, cache hit on identical requests, retry then give up on bad output, cost from token counts, include path runs 4 steps, exclude path stops after 2, failures captured.
- Every provider's LangChain model builds with the configured settings (checked with dummy keys; no network).
- Full fake run on the 40 calibration articles: 126 model calls, 0 failures, 0 warnings, all six output files written; screening scores identical to the keyword baseline, which confirms the pipeline and scorer pass results through correctly.
- Keyword traps seen in real data: "MOCA" as a chemotherapy regimen and as an industrial chemical. A real model has to read, not match words.
