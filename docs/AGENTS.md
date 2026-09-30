# Agents and the pipeline

How the agents are built (LangChain), wired together (LangGraph), and run. **All conditions are built:** E1 (one reviewer), E2 (3-reviewer panel + vote), E3 (+ adjudicator + verifier), E4 (+ human review, the proposed system) and D (+ debate). How to run them in order: [RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md).

**Detailed build documentation for each agent is in [docs/agents/](agents/README.md)**, with the exact prompts generated from the live code. Model choices: [MODELS.md](MODELS.md).

---

## The pipeline

```
START -> triage -> [screen_reviewer x N] -> screen_consensus
   -> (split, debate rounds left) screen_debate -> screen_consensus ...
   -> (split, adjudicator) screen_adjudicate
   -> screen_checks (safety rule) -> (escalated, human on) screen_human
   --include--> extract -> [annotate_reviewer x N] -> annotate_consensus
          -> (split, rounds left) annotate_debate -> annotate_consensus ...
          -> (split, adjudicator) annotate_adjudicate
   -> finalize (verifier) -> (escalated, human on) final_human -> END
   --exclude--> finalize -> ...
```

One graph serves every condition; the run config switches steps on (`panel`, `adjudicator`, `verifier`, `debate_rounds`, `human`).

Code: `src/adrd_review/graph/pipeline.py`. The N reviewers run in parallel (LangGraph `Send`) and never see each other's votes. E1 is N = 1; E2 is N = 3. Excluded articles stop after screening. How votes are combined: [agents/reviewer-panel.md](agents/reviewer-panel.md).

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
| `google` | `ChatGoogleGenerativeAI` (AI Studio key, or Vertex AI via `extra`) | `GOOGLE_API_KEY` |
| `anthropic` | `ChatAnthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `ChatOpenAI` | `OPENAI_API_KEY` |
| `openrouter` (open-weights models) | `ChatOpenAI` via openrouter.ai | `OPENROUTER_API_KEY` |
| `groq` (open-weights models, free tier) | `ChatOpenAI` via api.groq.com | `GROQ_API_KEY` |
| `ollama` (local, on standby) | `ChatOllama` | none |
| `fake` | keyword rules, no AI | none |

Built-in behaviour:
- **Structured output.** Uses LangChain's `with_structured_output`, so the model must return the answer form; a malformed answer is retried (3 attempts in total).
- **Response cache** (`data/cache/llm/`). The key is a hash of provider, model, temperature, max tokens, answer form, system prompt and article. An identical request is answered from disk for free. Any change to the prompt, codebook or model makes a new request.
- **Rate limit** per model (`requests_per_minute`), to stay under free-tier limits.
- **Retries** on network or provider errors, with increasing waits.
- **Cost** is computed when the config gives prices (USD per million tokens); otherwise tokens are still recorded.
- **Extra options** pass straight to the LangChain model (`extra:`), e.g. Vertex AI project/location or Gemini's `thinking_budget`; `structured_method:` picks how structured output is requested.

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

Panel runs keep `roles.triage` and `roles.extraction` and list the reviewers under `panel:` (id, persona, model, optional weight); see [agents/reviewer-panel.md](agents/reviewer-panel.md).

| Config | Needs | Use |
|---|---|---|
| `E1_fake.yaml` | nothing | one reviewer, plumbing check |
| `E1_gemini.yaml` | Gemini access | first real E1 run |
| `E2_fake.yaml` | nothing | 3-reviewer panel, plumbing check |
| `E2_gemini.yaml` | Gemini access | 3 Gemini versions × 3 perspectives |
| `E3_fake.yaml` / `E3_gemini.yaml` | nothing / Gemini | + adjudicator (Gemini Pro) + verifier |
| `E4_fake.yaml` / `E4_gemini.yaml` | nothing / Gemini | + human review (gold answers stand in for the person) |
| `E4_queue_fake.yaml` | nothing | + human review that really pauses for a person |
| `D_fake.yaml` / `D_gemini.yaml` | nothing / Gemini | E3 + 2 debate rounds |

Extra config keys: `adjudicator:` (model), `escalate_below:` (default 0.6), `verifier: {enabled, model}`, `debate_rounds:`, `human: none | gold | queue`.

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
| `report.json` | Scores, cost/time summary, failure and warning counts, panel agreement (Fleiss' κ among reviewers), how each decision was reached, escalations, adjudications, debate statistics |
| `checkpoints.sqlite` | `human: queue` only: saved state of paused articles |
| `curator_proposals.md/.json` | After running the Curator on a calibration run |
| `run_meta.json` | Config, codebook version + hash, gold hash, split file, git commit, timings |

**Failed articles are predicted `include`** (score 0.5) so a crash can never silently drop a study; they are listed in `failures.json` and counted in the report.

## Checks so far

- 59 tests pass without any key (`tests/test_agents_pipeline.py`): prompts built from the codebook, invented quotes flagged, unknown rules/labels dropped with warnings, cache hit on identical requests, retry then give up on bad output, cost from token counts, include path runs 4 steps, exclude path stops after 2, failures captured.
- Every provider's LangChain model builds with the configured settings (checked with dummy keys; no network).
- Full fake run on the 40 calibration articles: 126 model calls, 0 failures, 0 warnings, all six output files written; screening scores identical to the keyword baseline, which confirms the pipeline and scorer pass results through correctly.
- Keyword traps seen in real data: "MOCA" as a chemotherapy regimen and as an industrial chemical. A real model has to read, not match words.
