# Extraction agent

Code: `src/adrd_review/agents/extraction.py` · Exact prompt: [prompts/extraction.md](prompts/extraction.md) · Built: 2026-09-30 (Day 4)

## 1. Purpose

Pulls the key facts out of each **included** article, each with the exact words it came from: instruments evaluated, target condition, reference standard, study design, setting, population, sample size and country. It continues the data-capture role of the earlier Extraction Agent, which showed high sensitivity, and adds a quote for every value. Its facts are passed to the annotation reviewers as a checklist.

## 2. Where it sits

`screening → (include) extraction → annotation`. Excluded articles never reach it (saving one call each).

## 3. Inputs

The rendered article only. It does not see the codebook: extraction records what the paper says; judging it against the rules is the reviewers' job.

## 4. Output: `ExtractionRecord`

| Field | Meaning |
|---|---|
| `instruments` | List of `{value, quote}` for every screening instrument/test evaluated |
| `target_condition` | `{value, quote}`, e.g. all-cause dementia |
| `reference_standard` | `{value, quote}`, e.g. DSM-IV clinical diagnosis |
| `study_design` | `{value, quote}`, e.g. cross-sectional |
| `setting` | `{value, quote}`, e.g. primary care |
| `population` | `{value, quote}`, e.g. adults 65+ |
| `sample_size` | `{value, quote}` |
| `country` | `{value, quote}` |

`value` and `quote` are null when the text does not report the fact.

## 5. Prompt

System message: role, "fill each field from the article only, give the exact words, use null when not reported", and the evidence rules. User message: the rendered article. Exact text: [prompts/extraction.md](prompts/extraction.md).

## 6. Model and settings

| Run | Model |
|---|---|
| Gemini runs | `gemini-3.8-flash`, temperature 0, max 2,048 tokens |
| No-key runs | `fake` (lists instruments named in the text) |

## 7. Checks after the model answers

Structured output must match `ExtractionRecord`. Quotes are not yet checked here; the Verifier (Day 6) will check them.

## 8. Failure handling

3 attempts, then the article fails as a whole (listed in `failures.json`, predicted `include`).

## 9. Design decisions

- **No codebook in the prompt.** Keeps extraction factual and independent of the labelling decision, so an extraction error and a labelling error can be told apart.
- **Fixed fields.** Chosen to cover what the codebook's rules and dimensions depend on; they can be extended when the professor's codebook arrives.
- **Facts are shown to annotators as "check against the article".** Helps them without letting a wrong extraction silently decide a label.

## 10. Tests

`tests/test_agents_pipeline.py`: an included article gets an extraction containing its instrument; an excluded article gets none.

## 11. How to run and inspect it

Part of any run. Extractions are in `decisions.jsonl` under `extraction`.

## 12. Limitations and planned changes

- Fields will be revised to match the professor's codebook (e.g. AI method, data modality, dataset source for the AI/ML review topic).
- Quote checking moves to the Verifier (Day 6).
- Not yet scored: the gold standard has no extracted-fact labels.

## 13. Change log

- 2026-09-30: built (Day 4).
