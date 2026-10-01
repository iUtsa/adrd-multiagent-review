# Triage agent

Code: `src/adrd_review/agents/triage.py` · Exact prompt: [prompts/triage.md](prompts/triage.md) · Built: 2026-09-30 (Day 4)

## 1. Purpose

Decides whether an article's title and abstract contain enough information to apply each inclusion rule. It does **not** decide include or exclude. It addresses the earlier finding that *variable abstract reporting quality* made classification inconsistent: triage makes that visible per article and can trigger a full-text fetch from PubMed Central when the abstract is too thin (`fulltext: when_needed`).

## 2. Where it sits

First step for every article: `START → triage → screening`. Its report is stored in the article's state and in `decisions.jsonl`. With `fulltext: when_needed` in the run config, a report saying the abstract is not enough makes the pipeline fetch the PubMed Central full text (if the article has a PMCID) before the reviewers run ([TOOLS.md](../TOOLS.md)).

## 3. Inputs

- The article, rendered by `render_article` (title, journal and year, publication type, non-English language, MeSH terms, abstract or "(no abstract available)").
- The codebook's review question, inclusion and exclusion rules and decision rule (`render_screening_rules`).

## 4. Output: `TriageReport`

| Field | Meaning |
|---|---|
| `clearly_ineligible` | (model) True if the text already shows the study cannot qualify, e.g. it is about another topic |
| `missing` | (model) IDs of inclusion rules that cannot be judged **either way**, e.g. `["I2", "I3"]` |
| `note` | (model) One sentence on what is missing |
| `enough_information` | **(computed by the code)** `clearly_ineligible` or nothing is missing |

## 5. Prompt

A short system message: role, "you do not decide inclusion", the rendered rules, and the task (list the rules that cannot be judged). User message: the rendered article. Built entirely from the codebook, so a new codebook changes the prompt automatically. Exact text: [prompts/triage.md](prompts/triage.md).

## 6. Model and settings

Set per run in `configs/runs/*.yaml` under `roles.triage` (or the top-level `triage` for panel runs).

| Run | Model | Why |
|---|---|---|
| Gemini runs | `gemini-3.5-flash-lite`, temperature 0, max 1,024 tokens | A simple check; the cheapest capable model |
| No-key runs | `fake` (keyword rules: no abstract → not enough information) | Plumbing only |

## 7. Checks after the model answers

- The answer must match `TriageReport` (LangChain structured output; malformed answers are retried).
- Rule IDs in `missing` that are not inclusion rules in the codebook are removed; `missing` is emptied when the article is clearly ineligible.
- `enough_information` is computed, never taken from the model: in the first real run the model answered "not enough information" for every article, including off-topic ones, sometimes with nothing listed as missing.

## 8. Failure handling

Provider errors and malformed answers are retried (3 attempts in total, with increasing waits). If all fail, the article's whole run fails, is listed in `failures.json`, and is predicted `include` so it goes to a human rather than being dropped.

## 9. Design decisions

- **Separate from screening.** Keeping "is there enough information?" apart from "does it qualify?" lets us measure how often poor abstracts, not reasoning errors, cause mistakes: one of the four error types in the evaluation plan.
- **Cheapest model.** It runs on every article, and the task is simple.
- **Does not block screening.** Under the codebook's decision rule, reviewers give the benefit of the doubt when information is missing, so screening still runs; triage adds the signal and, later, the full text.

## 10. Tests

`tests/test_agents_pipeline.py`: prompt built from codebook rules; missing-abstract rendering; full pipeline runs triage first on included and excluded articles.

## 11. How to run and inspect it

Runs as part of any pipeline run. Inspect a run's reports:

```bash
python -c "import json; [print(json.loads(l)['pmid'], json.loads(l)['triage']) for l in open('data/runs/E1_fake/<timestamp>/decisions.jsonl')]"
```

## 12. Limitations and planned changes

- Full-text fetch from PubMed Central when `enough_information` is false: built (`fulltext: when_needed`); off by default until the professor confirms the human protocol.
- Not yet scored against a reference; once real runs exist, compare its "not enough information" rate with the no-abstract and error-analysis results.

## 13. Change log

- 2026-09-30: built (Day 4). Full-text branch added.
- 2026-09-30: after the first real run, split the question into `clearly_ineligible` + `missing`, with `enough_information` computed in code.
