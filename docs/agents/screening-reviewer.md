# Screening reviewer

Code: `src/adrd_review/agents/screening.py` · Exact prompt: [prompts/screening.md](prompts/screening.md) · Built: 2026-09-30 (Day 4)

## 1. Purpose

Decides whether an article should be **included or excluded** under the codebook's rules, and must back the decision with rule IDs, word-for-word quotes, a short rationale and a confidence score. It replaces the screening part of the earlier Extraction Agent. Used once in E1 and three times (with different perspectives and models) in the reviewer panel from E2 on.

## 2. Where it sits

`triage → screening → (include) extraction → annotation`, or `→ (exclude) END`. In panel runs, three copies run in parallel and their votes go to the consensus step ([reviewer-panel.md](reviewer-panel.md)).

## 3. Inputs

- The rendered article (see [triage.md](triage.md#3-inputs)).
- The review question, inclusion rules, exclusion rules and decision rule, all rendered from the codebook.
- Optionally a **persona** (a perspective such as "clinical neurologist"), set in the run config.
- The evidence rules: quote word for word, never paraphrase, say "not reported" rather than guess.

## 4. Output: `ScreeningAnswer` → stored as a `Vote`

| Field | Meaning |
|---|---|
| `decision` | `include` or `exclude` |
| `rule_ids` | Rules the decision rests on, e.g. `["I1","I2"]`, or the failing/applying rule for an exclusion |
| `evidence` | 1–3 exact quotes from the title or abstract |
| `rationale` | At most three sentences |
| `confidence` | 0–1 |

The code stores this as a `Vote` (`field="screening"`, `labels=[decision]`, plus reviewer ID, model, warnings), the same record used for annotation votes, so the consensus step treats both alike.

## 5. Prompt

System message: role, optional persona, "apply the rules exactly, judge only from the text", the rendered rules, the evidence rules, and what to return. User message: the rendered article. Exact text: [prompts/screening.md](prompts/screening.md).

## 6. Model and settings

| Run | Model |
|---|---|
| E1 on Gemini | `gemini-3.8-flash`, temperature 0, max 2,048 tokens, 10 requests/min |
| Panel on Gemini | See [reviewer-panel.md](reviewer-panel.md) |
| No-key runs | `fake` (include if a listed instrument and dementia are both named) |

## 7. Checks after the model answers

- Structured output must match `ScreeningAnswer`; malformed answers are retried.
- **Unknown rule IDs** are dropped and recorded as a warning.
- **Every quote is searched for in the title + abstract** (ignoring case and spacing). A quote that is not there is recorded as a warning: the quick form of the Verifier. Nothing is silently corrected.

## 8. Failure handling

As for every agent: 3 attempts, then the article is listed in `failures.json` and predicted `include` (recall guard).

## 9. Design decisions

- **Rule IDs and quotes are required.** Transparency and auditability are design requirements from the paper; they also let the adjudicator (Day 6) weigh evidence instead of votes alone.
- **Benefit of the doubt at abstract stage.** The codebook's decision rule tells reviewers to include when the abstract cannot rule a criterion out, matching standard title/abstract screening practice and protecting recall.
- **Confidence is recorded but not trusted alone.** LLM self-reported confidence is poorly calibrated; the panel uses agreement between reviewers as the main signal.
- **Personas are optional prompt lines**, so the same agent code serves every reviewer in the panel.

## 10. Tests

`tests/test_agents_pipeline.py`: all rule IDs appear in the prompt; persona line appears when set; unknown rule IDs dropped with a warning; invented quote flagged; include and exclude paths through the graph.

## 11. How to run and inspect it

```bash
python -m adrd_review.graph.runner configs/runs/E1_fake.yaml
```

Votes, quotes, rationales and warnings are in `decisions.jsonl` (`screening_votes`); the decision per article is in `predictions.csv`.

## 12. Limitations and planned changes

- Quotes are checked only against title + abstract; full text comes with the PMC tool.
- Prompts currently say "dementia screening research"; the review question itself comes from the codebook and will change with the professor's codebook.

## 13. Change log

- 2026-09-30: built (Day 4). Panel use added (Day 5).
