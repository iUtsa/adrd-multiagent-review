# Adjudicator

Code: `src/adrd_review/agents/adjudicator.py` · Exact prompt: [prompts/adjudicator.md](prompts/adjudicator.md) · Built: 2026-09-30 (Day 6)

## 1. Purpose

Settles a decision the reviewers **disagree** on, for screening or any annotation dimension, by weighing each reviewer's position against the article and the rules. It addresses the earlier system's weakness with **borderline studies**: instead of a bare majority, a stronger model reads the competing evidence. It is the senior editor of the human protocol. It never sees unanimous decisions.

## 2. Where it sits

```
[reviewers] -> consensus --split--> (debate rounds, variant D only) -> adjudicator -> safety checks -> ...
```

Used for screening (`screen_adjudicate`) and, after annotation, once per disputed dimension (`annotate_adjudicate`). Switched on in E3, E4 and D by an `adjudicator:` model in the run config.

## 3. Inputs

- The rendered article.
- The field being settled, its valid answers, and its rules (screening rules, or the dimension's question and labels).
- Every reviewer's latest answer: labels, rule IDs, quotes and reasoning (named A, B, C; models are not shown).

## 4. Output: `AdjudicationAnswer` → `Decision` + `Adjudication` record

| Field | Meaning |
|---|---|
| `labels` | The adjudicator's own answer |
| `reviewer_weights` | 0–1 per reviewer: how well the text and rules support that reviewer's position, with a one-line reason |
| `rule_ids` | Rules its answer rests on |
| `rationale` | Which evidence settles the disagreement |
| `confidence` | 0–1; asked to be low when the article is genuinely ambiguous |

**How the final answer is formed:** the reviewers' votes are combined **using the adjudicator's weights** (same rules as the panel's weighted vote). The adjudicator therefore cannot introduce an answer no reviewer gave; it can only decide how much each reviewer's case counts. The `Decision` has `method: adjudicator` and the adjudicator's confidence.

## 5. Prompt

System: role, the field and its valid answers, the field's rules, instructions (judge evidence not headcount; a quote not in the article counts for nothing; weight each reviewer; state your own answer and an honest confidence), evidence rules. User: the article plus all reviewers' answers. Exact text: [prompts/adjudicator.md](prompts/adjudicator.md).

## 6. Model and settings

| Run | Model |
|---|---|
| E3/E4/D on Gemini | `gemini-3.1-pro-preview` (strongest Gemini; not on the panel), max 2,048 tokens, 5 requests/min |
| No-key runs | `fake`: weighs reviewers by how many quotes they cited |

`escalate_below` (default 0.6) in the run config sets the confidence below which the decision goes to a human.

## 7. Checks after the model answers

- Labels that are not valid answers for the field are dropped (warning).
- A reviewer with no weight gets 0.5 (warning); weights for unknown reviewers are ignored; all-zero weights fall back to equal weights (warning).
- **Consistency check:** if the adjudicator's own answer differs from what its weights produce, the decision's confidence is set to 0 so it **goes to a human** (warning). A model that contradicts itself is not trusted either way.
- Low confidence (< `escalate_below`) → escalation `low_confidence`.

## 8. Failure handling

Retries as for every agent; then the article fails as a whole (listed in `failures.json`, predicted `include`).

## 9. Design decisions

- **Weighs rather than overrides.** An adjudicator that weighted reviewers' scores beat one that picked a single answer in multi-LLM abstract screening ([Akinseloyin et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12952526/)).
- **Only for splits.** Unanimous decisions skip it: most articles cost nothing extra.
- **Strongest model, not on the panel.** With Gemini only, independence comes from using Gemini Pro, which no reviewer uses, so it has no vote of its own to favour.
- **Honest confidence drives human review.** Its uncertainty is the main route to a person, alongside the screening safety rule and the verifier.

## 10. Tests

`tests/test_adjudication_debate_human.py`: weights decide the outcome; inconsistent answer → confidence 0 and warning; missing weights filled; invalid labels dropped; in the graph, splits go to the adjudicator and low-confidence adjudications are escalated; unanimous decisions skip it.

## 11. How to run and inspect it

```bash
python -m adrd_review.graph.runner configs/runs/E3_fake.yaml
```

Each adjudication (weights, rationale, confidence, warnings) is in `decisions.jsonl` under `adjudications`; `report.json` counts them. Fake E3 run: 20 of 40 screening decisions were split and adjudicated.

## 12. Limitations and planned changes

- `gemini-3.1-pro-preview` is a preview model: pin the exact version before the freeze.
- Same-family bias (a Gemini judge weighing Gemini reviewers) is a noted limitation; the optional mixed panel (Groq) partly tests it.

## 13. Change log

- 2026-09-30: built (Day 6).
