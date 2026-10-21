# Second look

Code: `src/adrd_review/second_look.py` (uses `agents/eligibility.py`) · Output: `<run>/second_look.csv` · Built: 2026-10-21

## 1. Purpose

Before a result is final, every paper the panel left **for a person** at full text ("unclear": no rule clearly failed, but not every inclusion rule was clearly met) is re-read by an independent, stronger model. It resolves cases where the panel missed evidence that is in the paper (for example an age table), so the professor only gets the papers that really do not say. Asked for by Utsa (2026-10-21): "whatever you put for person, double-check before the final result".

## 2. Where it sits

After a run, on the papers whose full-text decision is "unclear" (same assignment as the stage files). It does not change the run's decisions; its result is a separate table, reported as its own step.

## 3. Inputs

- The **same full text** the panel read (saved in the run's `decisions.jsonl` / `in_progress.jsonl`).
- The rules the panel left open (rules cited by the "unclear" and dissenting votes), named in the persona.
- **Not** the panel's votes, so it is not anchored by them.

## 4. Output

The full-text reviewer's form (`ScreeningAnswer`): a mark per rule with quotes, rationale, confidence → `include` / `exclude` / `unclear` by the full-text decision rule. In `second_look.csv`: PMID, link, title, year, outcome (included / excluded / for human review), open rules, the panel's votes, the second look's rule marks, quotes, reasoning, model.

## 5. Prompt

The full-text eligibility prompt (`agents/eligibility.py`, [prompts/eligibility.md](prompts/eligibility.md)) with this persona: *"the senior second reviewer. A panel of three reviewers read this paper and could not settle these rules: {rules}. Read the whole paper, including the methods, participant tables and the description of the data set, and look specifically for evidence on those rules. Mark a rule met or not met only with a quote from the paper; if the paper really does not say, mark it unclear."*

## 6. Model and settings

**GLM-5.3** (Z.ai), a model family not on the panel, on OpenRouter (paid, $0.079 / $1.40 per M tokens, about $0.002-0.007 per paper). **16,000 output tokens:** it reasons before answering, and with 4,096 long papers were cut off, retried and billed (about $0.6 lost on 2026-10-21). No overflow host: NVIDIA's free copy hung for minutes per paper. Concurrency 4 by default (`--concurrency`).

## 7. Checks after the model answers

The same as every full-text reviewer: every quote is searched for in the paper; a "not met" or "applies" without a real quote is treated as unclear; include needs all four inclusion rules met with quotes.

## 8. Failure handling

A failed call leaves the paper "for human review" with the error in the reasoning column; re-running is free for every paper already answered (cache).

## 9. Design decisions

- **Not the adjudicator:** the adjudicator re-weights the panel's votes, so it cannot turn a unanimous "unclear" into a decision. The second look re-reads the paper instead.
- **Rules named, votes hidden:** pointing at the open rules makes the search targeted; hiding the votes keeps it independent.
- **Same evidence standard:** no new rule; it cannot invent information the paper does not contain (an unreported age stays unclear).

## 10. Tests

`tests/test_three_stage.py::test_second_look_rereads_only_the_papers_left_for_a_person`: only papers left for a person are read; open rules are found; a failing model leaves the paper with the professor.

## 11. How to run and inspect it

```bash
python -m adrd_review.second_look data/runs/PROF_search_free/<timestamp>
```

Prints the counts and writes `second_look.csv`; `python -m adrd_review.eval.report_figures <run>` adds `second_look.png`.

## 12. Limitations

One model; its own accuracy is not measured. Papers that never report age stay with the professor unless she sets a rule for that.

## 13. Change log

- 2026-10-21: built; first run on the 120 papers left for a person in try 6.
