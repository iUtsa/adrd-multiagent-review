# Evaluation (the scorer)

For the results table, figures, error analysis, reviewer weights, cost and retrieval check, see [ANALYSIS.md](ANALYSIS.md).

How agent output is compared with the gold standard. Code: `src/adrd_review/eval/`.

---

## The idea

Every agent run writes a **predictions file in the same CSV format as the gold standard** (see [GOLD_STANDARD.md](GOLD_STANDARD.md)). The scorer lines the two files up by PMID and computes the metrics. One optional extra column, `score` (0–1, how likely the article is an include), enables WSS@95.

```csv
pmid,screening,instrument,setting,score
10812327,include,IQCODE;MMSE,primary_care,0.92
22840457,exclude,,,0.08
```

The same validation as the gold file applies (labels must be in the codebook, excluded articles carry no labels, and so on).

## How to run it

```bash
# Score on the 40 calibration articles (the default)
python -m adrd_review.eval.score --pred data/runs/E1/predictions.csv

# Save the full report as JSON too
python -m adrd_review.eval.score --pred data/runs/E1/predictions.csv --json data/runs/E1/report.json

# Final frozen runs only (prints a reminder)
python -m adrd_review.eval.score --pred data/runs/E4/predictions.csv --part test
```

Options: `--gold` (default `data/gold/gold.csv`), `--split` (default `configs/splits/placeholder.json`), `--part calibration | test | all`, `--codebook`.

The scorer stops with an error if any article in the chosen set has no prediction, so a crashed run can never look like a good score.

## What it measures

| Metric | Plain meaning | Function |
|---|---|---|
| Recall | Of the articles humans included, share the agents found | `binary_scores` |
| Precision | Of the articles agents included, share that were right | `binary_scores` |
| F1 | Balance of recall and precision | `binary_scores` |
| Specificity | Of the articles humans excluded, share the agents also excluded | `binary_scores` |
| Cohen's κ | Agreement with the humans beyond chance (0 = chance, 1 = perfect) | `cohen_kappa` |
| Fleiss' κ | Agreement among 3+ raters; used for the 3 agents and, when the gold file has per-reviewer columns, the 3 humans | `fleiss_kappa` |
| WSS | Share of articles a human no longer has to read, minus the share of relevant ones missed | `wss` |
| WSS@95 | Same, for a ranking: reading from the highest `score` down until 95% of relevant articles are found | `wss_at_recall` |
| Annotation P/R/F1 | Per label, micro (pooled) and macro (average over labels present in gold) | `multilabel_scores` |
| Annotation κ | Per label, agreement with humans beyond chance | `multilabel_scores` |
| Exact match | Share of articles where the whole label set is right | `multilabel_scores` |
| Cost and time | Tokens, dollars, seconds per article and total; speed-up vs human time | `usage_summary` |

Rules worth knowing:
- **Annotation is scored on the articles the gold standard includes.** If the agents wrongly excluded one, its labels count as empty, so a screening miss also costs annotation recall. This measures the whole pipeline, not annotation in isolation.
- A dimension with no gold labels at all is skipped rather than scored as zero.
- Empty denominators give 0.0 (same convention as scikit-learn).
- κ can be low even when accuracy is high, if one label dominates. Example below: MMSE.

## Scoring two-stage runs

When a run has a full-text stage, `predictions.csv` also has a `fulltext` column (the decision after reading the paper), and the report adds **FULL-TEXT ELIGIBILITY**: recall, precision, F1, specificity and κ against the gold `fulltext` column. An article excluded at stage 1 counts as excluded at stage 2. An unresolved "unclear" counts as include (it is waiting for a human).

Annotation is then scored on the articles the gold standard **finally** includes (after full text), not on everything that passed abstract screening.

Stage 1 (`screening`) is still scored against the gold abstract-stage decision, so each stage is compared with the matching human decision.

## What a pipeline run adds to the report

Runs made with `adrd_review.graph.runner` also report, in `report.json` and on screen:
- **Panel agreement:** Fleiss' κ among the reviewers' independent votes, per field, and how many articles were unanimous.
- **How decisions were reached:** counts of `single`, `unanimous`, `majority`, `tie_include`, `adjudicator`, `human` per field.
- **Escalations:** decisions sent to a human, by reason (`split_exclusion`, `low_confidence`, `verifier_fail`), the share of articles affected (human effort), and how many were answered.
- **Debate (variant D):** articles debated, answer changes, rejected changes, and changes toward vs away from the gold answer.

## How we know the scorer is right

`tests/test_metrics.py`:
- Hand-worked examples with known answers (including the worked example in the guide: 95% recall, 90% precision).
- On random data, precision/recall/F1 and Cohen's κ match **scikit-learn**, and Fleiss' κ matches **statsmodels**, to floating-point precision.
- Edge cases: no positive predictions, constant raters, uneven rater counts, missing predictions.

## Baselines (task 3.3)

The old 2-agent system's outputs are not available yet, so the scorer was validated end to end on simple non-AI baselines instead (`python scripts/score_baselines.py`). These are also the floor every agent condition must beat.

Calibration set, placeholder gold (40 articles, 13 includes):

| Baseline | Recall | Precision | F1 | κ | WSS | WSS@95 | Instrument F1 |
|---|---|---|---|---|---|---|---|
| Include everything | 1.000 | 0.325 | 0.491 | 0.000 | 0.000 | 0.000 | 0.000 |
| Exclude everything | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Random (seeded) | 0.308 | 0.286 | 0.296 | −0.061 | −0.042 | 0.025 | 0.000 |
| **Keyword match** | 0.769 | 0.435 | 0.556 | 0.240 | 0.194 | 0.000 | 0.848 |

The keyword baseline includes an article if its title/abstract names MMSE, MoCA, Mini-Cog or IQCODE **and** mentions dementia/Alzheimer's, and labels the instruments it names.

What it shows:
- Screening needs judgment: keyword matching finds most includes but half of what it includes is wrong (κ 0.24).
- Instrument labels are easy from text (F1 0.85), since the names are written in the abstract.
- MMSE: 80% precision and recall but κ only 0.13, because almost every include uses MMSE, so guessing it is easy. This is why κ is reported alongside F1.
- Setting labels score 0: they need understanding, not keyword matching.

## Human agreement (task 3.4)

The code is ready: when the gold file has per-reviewer columns (`screening__r1`, `screening__r2`, `screening__r3`), the report includes the humans' Fleiss' κ. The placeholder has no per-reviewer answers, so this waits for the professor's file.
