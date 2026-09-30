# Analysis tools

Everything used after the runs: repeats, the results table, figures, error analysis, reviewer weights, cost and the retrieval check. All live in `src/adrd_review/eval/` and run without an API key.

---

## Repeats (task 9.1)

```bash
python -m adrd_review.graph.runner configs/runs/E3_gemini.yaml --part test --repeat 1
python -m adrd_review.graph.runner configs/runs/E3_gemini.yaml --part test --repeat 2
python -m adrd_review.graph.runner configs/runs/E3_gemini.yaml --part test --repeat 3
```

Without `--repeat`, an identical request is answered from the response cache, so a "repeat" would just replay run 1. `--repeat N` puts the repeat number into the cache key: each repeat calls the model again, and its answers are cached separately (re-running the same repeat is still free). Run folders are named `<timestamp>-r<N>`; `run_meta.json` records `repeat`.

**Sharing across conditions.** Within one repeat number, conditions reuse identical model calls. E4 makes exactly E3's calls plus a human step, so after E3 it costs almost nothing, and the conditions are compared on the *same* reviewer answers (a paired comparison, which is what the ablation wants).

## Results table (tasks 9.1, 10.1)

```bash
python -m adrd_review.eval.aggregate                  # calibration
python -m adrd_review.eval.aggregate --part test      # final
```

Writes `data/results/<part>/summary.md`, `summary.csv` and `runs_used.csv`. For each condition: mean ± SD over repeats of recall, precision, F1, specificity, κ, WSS, WSS@95, panel Fleiss' κ, escalation rate, adjudications, tokens, cost, wall time and failures, plus annotation micro-F1 per dimension.

**Run-to-run consistency:** `repeat_kappa` (mean Cohen's κ between repeats' screening answers) and `includes_not_stable` (articles whose answer changed between repeats).

**Only comparable runs are averaged.** Each run records a fingerprint of its config, codebook, gold file, article set and agent code (prompts). Runs are grouped by fingerprint and the most recent group per config is used, so runs made before a code or prompt change are never mixed in. Quick checks (`--limit`) and unfinished runs are skipped.

## Figures (task 10.3)

```bash
python -m adrd_review.eval.figures conditions --part calibration
python -m adrd_review.eval.figures effort data/runs/E3_gemini/<timestamp> --mark data/runs/E4_gemini/<timestamp>
```

| Figure | Shows |
|---|---|
| `fig_conditions` | Recall, precision and κ per condition, mean ± SD over repeats |
| `fig_effort_<run>` | **Accuracy vs human effort:** starting from a run's answers, articles go to a human (gold answer) least-certain first. F1 and recall as the share reviewed rises; `--mark` plots an E4 run's actual operating point. |

Each is saved as PNG (slides) + SVG (paper) + CSV with the exact plotted numbers (the table view). Colours are the first three slots of a validated colour-blind-safe palette (checked all-pairs: worst CVD ΔE 9.2, normal-vision ΔE 24.0); one y-axis; solid hairline grid; legend on every multi-series chart.

## Error analysis (task 10.2)

```bash
python -m adrd_review.eval.errors export data/runs/E4_gemini/<timestamp>
# fill in the error_category column in Excel, then:
python -m adrd_review.eval.errors summarize data/runs/E4_gemini/<timestamp>/error_analysis.xlsx
```

`error_analysis.xlsx` has one row per disagreement with the gold standard (screening, and each annotation dimension of gold includes): title, year, gold vs agent answer, how it was decided, escalations, each reviewer's vote with rule IDs, reasoning, quotes, the adjudicator's rationale, the abstract, and a hint when information was missing. The `error_category` column is a drop-down:

| Category | Use when |
|---|---|
| unclear rule (boundary ambiguity) | The wording allows both answers |
| keyword trap | The agents matched a word (e.g. "MOCA" the chemical) without its meaning |
| wrong topic inference | The agents misread the population, test or aim |
| missing information in abstract | The answer needs the full text |
| gold label questionable | On reflection the agents look right |
| other | Explain in notes |

A second sheet explains each category. `summarize` counts categories overall and per field (`error_summary.csv`).

## Reviewer weights (protocol step 5)

```bash
python -m adrd_review.eval.weights data/runs/E2_gemini/<calibration timestamp>
# then in a run config:   weights_file: configs/weights/E2_gemini.json
```

Weight per reviewer = log(accuracy / (1 − accuracy)) of its independent screening votes on the **calibration** set, accuracy clipped to 0.55–0.95, rescaled to mean 1. This is the optimal weighting for a majority vote of independent voters with known accuracy (Nitzan & Paroush, 1982). Refuses test-set runs. The weights file is included in the freeze fingerprint.

Fake demonstration (E2_fake, calibration): accuracies A 0.60, B 0.80, C 0.35 → weights 0.61, 2.09, 0.30; weighted precision 0.435 → 0.667, κ 0.240 → 0.562, recall unchanged. **This gain is in-sample** (weights learned and measured on the same 40 articles); the test set shows the honest effect.

## Cost estimate (task 8.4)

```bash
# before real runs: price a fake run's call pattern as the Gemini models
python -m adrd_review.eval.cost data/runs/E4_fake/<timestamp> --as-config configs/runs/E4_gemini.yaml --articles 228 --repeats 3
# after a real calibration run: measured tokens
python -m adrd_review.eval.cost data/runs/E4_gemini/<timestamp> --articles 228 --repeats 3
```

Prices: `configs/prices.yaml` (Gemini paid tier, checked 2026-09-30; Flash 3.6–3.8 double on 2027-01-01). Fake runs record realistic *input* sizes but tiny outputs, so their calls are priced with 400 output tokens each (rationale, quotes and Gemini thinking tokens).

Pre-run estimate for the test set (228 articles × 3 repeats), from fake runs:

| E1 | E2 | E3 | E4 | D | Sum |
|---|---|---|---|---|---|
| $4.21 | $12.54 | $16.19 | $16.39 | $23.78 | $73 (upper bound) |

The sum double-counts calls shared across conditions (E4 ≈ E3's calls; D reuses the panel's calls), so expect roughly **$30–40**. Re-estimate from a real calibration run before the freeze.

**`gemini-3.1-pro-preview` (adjudicator) has no free tier.** On a free key, set the adjudicator to `gemini-3.8-flash` (noted at the top of the E3/E4/D Gemini configs).

## Retrieval check (task 9.3)

```bash
python -m adrd_review.eval.retrieval            # saved search results
python -m adrd_review.eval.retrieval --rerun    # run the professor's query again first
```

Recall of the professor's PubMed search against the gold PMIDs (all articles, and gold includes), with a likely reason for every missed include: published outside the query's date range, not English, or no abstract. Writes `data/results/retrieval_check.json` and `retrieval_missed.csv`.

On the placeholder gold: 0 of 90 includes found, all because they were published 1971–2014, outside the query's 2015–2025 range. This is expected (different topic and era) and confirms the tool explains misses correctly; the real number comes with the professor's PMIDs.
