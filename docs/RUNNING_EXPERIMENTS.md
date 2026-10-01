# Running the experiments

Everything is built. This is the order to run it once model access is set up, and the one command per step.

## The conditions

| ID | Config (Gemini / no key) | What it adds |
|---|---|---|
| E0 | (published results of the old 2-agent system) | baseline |
| E1 | `E1_gemini.yaml` / `E1_fake.yaml` | one reviewer |
| E2 | `E2_gemini.yaml` / `E2_fake.yaml` | 3-reviewer panel + weighted vote |
| E3 | `E3_gemini.yaml` / `E3_fake.yaml` | + adjudicator on splits + verifier |
| **E4** | `E4_gemini.yaml` / `E4_fake.yaml` | + human review of escalations (**proposed system**) |
| D | `D_gemini.yaml` / `D_fake.yaml` | E3 + up to 2 debate rounds before the adjudicator |

`E4_queue_fake.yaml` demonstrates real pausing for a person (see [agents/human-review.md](agents/human-review.md)).

## Two-stage (full-text) configs

| Config | What it is |
|---|---|
| `E3_twostage_gemini.yaml` / `E3_twostage_fake.yaml` | E3 with a full-text eligibility stage |
| **`E4_twostage_gemini.yaml`** / `E4_twostage_fake.yaml` | E4 with a full-text eligibility stage: **the proposed system if the human team judged from full papers** |

Put the team's PDFs in `data/pdfs/<PMID>.pdf` first. Papers without full text are flagged, not excluded, and listed in `pdfs_needed.csv`. See [agents/fulltext-eligibility.md](agents/fulltext-eligibility.md).

## Before every real run: preflight

```bash
python -m adrd_review.preflight configs/runs/E4_twostage_gemini.yaml
```

Checks keys, that every model answers a test call, that articles are downloaded, where each full text will come from, that calls fit each model's limits, estimated calls / time / cost, and (for the test set) the freeze. It prints READY or the lines to fix.

## The protocol

**1. Quick check (3 articles, real model)**
```bash
python -m adrd_review.graph.runner configs/runs/E1_gemini.yaml --limit 3
```

**2. Calibration runs (40 articles), all conditions**
```bash
for c in E1 E2 E3 E4 D; do python -m adrd_review.graph.runner configs/runs/${c}_gemini.yaml; done
python -m adrd_review.eval.score --pred data/runs/E3_gemini/<timestamp>/predictions.csv   # re-score any run
```

**3. Codebook refinement (professor approves)**
```bash
python -m adrd_review.curate data/runs/E3_gemini/<timestamp> --model google:gemini-3.1-pro-preview
```
Accepted changes go into `configs/codebook.yaml` with a new version; repeat step 2.

**4. Decide on debate (task 8.2):** keep D only if it beats E3 on calibration.

**5. Set reviewer weights (optional):**
```bash
python -m adrd_review.eval.weights data/runs/E2_gemini/<calibration timestamp>
```
then add `weights_file: configs/weights/E2_gemini.json` to the panel configs.

**5b. Estimate the cost of the test runs:**
```bash
python -m adrd_review.eval.cost data/runs/E4_gemini/<timestamp> --articles 228 --repeats 3
```

**6. Freeze (task 8.3)**
```bash
python -m adrd_review.freeze create configs/runs/E1_gemini.yaml configs/runs/E2_gemini.yaml configs/runs/E3_gemini.yaml configs/runs/E4_gemini.yaml configs/runs/D_gemini.yaml
git add configs/frozen && git commit -m "Freeze for test runs" && git tag v1.0-frozen && git push --tags
```
The freeze records hashes of the config, codebook, agent code (prompts), gold file and split. **The runner refuses `--part test` unless these still match.**

**7. Test runs (228 articles), 3 repeats each**
```bash
for rep in 1 2 3; do for c in E1 E2 E3 E4 D; do
  python -m adrd_review.graph.runner configs/runs/${c}_gemini.yaml --part test --repeat $rep
done; done
```
`--repeat` makes each repeat call the model again; within one repeat, conditions share identical calls (see [ANALYSIS.md](ANALYSIS.md)).

**8. Results, figures, error analysis, retrieval**
```bash
python -m adrd_review.eval.aggregate --part test
python -m adrd_review.eval.figures conditions --part test
python -m adrd_review.eval.figures effort data/runs/E3_gemini/<timestamp> --mark data/runs/E4_gemini/<timestamp>
python -m adrd_review.eval.errors export data/runs/E4_gemini/<timestamp>
python -m adrd_review.eval.retrieval
```

## What each report contains

Scores (recall, precision, F1, κ, WSS, WSS@95; annotation per label), cost and time, **panel agreement** (Fleiss' κ among reviewers), **how decisions were reached** (unanimous / majority / adjudicator / human), **escalations** (count, rate, reasons, answered), **debate statistics** (changes toward/away from gold), failures and warnings.

## Results so far (fake keyword model, calibration, 40 articles): plumbing only

| Run | Recall | Precision | κ | Adjudications | Escalated | Model calls |
|---|---|---|---|---|---|---|
| E1_fake | 0.769 | 0.435 | 0.240 | 0 | 0 | 126 |
| E2_fake | 0.769 | 0.435 | 0.240 | 0 | 12 recorded | 252 |
| E3_fake | 0.769 | 0.435 | 0.240 | 20 | 12 (30%) | 312 |
| E4_fake (gold human) | 0.846 | 0.458 | 0.299 | 20 | 12, all answered | 316 |
| D_fake | 0.769 | 0.435 | 0.240 | 20 | 12 (30%) | 432 |

These only show that every step runs and is counted correctly; the fake model is keyword matching.

## First real-model result (Groq free tier, calibration, 40 articles)

| Run | Codebook | Recall | Precision | Specificity | κ | Notes |
|---|---|---|---|---|---|---|
| E2_groq | v0.1.0 | 0.000 (0/13) | – | 1.000 | 0.000 | Excluded everything: reviewers treated "not reported" as "not met". Led to the per-rule redesign and codebook v0.2.0 ([PROGRESS.md](PROGRESS.md)). |
| E2_groq | v0.2.0 | pending | | | | Re-run after the fixes: `python -m adrd_review.graph.runner configs/runs/E2_groq.yaml` |
