# Reviewer panel + consensus (E2)

Code: `src/adrd_review/graph/pipeline.py` (panel wiring), `src/adrd_review/consensus.py` (combining votes) · Configs: `configs/runs/E2_*.yaml` · Built: 2026-09-30 (Day 5)

## 1. Purpose

Runs **three independent reviewer agents** on every article and combines their answers, mirroring the human protocol of three independent reviewers followed by reconciliation. It addresses the earlier system's single-annotator weakness on *nuanced contextual judgment*: one agent's blind spot is outvoted, and disagreement becomes a visible signal instead of a hidden error. E2 settles disagreements with a weighted majority vote; from Day 6 the adjudicator takes over split cases.

## 2. Where it sits

```
triage -> [screen_reviewer A | B | C] -> screen_consensus
       --include--> extract -> [annotate_reviewer A | B | C] -> annotate_consensus -> END
       --exclude--> END
```

The reviewers are the [screening reviewer](screening-reviewer.md) and [annotation reviewer](annotation-reviewer.md) agents, each run once per panel member with that member's persona and model. E1 is the same graph with one reviewer.

## 3. Inputs

Each reviewer receives only the article, the codebook and its own persona. **Reviewers never see each other's votes** (independent round). This keeps them independent, like the human reviewers, and keeps agreement statistics meaningful.

## 4. Output

- `screening_votes`: one `Vote` per reviewer.
- `screening_decision`: a `Decision` with `labels`, `method` (how it was reached), `confidence` (share of reviewer weight behind it) and `include_score` (0–1, used to rank articles for WSS@95).
- `annotation_votes`: one `Vote` per reviewer per dimension.
- `annotation_decisions`: one `Decision` per dimension.

## 5. The panel (personas and models)

| Reviewer | Perspective (persona line in the prompt) | Gemini model (`E2_gemini`) |
|---|---|---|
| A | Clinical neurologist: is the target condition, test and reference diagnosis clinically sound? | `gemini-3.8-flash` |
| B | Epidemiologist: who was studied, where recruited, does sample and setting match? | `gemini-3.7-flash` |
| C | Research methodologist: study design, data sources, can accuracy be assessed? | `gemini-3.5-flash` |

The persona is one line ("Your perspective: …") in the reviewer's system prompt; the rules and evidence requirements are identical for all three. Model and persona are set per reviewer in the run config:

```yaml
panel:
  - id: A
    persona: a clinical neurologist who diagnoses dementia. ...
    model: {provider: google, model: gemini-3.8-flash, requests_per_minute: 10}
    annotation_model: ...   # optional; defaults to `model`
    weight: 1.0             # optional; see weights below
```

## 6. How votes are combined (`consensus.py`)

**Screening** (`combine_screening`):

| Situation | Result | `method` |
|---|---|---|
| One reviewer | Its vote | `single` |
| All agree | That answer | `unanimous` |
| Weighted majority | The majority answer | `majority` |
| Exact tie | **Include** (never silently drop a study) | `tie_include` |

`include_score` = weighted mean of each reviewer's probability of include (its confidence if it voted include, 1 − confidence if exclude).

**Annotation** (`combine_labels`, per dimension):
- Multi-label dimensions keep every label backed by **at least half** of the reviewer weight.
- Single-label dimensions take the label with the most weight; "no label" wins if more weight chose nothing; ties go to the label listed first in the codebook.
- All reviewers giving the same set → `unanimous`.

**Weights** default to 1 per reviewer. They are meant to be set from each reviewer's accuracy on the **calibration** set (never the test set) and written into the config, so the weighting is visible and reproducible.

## 7. Checks

Each reviewer's own checks still apply (unknown rules/labels dropped, quotes searched for). The pipeline also refuses an empty panel or duplicate reviewer IDs. Reviewers sharing a model share one rate limiter, so a persona-only panel cannot exceed the model's request limit.

## 8. Failure handling

If any reviewer fails after its retries, the article fails as a whole (listed in `failures.json`, predicted `include`). Partial panels are not combined, so every decision rests on the full panel.

## 9. Design decisions

- **Independent first, combine second.** Prior work: most of a multi-agent panel's gain comes from combining independent answers, and debate adds little on average while agents drift toward the majority ([Debate or Vote, 2025](https://arxiv.org/abs/2508.17536); [multi-LLM abstract screening, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12952526/)). Debate is tested separately as variant D (Day 8).
- **Gemini-only diversity.** With a Gemini licence, variety comes from three Gemini versions plus three perspectives. The same prior work found the largest gains from *different model families*; the `E2_mixed` config (Gemini + a free Groq model) tests exactly that once a Groq key exists. See [../MODELS.md](../MODELS.md).
- **Ties go to include.** A split must never become a silent exclusion (recall guard).
- **Per-dimension combining.** Reviewers can agree on the instrument but not the setting; only the disputed dimension is uncertain, which is what the adjudicator will look at on Day 6.

## 10. Tests

`tests/test_consensus.py`: single, unanimous, majority, weighted flip, tie → include, include score; multi-label half-weight rule, codebook ordering, unanimous empty; single-label plurality, "no label" winning, tie order; the 3-reviewer graph runs three independent reviewers and takes the majority; Fleiss' κ among reviewers. `tests/test_agents_pipeline.py`: E1 as a panel of one; configs parse (E2 has three distinct Gemini versions).

## 11. How to run and inspect it

```bash
python -m adrd_review.graph.runner configs/runs/E2_fake.yaml                # no key
python -m adrd_review.graph.runner configs/runs/E2_gemini.yaml --limit 3    # Gemini
```

The run prints **panel agreement** (Fleiss' κ among the reviewers' independent votes and the number of unanimous articles, per field) and **how decisions were reached** (counts of unanimous / majority / tie). Both are in `report.json` (`panel_agreement`, `decision_methods`); every vote is in `decisions.jsonl`.

Fake run, 40 calibration articles: reviewers disagreed on 20 of 40 articles (screening Fleiss' κ 0.300), 252 calls, 0 failures. Its screening scores equal the keyword baseline because the three fake styles are nested, so the majority always equals the middle rule; this confirms the voting, not agent quality.

## 12. Limitations and planned changes

- Weights are all 1 until real calibration runs exist.
- Day 6: split votes go to the adjudicator instead of a plain majority, and a split-vote exclusion goes to a human.
- Panel size (3/5/7) is an experiment factor; the code already accepts any size.

## 13. Change log

- 2026-09-30: built (Day 5).
