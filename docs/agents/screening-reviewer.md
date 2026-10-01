# Screening reviewer

Code: `src/adrd_review/agents/screening.py` · Exact prompt: [prompts/screening.md](prompts/screening.md) · Built: 2026-09-30 (Day 4); redesigned the same day after the first real-model run

## 1. Purpose

Judges an article against **each codebook rule separately**, with an exact quote for every judgment. The code, not the model, then turns those judgments into include or exclude. It replaces the screening part of the earlier Extraction Agent. Used once in E1 and three times (different perspectives and models) in the reviewer panel from E2 on.

## 2. Where it sits

`triage → screening → (include) extraction → annotation`, or `→ (exclude) END`. In panel runs, three copies run in parallel and their votes go to the consensus step ([reviewer-panel.md](reviewer-panel.md)).

## 3. Inputs

- The rendered article (title, journal/year, publication type, language, MeSH terms, abstract, and a PubMed Central excerpt when full text is on).
- The review question, inclusion rules, exclusion rules and decision rule, all rendered from the codebook.
- Optionally a **persona** (a perspective such as "clinical neurologist"), set in the run config.
- The evidence rules: quote word for word, never paraphrase.

## 4. Output: `ScreeningAnswer` → `decide()` → a `Vote`

The model fills in:

| Field | Meaning |
|---|---|
| `inclusion[]` | One entry per inclusion rule: `rule_id`, `status` = **met / not_met / unclear**, `quote` |
| `exclusion[]` | One entry per exclusion rule: `rule_id`, `status` = **applies / does_not_apply / unclear**, `quote` |
| `rationale` | At most three sentences |
| `confidence` | 0–1, in its rule judgments |

There is **no include/exclude field**. `decide()` applies the title/abstract screening rule:

| Situation | Decision |
|---|---|
| Any inclusion rule **not_met**, or any exclusion rule **applies**, backed by a quote that really appears in the article | **exclude** (those rules are recorded as the reason) |
| Everything else, including any number of **unclear** | **include** (benefit of the doubt) |

The stored `Vote` has the decision, the deciding rule IDs, the verified quotes, the rationale, a confidence, any warnings, and **`rule_checks`**: the status of every rule, so each decision can be read rule by rule in `decisions.jsonl`.

Confidence: for an include, 0.5 + 0.5 × (share of inclusion rules clearly met), so articles that clearly meet more rules rank higher; for an exclude, the model's own confidence (at least 0.5).

## 5. Prompt

System message: role, optional persona, "you do not decide include or exclude; you judge each rule", the rendered rules, a precise definition of each status with the instruction that **something the abstract simply does not mention is unclear, never not_met**, the requirement of a quote for met / not_met / applies, and the evidence rules. User message: the rendered article. Exact text: [prompts/screening.md](prompts/screening.md).

## 6. Model and settings

| Run | Model |
|---|---|
| E1 on Gemini | `gemini-3.8-flash`, temperature 0, max 2,048 tokens |
| Panel on Gemini / Groq / mixed | See [reviewer-panel.md](reviewer-panel.md) and [../MODELS.md](../MODELS.md) |
| No-key runs | `fake`: keyword rules expressed as rule judgments (I1 needs a listed instrument, I2 needs dementia; the strict style also needs an accuracy term for I3) |

## 7. Checks after the model answers (all in `decide()`)

- A rule the model did not judge is treated as **unclear** (warning).
- Rule IDs that are not in the codebook are dropped (warning).
- **A not_met or applies without a quote that really appears in the article is downgraded to unclear** (warning). An exclusion must rest on real evidence; a hallucinated reason cannot drop a study.
- A met with a quote that is not found keeps its status but is flagged (warning).
- Every found quote becomes part of the vote's evidence.

## 8. Failure handling

As for every agent: up to 5 attempts (rate limits back off 15–60 s), then the article is listed in `failures.json` and predicted `include`.

## 9. Design decisions

- **Rule judgments instead of a verdict (added 2026-09-30).** In the first real-model run (three Groq models, 40 calibration articles) reviewers asked for include/exclude excluded **all 13 gold includes**. Their reasoning was mostly "the abstract does not report X, so the rule is not met", the opposite of the codebook's benefit-of-the-doubt rule. Asking for per-rule status and deciding in code makes that rule mechanical, and matches the per-criterion question-answering framing used in multi-LLM abstract screening ([Akinseloyin et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12952526/)).
- **Exclusions need real evidence.** Tying not_met/applies to a verified quote protects recall against invented reasons.
- **Transparent.** `rule_checks` shows exactly which rule decided each exclusion, which feeds the error analysis and the Codebook Curator.
- **Confidence is derived, not just self-reported.** The share of clearly met rules is a more usable ranking signal than a model's stated confidence.
- **Personas are optional prompt lines**, so the same agent code serves every reviewer in the panel.

## 10. Tests

`tests/test_agents_pipeline.py`: unclear never excludes; a clear violation with a real quote excludes (inclusion and exclusion rules); an exclusion without a real quote becomes unclear; missing and unknown rules handled; prompts contain every rule and the persona; the graph's include and exclude paths.

## 11. How to run and inspect it

```bash
python -m adrd_review.graph.runner configs/runs/E1_fake.yaml        # no key
python -m adrd_review.graph.runner configs/runs/E2_groq.yaml        # real models (Groq free tier)
```

Per article, `decisions.jsonl` → `screening_votes[]` shows each reviewer's `rule_checks`, quotes, rationale and warnings.

## 12. Limitations and planned changes

- **Not yet re-run on real models after the redesign** (the re-run needs network). Expect recall to rise and precision to fall compared with the first real run; the calibration run will show by how much.
- "Unclear → include" is right for title/abstract screening. If the professor's protocol screens differently, the rule lives in one function (`decide`).
- Quotes are checked against title + abstract (+ full-text excerpt when used), not MeSH terms.

## 13. Change log

- 2026-09-30: built (Day 4). Panel use added (Day 5).
- 2026-09-30: redesigned to per-rule judgments with the decision made in code, after the first real-model run excluded every gold include.
