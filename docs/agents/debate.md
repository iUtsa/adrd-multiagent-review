# Debate (variant D)

Code: `src/adrd_review/agents/debate.py` (one reviewer's turn), `src/adrd_review/graph/pipeline.py` (`screen_debate`, `annotate_debate`) · Exact prompt: [prompts/debate.md](prompts/debate.md) · Built: 2026-09-30 (Day 8)

## 1. Purpose

Lets reviewers who disagree **respond to each other** before the adjudicator decides. This tests the study's key open question: does agent-to-agent communication help on nuanced ADRD judgment, beyond independent voting? It is an **experimental variant (D)**, compared against E3, and kept only if it wins on the calibration set.

## 2. Where it sits

```
[reviewers] -> consensus --split--> debate round 1 -> consensus --still split--> debate round 2 -> consensus
            --still split--> adjudicator -> ...
```

Up to `debate_rounds` rounds (2 in the D configs), for screening and for each disputed annotation dimension. It stops early as soon as the reviewers agree.

## 3. Inputs (per reviewer, per round)

- The article, the field, its valid answers and rules, and the reviewer's persona.
- The reviewer's own previous answer and reasoning.
- **The arguments for each position**: evidence and reasoning grouped by position, anonymous and de-duplicated, **with no count of how many reviewers hold each position**.

## 4. Output: `DebateAnswer` → a new `Vote` with `round` = 1, 2, …

| Field | Meaning |
|---|---|
| `labels` | Answer after reading the arguments |
| `new_evidence` | Quotes not cited before; required to change the answer |
| `rule_ids`, `rationale` | Must answer the strongest opposing argument |
| `confidence` | 0–1 |

Earlier votes are never overwritten; every round adds a vote. Consensus always uses each reviewer's latest vote.

## 5. The anti-conformity rules

Evidence: when facing an incorrect majority, weaker models corrected themselves only 3.6% of the time, and evidence-based revisions corrected errors far more often than conformity did ([Can LLM Agents Really Debate?, 2025](https://arxiv.org/abs/2511.07784)). So:

| Rule | How it is enforced |
|---|---|
| No visible majority | `render_sides` groups arguments by position, removes duplicates and names no reviewers or counts |
| Minority speaks first | `minority_first` orders the turn; later speakers see the minority's new arguments |
| Change requires new evidence | Code accepts a change only if a new quote is (a) not cited by that reviewer before and (b) actually found in the article; otherwise the previous answer stands (warning `rejected`) |
| History kept | Each round is a new vote; `report.json` counts changes toward vs away from the gold answer |

## 6. Model and settings

Each reviewer debates with its own model and persona (the same as in the panel). `debate_rounds: 2` in `D_*.yaml`. No-key runs: `fake` always keeps its position.

## 7. Checks after the model answers

Invalid labels dropped; screening must have exactly one answer (otherwise previous answer kept); single-label dimensions keep one label; invented quotes flagged; the new-evidence rule above.

## 8. Failure handling

As for every agent: retries, then the article fails as a whole.

## 9. Design decisions

- **Tested, not assumed.** Across benchmarks, debate rarely beats majority voting ([Debate or Vote, 2025](https://arxiv.org/abs/2508.17536)); in abstract screening it reduced useful model diversity ([Akinseloyin et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12952526/)). But it helped weaker models and may help nuanced judgment, the earlier system's failure mode.
- **Only split fields debate.** Unanimous decisions cost nothing extra.
- **Sequential turns within a round.** Needed for "minority first"; rounds are short (at most one call per reviewer per field).

## 10. Tests

`tests/test_adjudication_debate_human.py`: change without new evidence rejected; invented or repeated quotes do not count; genuinely new evidence accepted; minority speaks first; arguments show no counts; in the graph, rounds 1–2 run before the adjudicator and a rejected switch is recorded.

## 11. How to run and inspect it

```bash
python -m adrd_review.graph.runner configs/runs/D_fake.yaml
```

All rounds are in `decisions.jsonl` (`screening_votes`/`annotation_votes` with `round` > 0). `report.json` → `debate`: articles debated, changes, rejected changes, and changes **toward** or **away from** the gold answer (the conformity check). Fake D run: 20 articles debated; the fake keeps its position, so no changes.

## 12. Limitations and planned changes

- A position backed by more reviewers may still show more distinct arguments; de-duplication reduces but cannot remove this hint.
- Decide keep/drop on the calibration set (task 8.2) before the freeze.

## 13. Change log

- 2026-09-30: built (Day 8).
