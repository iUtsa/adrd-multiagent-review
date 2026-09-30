# Human review (E4)

Code: `src/adrd_review/agents/human.py`, pipeline nodes `screen_human` and `final_human`, command `src/adrd_review/human_review.py` · Built: 2026-09-30 (Day 7)

## 1. Purpose

Sends the decisions the system is unsure about to a **person**, and uses the person's answer. This is the paper's human-in-the-loop protocol: humans spend time only where agents are unsure, and the number of escalations measures that human effort.

## 2. Where it sits

- `screen_human`: right after screening, so a person can rescue a study the panel would exclude; if the person says include, the article continues to extraction and annotation.
- `final_human`: after the verifier, for annotation escalations and verifier failures.

## 3. What gets escalated

| Reason | When |
|---|---|
| `split_exclusion` | **Screening safety rule**: the final screening answer is exclude but the reviewers disagreed. A split must never become a silent exclusion. |
| `low_confidence` | The adjudicator's confidence is below `escalate_below` (default 0.6), or it contradicted itself |
| `verifier_fail` | The verifier found a decision unsupported |

Escalations are recorded in every condition (E2 and E3 too), so each report shows how many decisions *would* need a person. Only E4 answers them. A human's answer on a field is final: the same field is not asked twice, and annotation escalations on an excluded article are dropped.

## 4. Two ways to answer (`human:` in the run config)

| Mode | What happens | Use |
|---|---|---|
| `gold` | The gold-standard label stands in for the person | Scoring E4 on calibration/test without new human work. Results are an **upper bound** (a perfect reviewer); the escalation count is the human effort. |
| `queue` | The run **really pauses** on each escalated decision (LangGraph `interrupt` + SQLite checkpoint). A person answers, and the run resumes from where it stopped. | Real use, and demonstrating the workflow |
| `none` | Escalations recorded, not answered | E2, E3, D |

## 5. The review screen (`queue` mode)

```bash
python -m adrd_review.graph.runner configs/runs/E4_queue_fake.yaml --limit 12
python -m adrd_review.human_review data/runs/E4_queue_fake/<timestamp>
```

For each paused decision the terminal shows: PMID and title, which decision is needed and why, the current answer, the abstract, and each reviewer's answer with reasoning and quotes. It accepts only valid answers (exactly one for screening; several comma-separated, or `none`, for multi-label dimensions). When every item is answered, it writes `predictions.csv` and `report.json` for the run. The checkpoint file (`checkpoints.sqlite`) lives in the run folder, so answering can stop and continue later.

## 6. Output

A `Decision` with `method: human` (confidence 1). Each answer is logged in `human_answers` (field, reason, labels, mode) in `decisions.jsonl`; `report.json` counts answered escalations.

## 7. Checks

Answers are validated against the codebook before they are accepted. The checkpointer only reads back our own registered record types (LangGraph's strict deserialization), so paused runs keep working after LangGraph upgrades.

## 8. Failure handling

An article that errored is not paused; it is listed in `failures.json` and predicted `include`. Paused articles are not scored until answered.

## 9. Design decisions

- **Screening review happens before extraction**, so a rescued study is fully annotated.
- **Gold mode for scoring** follows the evaluation plan (task 7.2): it isolates what human review can add without requiring new human labelling, and reports human effort as escalations.
- **Real pauses use LangGraph's built-in interrupt**, the standard human-in-the-loop mechanism, rather than a custom queue.

## 10. Tests

`tests/test_adjudication_debate_human.py`: gold answers escalations and a rescued include continues to annotation; a queued run pauses on a split exclusion, resumes with `Command(resume=...)`, and ends with a human decision.

## 11. Results so far (fake models, calibration)

| Run | Escalated | Answered | Effect |
|---|---|---|---|
| E3_fake | 12 of 40 articles (30%), all `split_exclusion` | 0 | recorded only |
| E4_fake (gold) | 12 of 40 (30%) | 12 | recall 0.769 → 0.846 (one more relevant study rescued) |
| E4_queue_fake (12 articles) | 3 | 3, answered via the review command | run paused, resumed and finished |

## 12. Limitations and planned changes

- A verifier failure on screening at the final step can flip exclude → include after annotation was skipped; such an article is predicted include without labels (rare; noted).
- A web review screen through the existing Django REST app is possible later; the terminal screen covers the study's needs.

## 13. Change log

- 2026-09-30: built (Day 7).
