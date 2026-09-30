# Annotation reviewer

Code: `src/adrd_review/agents/annotation.py` · Exact prompt: [prompts/annotation.md](prompts/annotation.md) · Built: 2026-09-30 (Day 4)

## 1. Purpose

Labels each included article on every codebook **dimension** (currently `instrument` and `setting`), using only the label IDs the codebook lists, with quotes and a short rationale per dimension. This is the job the earlier Annotation Agent struggled with (nuanced judgment, domain knowledge, borderline cases). From E2 on it runs as three independent reviewers whose labels are combined per dimension.

## 2. Where it sits

`extraction → annotation → END`. In panel runs, three copies run in parallel and the consensus step combines their labels dimension by dimension ([reviewer-panel.md](reviewer-panel.md)).

## 3. Inputs

- The rendered article.
- The extraction agent's facts, shown as "check them against the article".
- Every dimension from the codebook: its question, whether several labels are allowed, and each label ID with its definition.
- Optionally a persona. The evidence rules.

## 4. Output: `AnnotationAnswer` → one `Vote` per dimension

| Field | Meaning |
|---|---|
| `annotations[]` | One entry per dimension: `dimension`, `labels` (label IDs), `evidence` (quotes), `rationale` |
| `confidence` | 0–1, overall |

The code turns this into one `Vote` per codebook dimension (`field` = dimension name).

## 5. Prompt

System message: role, optional persona, "use only the listed label IDs, spelled exactly; leave empty if nothing is supported", every dimension with its labels, the evidence rules. User message: the rendered article plus extracted facts. Exact text: [prompts/annotation.md](prompts/annotation.md).

## 6. Model and settings

| Run | Model |
|---|---|
| E1 on Gemini | `gemini-3.8-flash`, temperature 0, max 2,048 tokens |
| Panel on Gemini | See [reviewer-panel.md](reviewer-panel.md) |
| No-key runs | `fake` (instrument and setting keyword tables) |

## 7. Checks after the model answers

- Structured output must match `AnnotationAnswer`; malformed answers are retried.
- **Labels not in the codebook are dropped** (warning). Duplicates are removed.
- **Single-label dimensions keep only the first label** (warning).
- **A dimension the model skipped** gets an empty vote (warning).
- **Quotes are searched for** in the title + abstract (warning if not found).

## 8. Failure handling

3 attempts, then the article fails as a whole (listed in `failures.json`, predicted `include` with no labels).

## 9. Design decisions

- **One call for all dimensions.** Fewer calls and the model sees the whole picture; the consensus step still combines each dimension separately, so only disputed dimensions need the adjudicator later.
- **Strict label vocabulary.** Labels must match the gold standard's values exactly to be scored; invented labels are dropped visibly rather than mapped by guesswork.
- **Extraction facts as hints, not answers.** Reduces missed details while keeping the article as the authority.
- **Scored end to end.** If screening wrongly excluded a gold-include article, its labels count as empty (see [EVALUATION.md](../EVALUATION.md)).

## 10. Tests

`tests/test_agents_pipeline.py`: every label appears in the prompt; unknown and duplicate labels dropped with a warning; missing dimension flagged; included article gets instrument and setting decisions.

## 11. How to run and inspect it

Part of any run. Votes are in `decisions.jsonl` (`annotation_votes`); final labels in `predictions.csv`.

## 12. Limitations and planned changes

- On the placeholder gold standard, annotation labels are derived and positives-only, so annotation scores are a pipeline check, not a result.
- Dimensions change with the professor's codebook (likely AI method, data modality, dataset source, population).

## 13. Change log

- 2026-09-30: built (Day 4). Panel use added (Day 5).
