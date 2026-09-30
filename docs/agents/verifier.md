# Verifier

Code: `src/adrd_review/agents/verifier.py` · Exact prompt: [prompts/verifier.md](prompts/verifier.md) · Built: 2026-09-30 (Day 6)

## 1. Purpose

Checks that every final decision is backed by real evidence before it is accepted. It supports the paper's requirements for **transparent annotation logic** and **human-in-the-loop verification**, and catches made-up evidence. It never changes a decision; a failed decision is sent to a human.

## 2. Where it sits

Last automatic step for every article (`finalize`), after screening (excluded articles) or after annotation (included articles), before any final human review. Switched on in E3, E4 and D (`verifier:` in the run config).

## 3. Inputs

- The article.
- The final decisions: screening, plus each annotation dimension for included articles.
- For each decision, the **supporting votes**: each reviewer's latest vote on that field that agrees with the decision.

## 4. Output: `VerificationReport`

| Field | Meaning |
|---|---|
| `quote_problems` | Deterministic check: decisions whose supporting quotes are **all** missing from the article |
| `checks` | Model check, one per decision: `field`, `supported` (true/false), `problem` |
| `model` | Model used for the check, or empty if quote check only |

Every failed field becomes an escalation with reason `verifier_fail`.

## 5. Prompt

System: "check whether the quoted evidence says what is claimed and satisfies the cited rules; do not re-review or change decisions", plus the rules of the fields being checked. User: the article, then each decision with its cited rules and quotes. Exact text: [prompts/verifier.md](prompts/verifier.md).

## 6. Model and settings

| Run | Model |
|---|---|
| Gemini runs | `gemini-3.5-flash-lite` (a cheap yes/no check), max 1,024 tokens |
| Quote check only | `verifier: {enabled: true}` with no model: no model call |
| No-key runs | `fake`: fails an include that has no evidence |

## 7. Checks

- **Layer 1 (always, no model):** each decision's supporting quotes are searched for in the title and abstract (ignoring case and spacing). A decision is flagged only if *every* supporting quote is missing.
- **Layer 2 (if a model is set):** the model judges whether the evidence and cited rules support each decision. Only decisions with labels are sent (empty labels have nothing to verify).

## 8. Failure handling

Retries as for every agent; then the article fails as a whole. A verifier failure on a decision is not an error: it is an escalation.

## 9. Design decisions

- **Two layers.** The code check is free and certain for fabricated quotes; the model check catches quotes that exist but do not support the claim.
- **Never edits decisions.** Verification that silently "fixes" answers would hide errors; escalation keeps them visible and countable.
- **Votes are matched by reviewer and field.** Found and fixed during the Day 6 test run: when votes were matched by reviewer only, a reviewer's annotation votes hid its screening vote, and the verifier flagged every included article. A regression test now covers this.

## 10. Tests

`tests/test_adjudication_debate_human.py`: invented supporting quotes flagged; real quotes pass; model check reports failed fields; votes matched by reviewer and field (regression); verification report present in the unanimous path.

## 11. How to run and inspect it

Runs in E3/E4/D. Reports are in `decisions.jsonl` (`verification`); escalations with reason `verifier_fail` are counted in `report.json`. Fake E3 run after the fix: 0 verifier failures on 40 articles.

## 12. Limitations and planned changes

- Quotes are checked against title + abstract only; full text comes with the PMC tool.
- Reviewer-level quote warnings (from each agent) and decision-level verification overlap on purpose: the first records every bad quote, the second decides whether a decision still stands.

## 13. Change log

- 2026-09-30: built (Day 6); fixed vote matching by field during the first test run.
