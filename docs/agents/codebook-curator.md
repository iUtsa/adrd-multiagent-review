# Codebook Curator

Code: `src/adrd_review/agents/curator.py` (agent), `src/adrd_review/curate.py` (command) · Exact prompt: [prompts/curator.md](prompts/curator.md) · Built: 2026-09-30 (Day 7)

## 1. Purpose

Proposes clearer codebook wording and borderline examples, based on where the agents disagreed with the human gold standard on the **calibration set**. It addresses the earlier finding that humans were better at **iterative refinement of criteria**: the Curator does the pattern-finding, and the professor decides.

## 2. Where it sits

Outside the per-article pipeline. It runs on a finished calibration run:

```
calibration run -> curator -> proposals file -> professor accepts/rejects -> codebook v(n+1) -> re-run calibration
```

## 3. Inputs

- The current codebook (rules, dimensions, labels).
- Up to 40 disagreements from the run: PMID, title, start of the abstract, the field, the human answer, the agents' answer, the rules the agents cited, and their reasoning.

## 4. Output: `CuratorAnswer`

3–8 `CodebookProposal`s, each with `target` (rule ID, `dimension:label`, or `new`), `problem`, `suggested_text`, and `example_pmids`. Written to the run folder as:
- `curator_proposals.md`: for the professor, with accept / reject / edit boxes
- `curator_proposals.json`: machine-readable, with model and usage

## 5. Prompt

System: role, the full current codebook, and guidance (look for patterns, prefer small precise edits, do not memorise examples, cite PMIDs). User: the disagreements. Exact text: [prompts/curator.md](prompts/curator.md).

## 6. Model and settings

Chosen on the command line: `--model google:gemini-3.1-pro-preview` (planned), `--model fake:keyword-rules` (no key; placeholder proposals).

## 7. Checks

- **Refuses any run that is not on the calibration set.** The test set must never shape the codebook.
- Nothing is applied automatically; the codebook changes only by a person editing `configs/codebook.yaml` and bumping its version.

## 8. Failure handling

Model errors stop the command with a message; the run folder is unchanged.

## 9. Design decisions

- **Learns only from calibration disagreements.** Keeps the test set clean (see the frozen split).
- **Human approval.** Codebook changes define the review; they are the professor's decision, and the proposals file records what was accepted.
- **Small edits over rewrites.** Research on LLM-human screening disagreement found boundary ambiguity in criteria to be the top cause ([Understanding LLMs in title-abstract screening, 2026](https://arxiv.org/abs/2606.17588)); targeted definitions and examples address that directly.

## 10. Tests

`tests/test_adjudication_debate_human.py`: refuses test-set runs. End to end with the fake model on the E3 calibration run: 26 disagreements → proposals file written.

## 11. How to run and inspect it

```bash
python -m adrd_review.curate data/runs/E3_fake/<timestamp> --model fake:keyword-rules
open data/runs/E3_fake/<timestamp>/curator_proposals.md
```

## 12. Limitations and planned changes

- Proposals from the placeholder codebook are only a demonstration; real use starts with the professor's codebook.
- At most 40 disagreements per call; enough for a 40-article calibration set.

## 13. Change log

- 2026-09-30: built (Day 7).
