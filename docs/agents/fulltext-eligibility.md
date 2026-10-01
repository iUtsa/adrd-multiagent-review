# Full-text eligibility reviewer (stage 2)

Code: `src/adrd_review/agents/eligibility.py`, pipeline nodes `get_fulltext`, `elig_*` · Full-text sources: `src/adrd_review/tools/fulltext.py`, `tools/pdf.py`, `tools/pmc.py` · Exact prompt: [prompts/eligibility.md](prompts/eligibility.md) · Built: 2026-10-01

## 1. Purpose

Judges an article against every codebook rule using the **whole paper**, the way a human reviewer does at full-text assessment. It exists because the human gold standard reflects decisions made from full papers, while abstracts often do not contain the answer (reference standard, design, setting). It turns the pipeline into the standard two-stage review: lenient abstract screening, then strict full-text eligibility.

## 2. Where it sits

```
stage 1 screening --include--> get_fulltext --> [elig_reviewer x N] -> elig_consensus -> (debate) -> (adjudicator)
        -> elig_checks -> (human) --include--> extraction -> annotation (both read the full paper)
                                  --exclude--> end
```

Switched on with `eligibility: fulltext` in a run config (`E3_twostage_*`, `E4_twostage_*`). The same reviewer panel, personas, weights, debate variant, adjudicator, verifier and human review apply to this stage as to stage 1.

## 3. Inputs

- The article **with its full text** (section headings kept, e.g. `[METHODS] …`), up to `eligibility_max_chars` (default 80,000 characters: the whole paper for almost every article; the longest of 57 sampled was 67,100).
- The inclusion and exclusion rules from the codebook (without the abstract-stage "benefit of the doubt" rule).
- Optional persona; the evidence rules. The instructions are **repeated after the paper**, because models attend best to the start and end of long inputs.

## 4. Where the full text comes from (`tools/fulltext.py`)

Tried in this order; the source is recorded per article in `fulltext_info`:

| # | Source | Notes |
|---|---|---|
| 1 | **The team's PDF**: `data/pdfs/<PMID>.pdf` | What the human reviewers read. Text is extracted with `pypdf` and must pass a quality check (enough text, mostly words, words not run together); the reference list is dropped. A failed PDF falls through to the next source. |
| 2 | **PubMed Central** | Structured XML, free. About 61% of the professor's 1,855 search results. |
| 3 | **Open-access PDF** | Downloaded only when Europe PMC marks the link "Open access". In practice most such links are publisher pages or bot-protected, so this rarely succeeds. |
| 4 | **None** | The article is **not excluded**. It is flagged (`no_fulltext`), sent to a human, and listed in the run's `pdfs_needed.csv` with a link and the exact file name to save the PDF as. Re-running after adding PDFs picks them up; finished calls come from the cache. |

About a third of the professor's search results are paywalled (measured; see [../FULLTEXT_PLAN.md](../FULLTEXT_PLAN.md)), so the team's PDFs matter.

## 5. Output: per-rule judgments → `decide()` → a `Vote` on field `eligibility`

The model fills the same form as stage 1 (`ScreeningAnswer`: each inclusion rule met / not_met / unclear, each exclusion rule applies / does_not_apply / unclear, with quotes). The decision rule is stricter:

| Situation | Decision |
|---|---|
| An inclusion rule clearly **not met**, or an exclusion rule clearly **applies**, with a quote found in the paper | **exclude** |
| **Every** inclusion rule **met** (with quotes) and no exclusion applies | **include** |
| Anything else: the paper does not settle a rule | **unclear** → a human decides |

Panel answers are combined by `combine_eligibility`: unanimous or a weighted majority decides; with no majority the result is `unclear`. From E3, split panels go to the adjudicator.

## 6. Model and settings

| Run | Model |
|---|---|
| `E3_twostage_gemini`, `E4_twostage_gemini` | The Gemini panel (3.8 / 3.7 / 3.5 Flash); adjudicator Gemini 3.1 Pro |
| No-key runs | `fake`: with the full paper it also judges I3 (accuracy terms) and I4 (setting words) |

A full-text call is about **11,700 tokens**. Groq's free tier (8,000 tokens/min) cannot run this stage; the preflight check refuses such a config.

## 7. Checks

- All stage-1 checks: missing judgments → unclear; unknown rule IDs dropped; **not_met / applies without a quote that appears in the paper → unclear**. Quotes are checked against title + abstract + full text.
- Escalations raised here: `no_fulltext`, `unclear_fulltext`, `low_confidence` (adjudicator), `split_exclusion` (reviewers disagreed and the result is exclude).
- The verifier checks the eligibility decision's quotes and rules, except decisions made by a human.

## 8. Failure handling

- Full-text fetching never raises: a source that errors is skipped and the reason recorded.
- Without full text the agent is not called; the article stays in and is flagged.
- An unresolved `unclear` with no human in the run is treated as **include** in the predictions (and counted as an escalation), so nothing is dropped for lack of an answer.
- Model errors: retries as for every agent, then the article is listed in `failures.json` and predicted include at both stages.

## 9. Design decisions

- **Two stages, lenient then strict.** Mirrors the human process and resolves the problem seen in the first real run: models are strict by nature, which is wrong when information is missing (abstract) and right when it is present (full text).
- **Evidence for LLM full-text screening:** pooled sensitivity and specificity about 0.99 from full text versus 0.92 / 0.94 from abstracts ([meta-analysis, 2026](https://www.medrxiv.org/content/10.64898/2026.03.17.26348656v1.full); only 4 full-text studies, so less certain).
- **Whole paper, not chunks.** Papers here are about 10,000–17,000 tokens, far below Gemini's limits, so retrieval or chunking would add failure modes for no gain. When a model's limit is smaller, `fit_text` puts Methods, Participants and Results first.
- **Unclear goes to a human rather than to exclude.** A reviewer who cannot settle a rule from the full paper has found a genuinely hard case.
- **Team PDFs are preferred over PubMed Central** when both exist, so agents read what the humans read.

## 10. Tests

`tests/test_two_stage.py`: include needs every rule met; a clear violation quoted from the paper excludes (and does not when only the abstract is available); the agent refuses to run without full text; instructions repeated after the paper; three-way consensus; source order (team PDF, PubMed Central), bad PDF falls back, misses explained with a link; `fit_text` whole vs methods-first; PDF quality check and section split; graph paths (full text found; none → flagged and kept; human decides; stage-1 excludes never fetch); scoring of both stages; preflight.

## 11. How to run and inspect it

```bash
python -m adrd_review.preflight configs/runs/E4_twostage_gemini.yaml     # check first
python -m adrd_review.graph.runner configs/runs/E4_twostage_fake.yaml    # no key
python -m adrd_review.tools.fulltext 23172761                            # which source would this PMID use?
python -m adrd_review.tools.pdf data/pdfs/23172761.pdf                   # what does this PDF extract to?
```

Each run reports **FULL-TEXT ELIGIBILITY** scores (against the gold `fulltext` column), where each paper's text came from, and writes `pdfs_needed.csv`.

Fake two-stage run, 40 calibration articles (placeholder; old papers): 23–24 reached stage 2; 1 full text from PubMed Central, 1 from an open-access PDF, 21–22 with none (flagged; in E4 answered by the gold-standard "human": stage-2 precision 1.00, κ 0.90).

## 12. Limitations and planned changes

- **Not yet run with a real model** (needs Gemini access).
- Tables and figures in PDFs are not interpreted; text only.
- PDF section headings are recognised by a simple list; unrecognised layouts keep the text as one block.
- The placeholder corpus is 1971–2014 and rarely has free full text, so real exercise of this stage waits for the professor's articles and the team's PDFs.

## 13. Change log

- 2026-10-01: built.
