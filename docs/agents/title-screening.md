# Title screening reviewer (stage 0)

Code: `src/adrd_review/agents/title_screening.py`, consensus `combine_title` in `src/adrd_review/consensus.py`, pipeline nodes `title_reviewer`, `title_consensus` · Stage counts: `src/adrd_review/eval/flow.py` · Exact prompt: [prompts/title.md](prompts/title.md) · Built: 2026-10-03

## 1. Purpose

The professor's team screened in three passes: **titles first, then abstracts, then full papers**. This agent is the first pass. It reads only the title and removes articles whose title alone makes it obvious they are off topic. Everything else goes on to abstract screening. It lets the agent side follow the same process as the human side, so the number of articles at each stage can be compared and reported as a PRISMA flow. It also saves cost on large searches: an article dropped on its title never costs an abstract or full-text call.

## 2. Where it sits

```
START --(title_stage: true)--> [title_reviewer x N] -> title_consensus
            --every reviewer excludes--> finalize (excluded at title; abstract never read)
            --anyone keeps it--> triage -> abstract screening (stage 1) -> full text (stage 2) -> extraction/annotation
START --(title_stage: false)--> triage -> ... (unchanged)
```

Switched on with `title_stage: true` in a run config (`E4_threestage_*`, `PROF_search_*`). Off by default, so every earlier config runs exactly as before.

## 3. Inputs

- **The title only** (plus PMID and year). No abstract, MeSH terms, journal or full text.
- The inclusion and exclusion rules from the codebook, without the abstract-stage decision rule.
- Optional persona (the same reviewer panel and personas as the other stages).

## 4. Output

The same per-rule form as abstract and full-text screening (`ScreeningAnswer`): each inclusion rule met / not_met / unclear, each exclusion rule applies / does_not_apply / unclear, with a quote; then a rationale and confidence. The code turns it into a `Vote` on the field `title`.

**Decision per reviewer** (`screening.decide`, the abstract-stage rule): exclude only if an inclusion rule is clearly not met or an exclusion rule clearly applies, **with a quote that appears in the title**. Otherwise include.

**Panel decision** (`combine_title`): the article is excluded only if **every** reviewer excludes it. One reviewer keeping it is enough (`any_include`). This is the usual rule for human title screening: the first pass removes only the clearly irrelevant.

## 5. Prompt

`SYSTEM` in `title_screening.py`, rendered from the codebook (topic, rules). It says that a title which simply does not mention something is **unclear, never not_met**, and that most rules will be unclear from a title. The exact prompt is generated in [prompts/title.md](prompts/title.md).

## 6. Model and settings

The panel's screening models (`Reviewer.llm`), so one config sets the models for all three stages. A title call is small (about 1,000 tokens including the rules), so it fits every free tier. Groq can run this stage; Gemini is needed only for the full-text stage.

## 7. Checks after the model answers

- Quotes are checked against a **title-only copy** of the article (`title_only`): a not_met or applies that quotes the abstract is not evidence at this stage and counts as unclear.
- Missing judgments count as unclear; unknown rule IDs are dropped with a warning (shared with `screening.decide`).

## 8. Failure handling

- An excluded-at-title article gets a screening decision with method `title_excluded`, so predictions, scoring and the flow treat it as a stage-1 exclusion. The verifier skips it (every reviewer agreed and there is nothing more to check).
- No escalation to a human at this stage: an article in doubt is simply kept, which is the safe direction.
- Model errors: retries as for every agent; then the article is listed in `failures.json` and predicted include.

## 9. Design decisions

- **Unanimous exclusion only.** At title stage the cost of a wrong exclusion (a lost study) is far higher than the cost of a wrong inclusion (one more abstract to read), so one reviewer keeping the article is enough.
- **Quotes from the title only.** Without this rule a reviewer could exclude on information it was not supposed to have.
- **Same answer form and rule as abstract screening.** The behaviour is consistent across stages and the decision is visible rule by rule.
- **Optional stage.** The placeholder gold standard has no title-stage labels, so title-stage losses are measured as gold abstract-stage and final includes removed at the title (`gold includes lost here` in the flow).

## 10. Tests

`tests/test_three_stage.py`:
- unanimous-only exclusion;
- the agent sees only the title, and an abstract quote cannot exclude;
- an off-topic title stops before triage and abstract screening;
- an on-topic title goes through all three stages;
- `title_stage: false` changes nothing;
- flow counts with reasons and gold losses, plus the flow figure;
- unlabelled PMID lists;
- the AI/ML codebook and the stand-in model following it;
- `annotate: false` stopping after screening;
- preflight on the search corpus.

## 11. How to run and inspect it

```bash
python -m adrd_review.graph.runner configs/runs/E4_threestage_fake.yaml    # placeholder gold, no key
python -m adrd_review.graph.runner configs/runs/PROF_search_fake.yaml      # the professor's 1,855 search results, no key
python -m adrd_review.preflight configs/runs/PROF_search_gemini.yaml       # calls, time and cost before a real run
python -m adrd_review.eval.flow data/runs/<name>/<timestamp>               # re-print a run's stage counts
```

Every run prints **ARTICLES THROUGH EACH STAGE (PRISMA flow)** and saves it in `report.json` (`flow`) and as `flow.png`:

- **Identified**: the number of articles.
- **Title**: screened / excluded / passed, with reasons.
- **Abstract**: screened / excluded / passed, with reasons, and how many a person decided.
- **Full text**: sought / retrieved (by source) / not retrieved / assessed / excluded / passed, with reasons, and those still open for a person.
- **Included**: confirmed, and waiting for a person.
- **Gold losses**: with a gold standard, how many of its includes each stage removed.

## 12. Limitations and planned changes

- **No title-stage gold labels** in the placeholder, so stage 0 is judged by the includes it loses, not by its own precision.
- **Real counts need real models.** The fake model's counts check the plumbing only; the lenient stand-in keeps every title by design.
- **The topic codebook is confirmed in part.** Counts for the professor's search use `configs/codebook_aiml.yaml` v0.3.0, which has her criteria (2026-10-07; older adults 60+, 2026-10-08). Three items are still PROPOSED: counting a mean age of 60+ as meeting I4, E3 (editorials, protocols, retracted papers) and E4 (animal studies).

## 13. Change log

- 2026-10-03: built, with stage counts (`eval/flow.py`), unlabelled-corpus runs (`gold: null` + `pmids_file`), and the draft codebook for the AI/ML topic.
- 2026-10-08: codebook v0.2.0 with the professor's criteria; `annotate: false` for counts-only runs. Later: v0.3.0 (older adults 60+).
