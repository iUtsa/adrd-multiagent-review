# Batch screening (titles and abstracts in batches)

Code: `src/adrd_review/agents/batch.py`, runner `prepare_batches` in `src/adrd_review/graph/runner.py` · Built: 2026-10-11 · Used by: `configs/runs/PROF_search_free.yaml`

## 1. Purpose

Runs the title and abstract screening reviewers on **several articles per model call**. Free model tiers limit calls and tokens per day, and every call repeats the same instructions and answer form (about 2,500 tokens), so one article per call spends most of a free budget on repetition. Batching keeps the rules and the per-article decision unchanged and cuts the number of calls about 50-fold for titles and 4-fold for abstracts.

## 2. Where it sits

```
runner: prepare_batches
  titles: every reviewer judges batches of `title_batch` titles  -> title votes per article
          (abstract batches wait until every title is decided, so batches are stable between runs)
  abstracts: every reviewer judges batches of `abstract_batch` articles that passed titles -> screening votes
graph (per article): title_consensus and screen_consensus use these votes instead of calling a model
  -> everything after (full text, verifier, flow counts) is unchanged
```

Switched on with `title_batch: 50` / `abstract_batch: 4` in a run config. `1` (the default) means one article per call, as before.

## 3. Inputs

- **Titles:** a numbered list of titles only.
- **Abstracts:** numbered articles, each rendered exactly as for single screening (title, journal, publication type, MeSH terms, abstract), separated by `=== ARTICLE n ===`.
- **Both:** the codebook rules and the reviewer's persona, as in single calls.

## 4. Output

- **Titles** (`TitleBatchAnswer`): only the titles that clearly break a rule, each with the rule ID and the exact words of that title. Every title not listed is kept.
- **Abstracts** (`ScreeningBatchAnswer`): the full per-rule form (`ScreeningAnswer`) for every article, with its number.
- **Votes:** each article gets a normal `Vote` (field `title` or `screening`) carrying its share of the call's tokens and cost (`Usage.batch_size`; reports count the call once).

## 5. Prompt

`TITLE_SYSTEM` and `ABSTRACT_SYSTEM` in `batch.py`. They are the single-article instructions plus three additions:
- judge each article only from its own text;
- answer with the article's number;
- for titles, report a title only when it states something *incompatible* with a rule (a different disease, animals, a review). A title that does not *mention* dementia, AI or age is never a reason.

## 6. Model and settings

Each reviewer's `title_model` (titles) and `model` (abstracts). In `PROF_search_free` these are the three Groq models: gpt-oss-120b, Qwen 3.8 27B and gpt-oss-20b. Sizes measured on Groq:
- **50-title call:** about 2,200 tokens in, 100–1,800 out.
- **5-abstract call:** about 3,900 tokens in, 1,800–3,100 out. The config uses 4 abstracts per call, to stay under Groq's 8,000 tokens per minute.

## 7. Checks after the model answers

- **Titles:**
  - a hit is used only if its rule ID exists and its quote appears in **that** title (`title_only`);
  - otherwise it is ignored with a warning;
  - hits on numbers outside the list are ignored.
- **Abstracts:**
  - each article goes through `screening.decide` with its own text, so a quote from another article in the batch is not evidence;
  - an article missing from the answer is kept, with a warning.
- **Panel decision:** unchanged. A title is excluded only if every reviewer excludes it; abstracts use the weighted majority, with split exclusions flagged.

## 8. Failure handling

- **A failed batch** (for example a free model's daily limit, `LimitReached`) *defers* its articles: they are not run this time and are listed as open in `failures.json`.
- **If any title is deferred, no article moves on.** That keeps abstract batch composition, and therefore the cache, stable.
- **Re-running** (`scripts/run_until_done.sh`) does only the open batches; finished ones come from the response cache.

## 9. Design decisions

- **Batches only for title and abstract screening.** Full papers are long (about 13,000 tokens) and are judged one per call.
- **Titles report exclusions only.** A title stage only removes the obviously off-topic. Asking for every rule on 50 titles would multiply the output for nothing, since the decision depends only on exclusions.
- **Same rules, same per-article decision.** Batching changes the cost, not the review protocol. The stand-in model gives identical stage counts batched and unbatched: 51 excluded at title and 313 at abstract, out of 1,855.
- **Risk of crosstalk** (one article's text influencing another's judgment) is limited by the per-article quote checks. It should be measured against single calls on a sample once a gold standard exists.

## 10. Tests

`tests/test_batch.py`:
- a title batch keeps unreported titles, and ignores quotes from another title and hits on unknown numbers;
- abstract batches decide each article like a single call; a missing article is kept; a quote from another article is not evidence;
- votes made in batches feed the graph;
- a failed batch defers all its articles;
- a missing key makes calls wait (`LimitReached`) instead of crashing.

## 11. How to run and inspect it

```bash
scripts/run_until_done.sh configs/runs/PROF_search_free.yaml
grep "batch" data/logs/until_done_PROF_search_free.log | tail
```

Each article's votes in `decisions.jsonl` have the rationale "Batch title screening (50 titles in one call)" or the model's own rationale for abstracts.

## 12. Limitations and planned changes

- **Crosstalk between articles in one call is untested** against single-call answers (needs labelled data).
- **Batch sizes are tuned for Groq's per-minute token limit.** Larger batches would suit models with bigger limits.

## 13. Change log

- 2026-10-11: built, for free-tier runs.
