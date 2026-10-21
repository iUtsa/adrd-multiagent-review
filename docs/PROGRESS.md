# Progress log

What was built, when, and how it was checked. Newest first. Task numbers refer to [10_DAY_PLAN.md](10_DAY_PLAN.md).

---

## 2026-10-21 (evening) — Final cleaned result, 65+ throughout; package for the professor

**Final result** (`data/runs/PROF_search_free/20261021-141010/final/`, built by `python -m adrd_review.final_results`): one outcome and one reason per paper, in the human team's order: titles (panel) → abstracts (panel, then the 65+ age rule on the abstract) → free full text (panel → second look → 65+ age check) → included / excluded / for the professor.

| Stage | In | Excluded | Passed / included | For the professor |
|---|---|---|---|---|
| 1. Titles | 1,855 | 253 | 1,602 | — |
| 2. Abstracts | 1,602 | 420 | 1,182 | — |
| 3. Full text (free) | 709 | 127 | **391** | 185 (+6 not finished) |

Not counted: 163 free copies still to save by hand; 310 paid access only. For the professor: 98 papers do not report age, 75 cannot be confirmed 65+, 12 have other rules unclear (one age policy decision settles 173). Paid calls in total: $8.80 (OpenRouter); everything else free.

**Abstract-stage age rule** (`age_check.py --stage abstract`): Nemotron 3 Ultra on the 337 abstracts that state an age (851 state none and pass without a call); excluded only when the abstract clearly shows under 65 with a quoted number: 7 papers, 1 of which is kept because its full text confirms 65+ (the abstract's number was a younger subgroup). Age reading is strict: with ages only per group, the youngest group's mean is used (noted for the professor).

**Package** (`scripts/make_report.py` → `data/deliverables/2026-10-21_ADRD_screening_results/`, git-ignored): report.pdf/.html, decisions.csv, for_professor.csv, not_assessed.csv, the blind random check (professor_check.xlsx: 40 papers, 10 included + 10 excluded per stage, AI columns hidden; answer key separate), charts and diagrams.

**Third look** (`third_look.py`, Utsa: one more model on the papers left for a person): DeepSeek V4.1 Flash (OpenRouter, ~$0.005/paper; a family used nowhere else) re-read the 185 papers: age facts with quotes at 65+, plus every rule. Settled 33 (17 included, 16 excluded: 9 for age under 65 found in the paper, 4 I1, 3 I3); 25 not re-read because the $10 OpenRouter credit ran out ($10.03 spent in total).

**Final (with the third look):** titles 1,855 → 1,602; abstracts → 1,182 (420 excluded); free full text 709 → **408 included**, 143 excluded, **152 for the professor** (94 do not report age, 53 cannot be confirmed 65+, 5 other rules), 6 not finished. Three per-stage CSVs (`final/stage1_titles.csv`, `stage2_abstracts.csv`, `stage3_fulltext.csv`) list every paper at each stage with its decision, reason and quotes.

**Tests: 158 pass.**

---

## 2026-10-21 — First full pass done; second look at the papers left for a person

**Try 6 (finished 09:10, `data/runs/PROF_search_free/20261021-112117/`), free full text only:** titles 1,855 → 1,602 passed (253 excluded); abstracts 1,602 → 1,188 passed (414 excluded, 224 of them 2-to-1 and flagged); full text 672 free papers → **498 included**, 30 excluded, 120 left for a person, 24 to retry. Side note: 516 paid-access papers found, not assessed (Utsa: free full text only, in every count and accuracy measure). Panel agreement (Fleiss κ): titles 0.40, abstracts 0.56, full text 0.43. OpenRouter spent $2.34.

**Changes today:**
- Reviewer B full text: Gemma 4 31B via OpenRouter (paid, ~$0.003/paper) first, NVIDIA as overflow: NVIDIA's free Gemma timed out on most calls (~2 min for "OK").
- OpenAI-compatible clients: no hidden client retries (`max_retries=0`); one hung request had blocked a run ~30 min.
- `caffeinate -i` while the loop runs (idle sleep on battery stopped the run 14:51-15:08 on 2026-10-20).
- The professor's random sample excludes paid-access papers.

**Second look** (`src/adrd_review/second_look.py`, Utsa: double-check every paper left for a person before the final result): an independent model, GLM-5.3 (Z.ai; OpenRouter, paid ~$0.004/paper, NVIDIA free copy as overflow), re-reads the same full text, told which rules the panel left open (not the panel's votes), with the same per-rule form, quote check and decision rule as a full-text reviewer: include / exclude / still for the professor. Written to `<run>/second_look.csv`; the run's own decisions are not changed.

**Age rule corrected to 65+ (2026-10-21):** Utsa confirmed the professor's rule is 65 or over (email 2026-10-07); the 60+ record of 2026-10-08 was wrong. Because the codebook text is in every reviewer prompt, changing it would re-judge all 1,855 papers; age only decides at full text, so a dedicated **age check** (`age_check.py`) re-applies I4 at 65 to every paper included or left for a person: a free model (Nemotron 3 Ultra, NVIDIA) extracts the age criterion, youngest/oldest and mean/median age with quotes; quotes are verified; a fixed rule decides met (criterion/youngest/mean or median ≥65) / not met (mean or median <65, or oldest <65) / unclear. Included papers with I4 not met become excluded; papers left for a person only on age become included or excluded. Pilot: ~90 s per paper.

**More free full texts (2026-10-21):** of the 516 papers without a fetched full text, 98 had a free copy Europe PMC could not download. `scripts/fetch_free_fulltext.py` looks up every legal open-access copy in OpenAlex (repositories, author manuscripts, preprints), downloads it with an honest User-Agent and keeps it only if it is the right paper (PDF quality check + ≥70% of title words): **43 downloaded** (16 arXiv, 8 Amsterdam UMC repository, bioRxiv, medRxiv, university repositories; 23 are preprints), **163 free but behind pages that block scripts** (listed in `<run>/save_by_hand.xlsx` for a person), **310 no free copy**. No paywall or bot protection is bypassed. Try 7 read the 43 (done 10:34; 6 papers still open).

**Second look cost:** OpenRouter billed $7.07 in total by 10:00 (second look ~$3.90: GLM-5.3 bills long reasoning). Second look on try 6: 9 included, 15 excluded (mostly MCI-only studies), 96 still for the professor (84 only on age).

**Architecture docs, diagrams and charts updated (2026-10-21):** [ARCHITECTURE.md](ARCHITECTURE.md) rewritten as built (v1.0): pipeline path and the complete graph exported from LangGraph, decision rules per stage, models and hosts with every change during the run, unattended loop, outputs, results, evaluation plan. Nine Mermaid diagrams with sources (`docs/diagrams/src/`), rendered with mermaid-cli; nine result charts + CSVs from the run (`eval/report_figures.py`, validated colour-blind-safe palette). The v0.3 design diagrams moved to `docs/diagrams/design-v0.3/`. Index: [diagrams/README.md](diagrams/README.md). Build doc: [agents/second-look.md](agents/second-look.md).

**Tests: 155 pass.**

---

## 2026-10-20 — First complete stage files; decisions to finish faster

**Counts after try 13** (`data/runs/PROF_search_free/20261018-174229/stages/`): titles 1,855 → 1,602 passed; abstracts 744 judged (220 excluded, 524 passed), 858 waiting; full text 62 included, 3 excluded, 37 for a person, 207 open, 215 without a free full text.

**Decisions (Utsa):**
- **Free full text only.** Papers whose full text needs a subscription are not used (left out as "not retrieved", reported as such in the PRISMA flow). 215 so far.
- **$10 OpenRouter credit** (Utsa adds it): lifts reviewer C's free model limit from 50 to 1,000 requests/day and pays for the abstract overflow below.
- Groq's paid tier is closed ("temporarily unavailable due to high demand").

**Changes:**
- **Reviewer A, full text:** `nvidia/nemotron-3-super-120b-a12b` stopped working (NVIDIA: 404 "Function not found", also via OpenRouter) after judging 175 full texts. Replaced by `nvidia/nemotron-3-ultra-550b-a55b`, the larger model of the same generation (answers in ~1 s; passed preflight). *Correction (2026-10-21):* those 175 answers were **not** reused: the cache key includes the model, so Ultra re-read all 648 full texts, and every A full-text vote is from Ultra.
- **Abstract overflow** (`ModelConfig.overflow`, `OverflowLLM`): once a Groq model's daily limit is used up, the same model is called on OpenRouter (paid; ~$1-3 for all remaining abstracts). Answers cached from either host are reused first, so no batch is judged twice; no credit at the overflow host only defers the batch (it never stops the run). The overflow host is also used when Groq's per-minute budget is busy (8,000 tokens/min fits about one abstract batch) or Groq keeps asking to wait over 20 s, so both hosts work side by side. First check (2026-10-20 11:00): ~$0.001 per batch call, abstracts ~4/min. Prices in `configs/prices.yaml`; preflight tests overflow hosts as warnings.
- **Faster loop:** 1 h between tries (was 3 h); 6 articles at a time (was 3). Loop restarted 2026-10-20 10:5x; its tries count from 1 again.

**Age (I4) drives the person-decides cases:** 22 of the 37 are open only because a paper does not report age clearly; I4 is the rule most often marked unclear (82 votes, vs at most 11 for any other). To ask the professor: does an unreported age, or a dataset of older adults such as ADNI or NACC, count as 60+?

**Tests: 152 pass.**

---

## 2026-10-17 — Full-text keys, reviewer C's model, stage files for checking

**Keys:** NVIDIA and OpenRouter keys added to `.env` (2026-10-16); both answered a test call. Full text starts with try 11.

**Reviewer C's full-text model:** `thinkingmachines/inkling:free` is now refused for direct API calls (OpenRouter allows it only in agent apps, 403). Replaced by `dots-studio/dots-3-note-preview:free` (512k context, structured output; passed preflight). C had not read any full paper yet, so no results are mixed.

**Reviewer B's full-text model moved to NVIDIA (same model):** on OpenRouter's free pool every `google/gemma-4-31b-it:free` call was rate limited (try 11: 42 papers failed, 0 B answers), so no paper could finish full text. NVIDIA hosts the same `google/gemma-4-31b-it`; it passed preflight but answers slowly (~2 min per call). Its retries also used up OpenRouter's 50 free requests/day, which reviewer C (dots) now has to itself. Try 11 stopped early (15:32; everything cached) and try 12 started with the change.

**OpenRouter's daily limit now fails fast:** its 429 gives a reset time (`X-RateLimit-Reset`, ms) instead of Groq's 'try again in ...', so each paper retried 5 times (~2.5 min) before failing. `retry_after_seconds` now reads the reset time, so the model is marked used up until then and its papers are skipped at once. Try 12 stopped at 13:42 on 2026-10-18 (paper 330 of 829; all answers cached); try 13 started with the fix.

**Overnight pause:** the Mac slept with the lid closed, and `sleep` in `run_until_done.sh` does not count sleep time, so try 11 started only on 2026-10-17 10:59. `caffeinate -s` now keeps the Mac awake while the run is going (plugged in only).

**Abstract counts so far (after try 10):** 508 of 1,602 abstracts screened, 134 excluded, 374 passed. The flow printed 281 screened, because papers that passed abstracts but waited for the full-text keys were counted as open. Of the 374: 227 have a free full text, 147 have none (`pdfs_needed.csv`).

**Stage files for checking** (`eval/stages.py`, written by every run into `<run>/stages/`):
- `1_title.csv`, `2_abstract.csv`, `3_fulltext.csv`, `4_included.csv`: every finished paper at that stage with its outcome, main reason, each reviewer's vote, rule marks, quotes and reasoning. Row counts match the flow.
- `stages.xlsx`: the same as sheets, plus `random sample` (up to 5 excluded and 5 passed per stage, fixed seed, so the same draw can be repeated) and `how to check`.
- Blind check: the paper and empty `prof_decision` / `prof_rule` / `prof_notes` columns come first; the AI's columns are hidden until unhidden.
- Re-run on any run folder: `python -m adrd_review.eval.stages <run dir> [--sample N --seed S]`.
- **Papers still open are included:** the runner now keeps what an article finished before it was put on hold (`in_progress.jsonl`, same format as `decisions.jsonl`): its title decision, and its abstract decision if made. The stage files list it at each finished stage and as `still open` where it waits. Before this, a paper waiting for full text was missing from the title and abstract files too. Takes effect from the next try (try 11 started with the old code).

**Tests: 150 pass.**

---

## 2026-10-11 — Free tiers only; batch screening; unattended multi-day runs

Decision (Utsa): use free model APIs (OpenRouter, Groq and others), even if slow. Details: [FREE_MODELS.md](FREE_MODELS.md).

**Free sources checked today:**
- **Groq:** works (key in `.env`).
- **NVIDIA NIM and OpenRouter:** free, but need keys.
- **GitHub Models:** the endpoint answers only "OK", so it is no longer usable.
- **Cerebras:** now a paid trial.
- **MCP:** connects models to tools; it does not run models, so it cannot replace a model API.

**Built:**
- **Batch screening** ([agents/batch-screening.md](agents/batch-screening.md)): 50 titles or 4 abstracts per call, with the same rules and per-article decisions. The stand-in model gives identical counts batched and unbatched.
- **Per-stage models per reviewer** (`title_model`, `eligibility_model`): Groq cannot fit a full paper (8,000 tokens/min).
- **Optional triage** (`triage: false`): saves a call per article.
- **Providers:** `nvidia` and `mistral` (OpenAI-compatible).
- **Waiting instead of failing:**
  - a model whose daily limit is used up fails fast (`LimitReached`) and its articles are deferred;
  - a model without a key makes only its articles wait.
- **Preflight:**
  - reports multi-day runs as days needed, not failures;
  - warns, rather than fails, for keys needed only at full text.
- **`scripts/run_until_done.sh`:** re-runs every 3 hours until no article is open; each try appends the flow counts to the log.

**Model checks on real papers:**
- **One batch of 50 titles:** Qwen excluded 3, all correct. The two gpt-oss models excluded about half, for not *mentioning* dementia or AI. Unanimity is required, so the panel excluded only those 3. The instructions were tightened.
- **One batch of 5 abstracts:** the three models agreed (Parkinson's and MCI-only studies excluded). gpt-oss uses "medium" reasoning for abstracts.

**Title stage finished 2026-10-11 (first real stage on the professor's corpus, free Groq panel, ~45 min):**
- **1,855 titles screened, 253 excluded, 1,602 passed;** no batch deferred.
- Reasons: I1 not dementia/Alzheimer's 170, E1 review 47, I2 no AI/ML 22, E3 no original results 8, I3 3, E4 not human 2, I4 1.
- Each reviewer alone would have excluded: A (gpt-oss-120b) 711, B (Qwen) 318, C (gpt-oss-20b) 847. The every-reviewer-must-agree rule keeps only the clear cases.
- A sample of 15 exclusions read by hand: mostly clearly off topic (hormone therapy, COVID mortality, a neuroimaging toolbox, urinary infections). There are borderline ones to review with the professor: Creutzfeldt-Jakob disease (a rapidly progressive dementia), MCI-only titles (excluded under her "MCI alone does not count" rule, although the abstract might add Alzheimer's), and a neurodegeneration-spread model.

**Running since 2026-10-11 08:42** (`PROF_search_free`, unattended):
- **Titles:** about 1 day.
- **Abstracts:** about 2 weeks at Groq's 200,000 tokens/day per model.
- **Full text:** waits for NVIDIA and OpenRouter keys.

**Tests: 148 pass.**

---

## 2026-10-10 — Claude models, first real pilot on the professor's search, four fixes

Gemini on hold. Anthropic and OpenAI keys added to `.env`. **The OpenAI account has no credit** ("You have no credits remaining"); the Anthropic key works.

**Configs:** `PROF_search_claude.yaml` (Sonnet 5.5 / Sonnet 4.6 / Haiku 5.5, adjudicator Opus 5.5) and `PROF_search_mixed.yaml` (OpenAI gpt-6-sol replaces Sonnet 4.6). See [MODELS.md](MODELS.md). Prices in `configs/prices.yaml`.

**Pilot, 100 random papers** (`python -m adrd_review.graph.runner configs/runs/PROF_search_claude.yaml --sample 100`; `data/runs/PROF_search_claude/20261010-152004/`):
- **65 papers completed; 35 failed because the Anthropic account ran out of credit** mid-run. They were kept (predicted include), not dropped.
- Cost **$6.37 for 65 papers (~$0.10/paper)**, 4 minutes. By role: eligibility $2.06, screening $1.57, title $1.43, adjudicator $1.24.
- Of the 65: 12 excluded at title, 9 at abstract, 44 sought at full text (23 retrieved, 21 not retrieved), 9 still unclear at full text. Panel agreement: titles κ 0.72, abstracts κ 0.68, full text κ 0.21.
- Not a result: incomplete, and the fixes below change the full-text stage.

**Fixed (found by the pilot):**
1. **Tables were missing from PubMed Central full texts.** "Age unclear" (I4) was the most common open judgment at full text (15 times), and the participants' age is usually only in Table 1. `tools/pmc.py` now adds every table (caption, one line per row, footnotes) as a "Tables" section, and tables are sent early for long papers. Cached full texts carry a parser version, so older ones are fetched again automatically (refresh running: `scripts/fetch_search_corpus.py --no-oa`).
2. **The verifier flagged includes at abstract stage** because the age was not reported, though screening keeps such papers by design. Its instructions now state what "supported" means at each stage. It still caught real problems (a study of simulated MRI scans; "cognitive impairment, not dementia").
3. **A run kept going after the account ran out of credit** (5 retries per call, then every remaining article). Now the first out-of-credit error stops the call at once and the runner starts no new articles; re-running the same command resumes from the cache.
4. **Cost was not reported for Claude.** Usage is now priced from `configs/prices.yaml`, including prompt-cache rates.

**Added:** `temperature: null` (newer models reject it), `--sample N` (fixed random pilot), Anthropic prompt caching (`prompt_cache: true`), schema overhead in the preflight estimate. **Tests: 144 pass.**

**Needed:** Anthropic credit (full run ~$180 at the pilot rate, less with caching); OpenAI credit to use the mixed config.

---

## 2026-10-08 (later) — Older adults 60+; papers without full text left out

> **Correction (2026-10-21):** the professor's age rule is **65 or over** (her email of 2026-10-07, confirmed by Utsa). The 60+ below was recorded wrongly. The run's prompts kept 60+ (changing them would re-judge everything); 65+ is applied at full text by the age check (`src/adrd_review/age_check.py`).

The professor sent the review's age groups (young adult 20–39, adult/middle age 40–59, **older adult 60+**). She also said papers without obtainable full text need not all be covered ("whatever we can pass through"). Gemini access is on the way.

- **Codebook v0.3.0**: I4 means age 60+. Still PROPOSED: a mean or median age of 60+ counts (so ADNI studies, ages 55–90, mean ~73, count).
- **`missing_fulltext: not_retrieved`**: a paper whose full text cannot be obtained is reported as **not retrieved** (PRISMA "reports not retrieved") and left out, with no model call and no person. The default (`human`) still keeps such papers for a person, so gold-standard runs are unchanged. The verifier skips these decisions.
- Set in `PROF_search_*`. The stand-in run on all 1,855 papers shows the flow with "not retrieved 602 (left out)".
- **Gemini estimate for all 1,855: ~$96 at most, ~6.5 hours; 100-paper pilot ~$5.** Preflight no longer counts full-text calls for papers without a full-text source.
- **Tests: 140 pass.**

---

## 2026-10-08 — The professor's criteria in the codebook; counts-only runs

**Professor's answers (email, 2026-10-07):**
- No comparison with the human reviewers for now. She wants the **number of papers at title, abstract and full-text screening**.
- MCI alone does not count; MCI together with Alzheimer's or dementia counts.
- Logistic regression alone counts as ML.
- Reviews do not count; full papers in conference proceedings count, abstract-only publications do not.
- Older adults means 65 or over.
- Previous system: a dual agent (no further detail).

**Codebook v0.2.0** (`configs/codebook_aiml.yaml`, renamed from `codebook_aiml_draft.yaml`):
- I1 now excludes MCI-only studies; I2 counts logistic regression; I4 means age 65+.
- New exclusions: E1 Review, E2 Abstract only.
- Still PROPOSED, for her to confirm:
  - **I4: a mean or median age of 65+ counts.** Read literally, "65 or over" would exclude every study on ADNI, whose participants are aged 55–90 (mean ~73), and ADNI appears in 329 of the 1,855 abstracts (18%).
  - E3: editorials, protocols without results, retracted papers (4 retracted papers and 6 trial protocols are in the corpus).
  - E4: animal or cell studies.
- Worked examples added for I1 and I4.

**`annotate: false`**: a counts-only run stops after full-text screening (no extraction, no annotation). Set in `PROF_search_*`. The Gemini estimate for all 1,855 papers falls from ~$143 to **~$107 at most, ~7 hours**; a 100-paper pilot costs ~$6. Preflight now warns about PROPOSED rules.

**Checked:** stand-in run on all 1,855 papers with v0.2.0, 0 failures, 29 s (counts are plumbing only). **Tests: 139 pass.**

**Still needed for the real counts:** the Gemini key, and the team's PDFs. 602 papers that pass the abstract stage have no free full text, so the full-text count is incomplete without them.

---

## 2026-10-03 — Three stages like the human team (titles → abstracts → full text), stage counts, step 4 finished

**Step 4 finished: `E2_groq`, all 40 calibration articles, 0 failures** (`data/runs/E2_groq/20261002-113941/`).

| | Recall | Precision | F1 | Specificity | κ | WSS@95 |
|---|---|---|---|---|---|---|
| Keyword baseline | 0.769 | 0.435 | 0.556 | 0.519 | 0.240 | – |
| **E2_groq (3 free models)** | **0.769** (10/13) | **0.769** | **0.769** | **0.889** | **0.658** | **0.475** |

- Panel: Fleiss' κ 0.597, 29 of 40 unanimous; 7 split-vote exclusions flagged for a human. Instrument labels micro F1 0.800; setting 0.143 (placeholder label problem, see below).
- **All 3 missed gold includes were split votes flagged for a human** (so in E4 a person would have seen them), and Cochrane itself excluded all 3 after reading the full paper.
- Against Cochrane's **final** (full-text) includes: the panel kept **6 of 6** (keyword baseline 5 of 6) while passing only 13 of 40 articles to full text (baseline 23 of 40).
- The partial result on 29 articles (recall 0.90) fell to 0.77 on all 40: the last 11 articles held 3 harder includes. Placeholder data and free models: a calibration result, not a study result.

**Three-stage review (the professor's description of the human process).** Build doc: [agents/title-screening.md](agents/title-screening.md).
- **Title screening agent** (`agents/title_screening.py`): title only; same per-rule form; quotes must come from the title; the panel excludes only if every reviewer excludes (`consensus.combine_title`). `title_stage: true` in a config; off by default (earlier configs unchanged).
- **Stage counts** (`eval/flow.py`): every run now reports a PRISMA flow: identified → title (screened / excluded / passed, reasons) → abstract (same, plus person-decided) → full text (sought / retrieved by source / not retrieved / assessed / excluded, reasons, still open) → included (confirmed / waiting for a person). With a gold standard it also counts the gold includes each stage removed. Printed, saved in `report.json` and drawn as `flow.png`; `python -m adrd_review.eval.flow <run dir>` re-prints it.
- **Unlabelled corpora**: `gold: null` + `pmids_file:` runs any PMID list and reports counts without scores (the professor's 1,855 search results have no labels).
- **Draft codebook for the real topic**: `configs/codebook_aiml_draft.yaml` (v0.1.0-draft): I1 ADRD, I2 AI/ML, I3 detection/diagnosis/monitoring/prediction, I4 older adults (from the four concept groups of her query); E1 not original research, E2 not human (PROPOSED); annotation dimensions data modality / population / dataset source (PROPOSED, from the abstract's categories). **Needs the professor's approval**; counts made with it are not reportable until then. Generated prompt: [agents/prompts/screening_aiml_draft.md](agents/prompts/screening_aiml_draft.md).
- The codebook now sets the topic named in every prompt (`topic:`); the placeholder's prompts are byte-identical, so cached answers stay valid. The stand-in model follows any codebook with `keywords:` (used by the stand-in only, never shown to real models).
- Preflight understands title stages and unlabelled corpora.
- Configs: `E4_threestage_{fake,gemini}`, `PROF_search_{fake,gemini}`.

**Checked:**
- `E4_threestage_fake`, 40 calibration articles: 40 titles screened, 0 excluded (the placeholder came from a targeted search: every title is on topic); 40 abstracts → 24 passed; 24 full texts sought, 2 retrieved; 5 included, gold final includes kept 5 of 6.
- `PROF_search_fake`, **all 1,855 search results, stand-in model** (plumbing check, NOT results): 41 excluded at title, 302 at abstract, 1,512 sought at full text, 909 retrieved (907 PubMed Central, 2 open-access PDF), 603 without full text (flagged). 24,303 calls in 105 s, 0 failures.
- Preflight for `PROF_search_gemini`: about 18,800 calls, ~9 hours at 10 requests/min per model, **about $143 upper bound** (assumes 80% pass titles, 50% of those pass abstracts); a 100-article pilot ≈ $8, 30 min.
- **Tests: 138 pass** (12 new in `tests/test_three_stage.py`).

**Waiting for:** Gemini key (real counts for the 1,855); the professor's approval of the draft codebook (or her own); the team's PDFs for papers without free full text.

---

## 2026-10-01 — Step 4 (real re-run after the fix) and the professor's papers collected

**Professor's keyword search collected** (`python scripts/fetch_search_corpus.py`): 1,855 abstracts in `data/corpus_search/`; **full text for 1,118 (60%)**: 1,112 PubMed Central + 6 open-access PDFs. No full text for 737: 591 paywalled, 143 with a free copy that needs a manual download, 3 other. All 737 are in `data/search/pdfs_needed.csv` (link + file name to save as). Typical paper ~8,600 tokens; 22 exceed 80,000 characters (one is ~285,000 tokens), handled by the size cap.

**Step 4: `E2_groq` re-run on the 40 calibration articles (codebook v0.2.0, per-rule screening).** 29 of 40 articles completed; 11 are waiting on Groq's daily token limit.

Screening on the **29 completed articles** (10 gold includes):

| | Recall | Precision | F1 | Specificity | κ | WSS |
|---|---|---|---|---|---|---|
| First real run, before the fix (all 40) | 0.000 | – | 0.000 | 1.000 | 0.000 | 0.000 |
| Keyword baseline (same 29) | 0.800 | 0.471 | 0.593 | 0.526 | 0.280 | 0.214 |
| **E2_groq after the fix (same 29)** | **0.900** | **0.818** | **0.857** | **0.895** | **0.776** | **0.521** |

- Panel agreement on screening: Fleiss' κ 0.609, 21 of 29 unanimous; 4 split-vote exclusions flagged for a human.
- Instrument labels: micro F1 0.846. Setting labels: 0.160 (the placeholder's setting labels are derived, and `community_or_primary_care` overlaps two other labels: a placeholder problem).
- Errors among the 29: one missed include (a 37-item MMSE version, excluded under I1 by two reviewers), two wrong includes.
- This is a **partial calibration result on placeholder data with free models**, not a study result.

**Three problems found by running for real, all fixed:**
1. **Qwen's function-calling mode intermittently produced invalid answers** (7 of the first 12 articles). Fix: Qwen uses the constrained JSON-schema mode, and any rejected answer is retried with the next request format instead of the same one.
2. **Two roles on one model each had their own rate limiter** and together exceeded 8,000 tokens/min. Fix: one limiter per model, with a tokens-per-minute budget, and "try again in N seconds" is honoured.
3. **Groq's free tier has a daily token limit (200,000 per model) that its headers do not show.** One model reached it. Fix: `tokens_per_day` in the Groq configs; the preflight check now fails a run that would exceed it. Finished calls are cached, so the run continues the next day with the same command.

- A background time limit (~30 minutes) also stopped one attempt; nothing was lost (cache).
- **Tests: 126 pass.**

**To finish the remaining 11 articles** (after the daily limit rolls over): `python -m adrd_review.graph.runner configs/runs/E2_groq.yaml`

---

## 2026-10-01 — Two-stage review: agents now read the full paper

Decision (Utsa): the human reviewers judged from full papers, so the agents should too. Analysis: [FULLTEXT_PLAN.md](FULLTEXT_PLAN.md). Build document: [agents/fulltext-eligibility.md](agents/fulltext-eligibility.md).

**Measured on the professor's 1,855 search results:** ~61% have full text in PubMed Central, ~5% free elsewhere, **~34% paywalled** (need the team's PDFs). A full paper is ~9,700 tokens (max ~17,000 in a sample of 57).

**Built:**
- **Full-text sources** (`tools/fulltext.py`, `tools/pdf.py`): team PDF → PubMed Central → open-access PDF → none (flagged, never excluded, listed in `pdfs_needed.csv` with a link). PDF quality check; reference lists dropped; long papers put Methods/Results first.
- **Stage 2 full-text eligibility agent** (`agents/eligibility.py`): same per-rule form as stage 1, stricter decision (include only if every inclusion rule is met; unclear → human). Panel, debate, adjudicator, verifier and human review all apply to it.
- **Pipeline** rewritten with the stage-2 branch; extraction and annotation read the full paper. New escalation reasons `no_fulltext`, `unclear_fulltext`. The verifier no longer re-checks a human's decision.
- **Scoring**: stage 2 scored against the gold `fulltext` column; annotation scored on gold final includes.
- **Preflight check** (`python -m adrd_review.preflight <config>`): keys, live model test calls, articles, full-text sources, call sizes vs token limits, calls vs daily limits, time, cost, freeze, disk.
- Codebook examples are now rendered into prompts when present.
- Configs `E3_twostage_*`, `E4_twostage_*`; Groq configs carry their limits (8,000 tokens/min, 1,000 requests/day).

**Checked:**
- Real papers: PubMed Central path works; paywalled and blocked papers fall through with a reason and link; the PDF reader extracts clean text from a real two-column academic PDF and rejects non-PDF files.
- Preflight: Groq panel READY (three models answered live test calls); Gemini two-stage NOT READY (no key), ~$3 for 40 articles; full text on Groq refused (11,700-token call vs 8,000 tokens/min).
- Fake two-stage runs on calibration: 23–24 reached stage 2; 1 PubMed Central, 1 open-access PDF, 21–22 flagged with no full text (old placeholder papers); in E4 the gold "human" answered them (stage-2 precision 1.00, κ 0.90).
- **Tests: 123 pass.**

**Not yet done:** any real-model run of stage 2 (needs Gemini); the Groq re-run of stage 1 (waiting for the go-ahead).

---

## 2026-09-30 — First real-model run, what it showed, and the fixes

**Groq key added** (free tier; in `.env`, not in git). Models: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` (Qwen needs function-calling structured output). Limits: 1,000 requests/day, 8,000 tokens/min per model (~3 calls/min). Configs `E1_groq`, `E2_groq`, `E2_mixed`; rate-limit backoff 15–60 s.

**First real run: `E2_groq`, 40 calibration articles, 18 minutes, 160 calls, 0 failures.**

| | Recall | Precision | Specificity | κ |
|---|---|---|---|---|
| Keyword baseline | 0.769 (10/13) | 0.435 | 0.519 | 0.240 |
| E2_groq (3 real models, codebook v0.1.0) | **0.000 (0/13)** | – | 1.000 (27/27) | 0.000 |

The panel excluded all 40 articles; 37 of 40 unanimously. Three were 2–1 splits, all flagged by the screening safety rule.

**Diagnosis** (read from every vote on the 13 gold includes):
1. **Reviewers treated "not reported" as "not met".** The codebook said to give the benefit of the doubt at abstract stage; given a free include/exclude choice, the models did the opposite (e.g. "the abstract does not report a reference standard → I2 not met → exclude").
2. **The placeholder codebook was stricter than the labels it is scored against.** E1–E3 (case-control, special populations, nursing homes) came from the MoCA review only but were applied to all instruments (10 wrong exclusions cited E1). I4 was ambiguous for studies of two instruments.
3. **The gold labels are lenient abstract-stage decisions**: Cochrane itself excluded 7 of these 13 after full text, often for the reasons the models gave. But the models also missed the 6 that Cochrane finally kept.
4. Triage answered "not enough information" for every article.

Also seen: the real models handled the "MOCA" keyword trap (chemotherapy regimen) that fooled the keyword baseline, and the quote checker flagged a shortened quote.

**Fixes (offline; real re-run pending network):**
- **Screening reviewer redesigned** ([agents/screening-reviewer.md](agents/screening-reviewer.md)): the model judges each rule met / not_met / unclear with a quote, and the code decides: exclude only on a clear violation backed by a real quote; unclear never excludes.
- **Codebook v0.2.0-placeholder**: decision rule rewritten; I4 satisfied if the setting fits at least one evaluated instrument; E1–E3 scoped to MoCA studies.
- **Triage**: model reports `clearly_ineligible` + `missing`; `enough_information` is computed.
- Adjudicator and verifier now see each reviewer's rule judgments.
- Fake model updated to the new answer forms; fake scores unchanged (0.769 / 0.435 / 0.240), 0 warnings.
- **Tests: 109 pass.**

**Next:** re-run `E2_groq` on calibration and compare (about 20 minutes, free).

---

## 2026-09-30 — Everything that can be built without a key is built

| Item | Status | Where |
|---|---|---|
| Repeats that really call the model again (`--repeat N`) | ✅ | runner, [ANALYSIS.md](ANALYSIS.md) |
| Results table over repeats + run-to-run κ; only comparable runs averaged | ✅ | `eval/aggregate.py` |
| Error-analysis spreadsheet with category drop-down + summary | ✅ | `eval/errors.py` |
| Figures: conditions; accuracy vs human effort (validated palette, CSV twins) | ✅ | `eval/figures.py` |
| Retrieval check of the professor's search | ✅ | `eval/retrieval.py` |
| PubMed Central full text (`fulltext: never / when_needed / always`) | ✅ | `tools/pmc.py`, [TOOLS.md](TOOLS.md) |
| Reviewer weights from calibration (log-odds of accuracy) | ✅ | `eval/weights.py`, `weights_file:` |
| Cost estimator + Gemini prices | ✅ | `eval/cost.py`, `configs/prices.yaml` |
| Converter for the professor's spreadsheet | ✅ | `convert_gold.py`, `configs/gold_mapping.example.yaml` |

- **Bug caught by a test:** the full-text parser dropped any section whose title began with "Reference", including "Reference standard" (how dementia was confirmed, rule I2). Fixed; only a section titled exactly "References" is dropped now.
- **Found while pricing:** `gemini-3.1-pro-preview` (adjudicator) has no free tier; the E3/E4/D Gemini configs now say to use `gemini-3.8-flash` on a free key.
- Pre-run test-set cost estimate: ~$30–40 (upper bound $73 without shared calls).
- Every run records a prompt/code fingerprint; the freeze also locks the weights file.
- Stand-in results: aggregation of 3 repeats × 5 conditions (SD 0, repeat κ 1.0, as expected for a deterministic stand-in); weighted panel precision 0.435 → 0.667 in-sample; full text fetched for the 5 calibration articles with a PMCID; retrieval 0/90 (placeholder is 1971–2014, outside the query's 2015–2025).
- **Tests: 105 pass** (+1 live PubMed test, opt-in).

---

## 2026-09-30 — Days 6–8: every remaining agent built

All planned agents now exist; each has a build document in [agents/](agents/README.md) and a generated exact prompt.

| Task | Status |
|---|---|
| 6.1 Adjudicator | ✅ [agents/adjudicator.md](agents/adjudicator.md): weighs reviewers (0–1) instead of overriding; self-contradiction → human |
| 6.2 Screening safety rule | ✅ split-vote exclusion always escalated ([agents/human-review.md](agents/human-review.md)) |
| 6.3 Verifier | ✅ [agents/verifier.md](agents/verifier.md): quote check (no model) + optional model check |
| 6.4 E3 on calibration | ⏳ fake run done; real run needs Gemini access |
| 7.1 Human review with interrupt() + SQLite checkpoint + review screen | ✅ `human: queue` + `python -m adrd_review.human_review` |
| 7.2 Gold answers stand in for the human; escalations counted | ✅ `human: gold` |
| 7.3 Codebook Curator | ✅ [agents/codebook-curator.md](agents/codebook-curator.md): calibration runs only; proposals for the professor |
| 7.4 Re-run with codebook v2 | ⏳ needs real runs + professor |
| 8.1 Debate (variant D) | ✅ [agents/debate.md](agents/debate.md): hidden counts, minority first, change needs new evidence |
| 8.2 Keep/drop D | ⏳ needs real calibration runs |
| 8.3 Freeze | ✅ tool built (`python -m adrd_review.freeze`); runner refuses unfrozen test runs. Freezing itself happens after calibration. |
| 8.4 Cost estimate | ⏳ needs real token counts |

- One graph serves every condition; configs `E3_*`, `E4_*`, `E4_queue_fake`, `D_*` added. Protocol: [RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md).
- **Bug found and fixed in testing:** the verifier matched votes by reviewer only, so annotation votes hid screening votes and every included article was flagged (23/23). Now matched by reviewer and field; regression test added. After the fix: 0 verifier false alarms.
- LangGraph checkpoints: our record types registered for strict deserialization (checked with `LANGGRAPH_STRICT_MSGPACK=true`).
- Fake runs (calibration, 40): E3 20 adjudications, 12 escalations (30%, all split exclusions); E4 (gold human) answered 12, recall 0.769 → 0.846; D debated 20 articles; queue mode paused 3 of 12, answered via the review command, finished. Curator: 26 disagreements → proposals file. Test-set runs refused without a freeze.
- **Tests: 88 pass** (+1 live PubMed test, opt-in).

---

## 2026-09-30 — Gemini plan, professor's search, agent docs, Day 5 panel

**Model plan changed to Gemini** (professor provided a Gemini licence): [MODELS.md](MODELS.md).
- Panel = three Gemini versions × three perspectives; adjudicator = Gemini Pro (not on the panel).
- `llm.py`: added `groq` provider (optional free second family), `extra` options (Vertex AI project/location, Gemini thinking budget) and `structured_method`.
- Waiting on: which kind of Gemini access the licence is (AI Studio key vs Vertex AI project).

**Professor's PubMed search** (`configs/search/prof_query.yaml`):
- Original and corrected Boolean both return **1,855** articles (PubMed inserted the missing AND); PMIDs saved to `data/search/`.
- The query shows the real review topic: **AI/machine learning for ADRD detection/prediction in older adults, 2015–2025**. The placeholder codebook (Cochrane cognitive-test reviews) is a different topic; it still tests screening mechanics, and prompts will follow the real codebook automatically.

**Agent build documentation** ([agents/](agents/README.md)): a 13-section build document for every agent (triage, screening reviewer, extraction, annotation reviewer, reviewer panel), plus exact prompts generated from the live code by `scripts/render_prompts.py`.

**Day 5: reviewer panel (E2)**

| Task | Status |
|---|---|
| 5.1 Panel config | ✅ `E2_gemini.yaml` (3 Gemini versions × 3 perspectives), `E2_fake.yaml` |
| 5.2 Parallel reviewers, votes, agreement check, weighted vote | ✅ LangGraph `Send` fan-out; `consensus.py` |
| 5.3 Reusable for screening and each dimension | ✅ Same graph, per-dimension combining |
| 5.4 Agents' Fleiss' κ vs humans' | ⏳ Agents' κ reported every run; needs a real model run and per-reviewer human labels |

- E1 is now a panel of one: one graph for all conditions; E1 results unchanged.
- Fake E2 run, 40 calibration articles: reviewers disagreed on 20/40 (Fleiss' κ 0.300), 252 calls, 0 failures, 0 warnings.
- **Tests: 69 pass** (+1 live PubMed test, opt-in).

---

## 2026-09-30 — Day 4 (first agents) built, two days early

| Task | Status |
|---|---|
| 4.1 Shared state + answer forms | ✅ `graph/state.py`, `schemas.py` (Vote, Decision, Usage, and the four answer forms) |
| 4.2 Model factory + cache | ✅ `llm.py`: Google, Anthropic, OpenAI, OpenRouter, Ollama via LangChain, plus a fake model; cache, retries, rate limit, cost |
| 4.3 Agents | ✅ Triage, Screening reviewer, Extraction, Annotation reviewer; prompts generated from the codebook |
| 4.4 LangGraph pipeline + runner | ✅ `graph/pipeline.py`, `graph/runner.py`; configs in `configs/runs/` |
| 4.5 Run E1 on calibration with a real model | ⏳ Ran end to end with the fake model; real run waits for an API key (`E1_gemini.yaml` ready) |

- Fake run, 40 calibration articles: 126 calls, 0 failures, 0 warnings, 6 output files; screening scores identical to the keyword baseline, as expected (same rules), confirming the plumbing.
- Found keyword traps in the data: "MOCA" as a chemotherapy regimen and as an industrial chemical.
- All five LangChain providers build with the configured settings (dummy keys, no network).
- **Tests: 59 pass** (+1 live PubMed test, opt-in). Details: [AGENTS.md](AGENTS.md).

---

## 2026-09-30 — Day 3 (scorer) done, a day early

| Task | Status |
|---|---|
| 3.1 Metrics | ✅ `eval/metrics.py`: precision, recall, F1, specificity, Cohen's κ, Fleiss' κ, multi-label P/R/F1/κ, WSS, WSS@95, cost/time summary |
| 3.2 Tests | ✅ Hand-worked answers + matches scikit-learn and statsmodels on random data |
| 3.3 Score a known system | ✅ Substitute: old 2-agent outputs not available yet, so validated end to end on 4 non-AI baselines |
| 3.4 Human Fleiss' κ | ⏸ Code ready; placeholder gold has no per-reviewer answers |

- `eval/score.py`: compares a predictions CSV (same format as gold) with the gold file on the calibration or test set; stops if any prediction is missing; warns when scoring the test set.
- `scripts/score_baselines.py`: keyword baseline on calibration = recall 0.769, precision 0.435, κ 0.240, instrument F1 0.848, setting F1 0. This is the floor agents must beat. Details in [EVALUATION.md](EVALUATION.md).
- `gold.py` now also reads an optional `score` column (for WSS@95).
- `.vscode/settings.json` points VS Code at the project's `.venv`.
- Ollama on standby by decision (disk). Freed 1.8 GB of caches; 2.3 GB free.
- **Tests: 46 pass** (+1 live PubMed test, opt-in).

---

## 2026-09-30 — Days 1 and 2 closed

| Task | Status |
|---|---|
| 1.1 Questions to professor | ✅ Sent by Utsa |
| 1.2 Gold standard + codebook | ✅ Placeholder in use (public CLEF 2017 data); swap in the professor's file when it arrives |
| 1.3 API keys | ⏸ Deferred by decision; `.env.example` ready |
| 1.4 Python setup | ✅ |
| 1.5 Ollama | ⚠️ Code side done; install blocked by a full disk (see [LOCAL_MODELS.md](LOCAL_MODELS.md)) |
| 2.1 Codebook | ✅ Placeholder v0.1.0 from Cochrane selection criteria |
| 2.2 PubMed tool + corpus | ✅ All 268 articles downloaded |
| 2.3 Gold table | ✅ 268 rows, standard format |
| 2.4 Frozen split | ✅ 40 calibration / 228 test |

**1.2 / 2.3 Placeholder gold standard — done** (see [GOLD_STANDARD.md](GOLD_STANDARD.md))
- Built from CLEF 2017 TAR (pinned commit, MIT licence): real Cochrane screening decisions for six dementia screening-instrument reviews (MMSE ×2, MoCA, Mini-Cog, IQCODE ×2).
- 268 articles: 90 include / 178 exclude; 32 included after full text. Instrument labels: MMSE 45, IQCODE 36, MoCA 11, Mini-Cog 7.
- `src/adrd_review/gold.py`: standard gold format + validating loader (lists every problem at once).
- `scripts/build_placeholder_gold.py`: rebuilds it reproducibly.
- Checked: all 268 articles on disk; 9 without abstracts (realistic); every label is in the codebook.

**2.1 Codebook — done (placeholder)**
- `configs/codebook.yaml` v0.1.0-placeholder: 4 inclusion rules, 3 exclusion rules, each citing the Cochrane review(s) it comes from; 2 annotation dimensions (instrument, setting).
- Rule I4 makes the setting depend on the instrument (e.g. Mini-Cog: community only), matching how the source reviews decided.
- Borderline examples are empty on purpose; they get added from real disagreements during calibration (Day 7).
- `src/adrd_review/codebook.py`: loader that rejects duplicate rule/label IDs and unknown sources.

**2.4 Split — done**
- `configs/splits/placeholder.json`: calibration 40 (13 include / 27 exclude), test 228 (77 / 151); seed 20260930; stratified.
- Checked: re-running refuses to overwrite; a test fails if the gold file changes after the split.

**1.5 Ollama — partly done**
- `langchain-ollama` installed; `scripts/check_ollama.py` diagnoses each missing step.
- Not installed: disk had ~0.5 GB free. Needs ≥10 GB free, then 3 commands in LOCAL_MODELS.md.

**Tests:** 24 pass (+1 live PubMed test, opt-in).

---

**1.4 Python setup — done**
- `pyproject.toml` with all dependencies; virtual environment in `.venv/`; package layout under `src/adrd_review/`.
- Installed: LangGraph 1.2.12, langchain-core 1.6.6, langchain-anthropic 1.7.5, langchain-google-genai 4.4.0, langchain-openai 1.6.6, pydantic 2.13.5, pandas 3.0.6, scikit-learn 1.9.1, statsmodels 0.15.0 (Python 3.14.7).
- `.env.example` added; keys will be filled in when available.
- Checked: a small LangGraph graph sent one article to 3 reviewers in parallel, collected 3 votes and detected agreement.

**2.2 PubMed tool — done** (see [TOOLS.md](TOOLS.md))
- `src/adrd_review/tools/pubmed.py`: `fetch`, `search`, `lookup` (command line and Python).
- `src/adrd_review/schemas.py`: the `Article` record every agent will read.
- Checked:
  - 15 tests pass (14 offline + 1 live PubMed test).
  - Real run: a search for dementia + screening + MoCA (2024–2025) found 159 articles; 5 downloaded with complete fields; a second run read all 5 from cache; a DOI lookup returned the correct PMID.
- Waiting on: the professor's list of 268 PMIDs to run it on the real corpus.

**Still open from Day 1**
- 1.1 / 1.2 Request the gold-standard file (PMIDs + human labels), codebook, old system outputs, and the PubMed query + date used by the human team.
- 1.3 API keys.
