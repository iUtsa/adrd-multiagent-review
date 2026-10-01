# Full-text review: analysis and plan

> Status: **built 2026-10-01** (approved by Utsa). Build document: [agents/fulltext-eligibility.md](agents/fulltext-eligibility.md). This page keeps the analysis and measurements behind the design. Not yet run with a real model (needs Gemini access).

## 1. The question

The gold standard reflects decisions made by people who read the **full paper**. Our agents currently read the **title and abstract** (plus an optional PubMed Central excerpt). Comparing the two is not like for like, and it showed in the first real run: reviewers were asked to make final-quality judgments from abstracts that often do not contain the answer.

## 2. What we measured

### 2.1 How much full text can be obtained (the professor's 1,855 search results)

| Group | Share | How we know |
|---|---|---|
| Has a PubMed Central ID | 64% (1,181) | NCBI ID converter, all 1,855 PMIDs |
| …and PMC serves the full text | **95% of those → ~61% of all** | Fetched a random 60: 57 returned body text, 3 blocked by the publisher |
| Not in PMC but free elsewhere | ~5% of all | Europe PMC records for a random 60 non-PMC articles: 8 had a free/open link |
| **Paywalled (subscription only)** | **~34% of all** | Same sample: 52 of 60 subscription only |

So about a third of the papers cannot be fetched automatically by any tool. The human team read them through the university library; **their PDFs are the only complete source.**

### 2.2 How long a full paper is (57 real PMC full texts)

| | Characters | ≈ Tokens |
|---|---|---|
| Median | 38,678 | 9,700 |
| 90th percentile | 56,276 | 14,100 |
| Longest | 67,100 | 16,800 |

54 of 57 have a labelled Methods section. Licences: mostly CC BY / CC BY-NC.

### 2.3 What that means for models and cost

- One full-text call ≈ **11,300 tokens** in (paper + rules). Gemini handles this easily (its context is far larger).
- **Groq's free tier cannot do full text**: its limit is 8,000 tokens per minute per model, less than one call. Groq stays useful for abstract-stage runs only.
- Cost at Gemini Flash prices: about **$0.10 per fully processed article** → about **$27 per run of the 268 articles** (upper bound $180 if all 1,855 reached full text). Abstract-only was about $0.02 per article.

## 3. What the research says

| Finding | Source |
|---|---|
| LLM screening from **full text**: pooled sensitivity 0.99, specificity 0.99. From title/abstract: 0.92 and 0.94. (18 studies; only 4 on full text, so the full-text estimate is less certain.) | [Meta-analysis, 2026](https://www.medrxiv.org/content/10.64898/2026.03.17.26348656v1.full) |
| Examples or step-by-step reasoning in the prompt raised sensitivity from 0.86 to 0.95. | same |
| Explicit inclusion/exclusion criteria improved abstract screening F₂ by 28.8%; researcher-written rationales improved full-text screening by 15%. | [SciLitBench, 2026](https://arxiv.org/abs/2609.05505) |
| Data extraction is the weak point: near-perfect for simple fields (0.97 for year) but poor for open-ended categories (0.37 overlap); the best models recovered only 30% of evaluation evidence. | same |
| Very long inputs give less consistent answers ("lost in the middle"). | [LLM review of reviews](https://arxiv.org/pdf/2409.04600) |
| A full-text pipeline that shows the passage behind each criterion judgment cut reviewer time from ~20 min to under 1 min per article with κ 0.82 between reviewers; authors still require human checks of borderline cases and version pinning. | [Auditable full-text screening, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13529844/) |

Take-aways: full text is where LLM screening is most accurate; give explicit criteria and worked examples; keep annotation labels closed and defined, with human checks on the hard categories; our papers (10–17k tokens) are short enough that the whole paper can be given at once.

## 4. Recommended design: two stages, like the human process (PRISMA)

```
Stage 1  Title/abstract screening   lenient: exclude only on a clear violation; unclear -> keep
            │ kept
            ▼
         Get the full text           PMC -> team's PDF -> free PDF -> (none: flagged)
            │
            ▼
Stage 2  Full-text eligibility       strict: every inclusion rule must be MET with a quote
            │                         from the paper; a clear violation excludes;
            │                         still unclear -> a human decides
            ▼ included
         Extraction + annotation     from the full text, with quotes and section names
            ▼
         Verifier -> human for anything uncertain
```

Why two stages rather than "just give everyone the full text":
- It is how the human team worked, so each stage can be compared with the matching human decision.
- It resolves the strict-vs-lenient problem from the first real run: **lenient where information is missing (abstract), strict where it is not (full text)**. The models' natural strictness becomes an asset at stage 2.
- It is cheaper: obviously irrelevant papers never cost a full-text call.

Stage 2 reuses the per-rule judgment form already built for stage 1 (met / not_met / unclear with a quote); only the decision rule differs. The reviewer panel, adjudicator, debate variant, verifier and human review all apply to stage 2 unchanged.

## 5. Making sure nothing fails

| Risk | Safeguard |
|---|---|
| No full text can be found | Source ladder: (1) PubMed Central XML [built], (2) **the team's PDFs** in `data/pdfs/<PMID>.pdf`, (3) free publisher PDF, (4) none → the article is **not excluded**: it is flagged "full text unavailable" and sent to a human, and listed in a `pdfs_needed.csv` to fetch through the library |
| A PDF converts to garbage (scans, two columns, tables) | Quality check after extraction (enough text, mostly words, has recognisable sections); on failure fall back to the next source and flag it |
| Paper too long for a model's limits | Check size before the call; whole paper when it fits (Gemini); otherwise Methods + Results first, then the rest up to the limit, and say so in the record |
| Model loses details in a long paper | Text is sent with section headings; instructions are repeated after the paper; each judgment must name the section and quote it; the verifier checks every quote against the full text |
| Rate limits, outages, bad output | Already built: throttling, 5 retries with back-off, structured output, response cache so a re-run only repeats what failed |
| A run stops halfway | Every finished call is cached; re-running the same command continues from where it stopped |
| Wrong key, model name or quota discovered mid-run | **Preflight check** before every run: keys valid, each model answers a tiny call, every article has a text source, sizes fit each model, estimated calls vs daily limits, estimated cost and time |
| A study silently dropped | Unchanged rules: failures are predicted "include"; split-vote exclusions and all "unclear at full text" go to a human |
| Preview models changing | Exact model IDs recorded per run; freeze before the test set |

## 6. What was built (all items done 2026-10-01)

| # | Item | Size |
|---|---|---|
| 1 | Full-text sources: PDF folder reader (text extraction + quality check), free-PDF fetch, source ladder, `pdfs_needed.csv` | medium |
| 2 | Stage 2 "full-text eligibility" step in the pipeline (strict decision rule, unclear → human) with its build document | medium |
| 3 | Extraction and annotation read the full text; quotes carry their section | small |
| 4 | Long-text handling: size check, section-priority fallback, instructions repeated after the paper | small |
| 5 | Preflight check command | small |
| 6 | Worked examples in prompts from the codebook's `examples` field (filled during calibration) | small |
| 7 | Scoring both stages: stage 1 vs abstract-stage gold, stage 2 vs full-text gold (the placeholder already has both columns) | small |
| 8 | Configs, tests, cost estimator and docs updated | small |

Testing before the professor's data arrives: the placeholder gold has both an abstract-stage and a full-text-stage decision per article, so both stages can be scored; but its papers are old and only ~12% have PMC full text, so the full-text path will be exercised mainly on a sample from the professor's 1,855 (mechanics only, no gold labels).

## 7. Decisions needed from the professor

1. **Did the human team screen titles/abstracts first and then read full papers?** And are the 268 the papers they read in full, or the ones they finally included?
2. **Can we have the PDFs they read** (a Zotero/EndNote export or a folder)? Without them, about a third of the papers have no full text for the agents.
3. **Were the category labels (annotation) assigned from the full text?**
4. Is Gemini access a paid/licensed project? Full-text runs do not fit free tiers.
