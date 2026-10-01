# The professor's PubMed search

Sent by Prof. Wei on 2026-09-30 ("please adjust the Boolean expression when you use it"). Stored in `configs/search/prof_query.yaml` (original verbatim + corrected).

## What was corrected

The dementia group was not closed and had no `AND` before the machine-learning group:

```
( "dementia"[tiab] OR ... OR "Alzheimer Disease"[MeSH Terms] ("machine learning"[tiab] ...
```

The corrected version closes the dementia group and joins the four groups with `AND`. Terms, date range (2015/01/01–2025/07/31) and language (English) are unchanged.

## Result

| Version | PubMed count |
|---|---|
| Original | 1,855 |
| Corrected | 1,855 |

PubMed's parser inserted the missing `AND` itself, so both return the same set. We cite the corrected version because it is unambiguous. **To confirm with the professor.**

The 1,855 PMIDs are saved in `data/search/prof_query_pmids.txt` (git-ignored) with the run date in `prof_query_meta.json`. Re-run:

```bash
python -c "import yaml; from adrd_review.tools.pubmed import search; ids, n = search(yaml.safe_load(open('configs/search/prof_query.yaml'))['corrected']); print(n)"
```

## What it tells us

- The review topic is **AI / machine learning for detecting, diagnosing, monitoring or predicting ADRD in older adults**. The placeholder gold standard and codebook (Cochrane reviews of cognitive tests such as MMSE and MoCA) are a different topic. They still test the screening mechanics; the agents' prompts follow the real codebook automatically once it arrives.
- The human team's 268 articles are presumably drawn from these 1,855. Once we have their PMIDs, the retrieval check (task 9.3) measures how many the search recovers.
- `"AI"[tiab]` also matches unrelated uses of the abbreviation (for example aromatase inhibitor), a keyword trap the screening agents will need to handle.

## Collecting the papers (abstracts + full text)

```bash
python scripts/fetch_search_corpus.py            # all saved search results
python scripts/fetch_search_corpus.py --rerun    # run the query again first
```

Downloads every article's title and abstract to `data/corpus_search/`, then looks for each full paper (team PDF in `data/pdfs/`, PubMed Central, open-access PDF). Safe to stop and re-run. Writes `data/search/fulltext_report.csv` (source and size per article, or why none) and `data/search/pdfs_needed.csv` (papers with no full text: link and the file name to save the PDF as).

### Result of the collection (2026-10-01)

| | Count |
|---|---|
| Articles found | 1,855 |
| Abstracts downloaded | 1,855 (3 without an abstract) |
| **Full text obtained** | **1,118 (60%)**: 1,112 PubMed Central, 6 open-access PDFs |
| No full text | 737 (40%): 591 paywalled, 143 free copy needing a manual download, 3 other |

Full text found by publication year: 2015 24/51 · 2016 23/61 · 2017 41/79 · 2018 49/91 · 2019 75/139 · 2020 102/157 · 2021 116/171 · 2022 87/123 · 2023 82/130 · 2024 230/396 · 2025 288/448 · 2026 1/9.

Typical full text: ~34,000 characters (~8,600 tokens). 22 papers exceed 80,000 characters; the longest is ~1.1 million characters. The pipeline's size cap sends Methods and Results first for those.
