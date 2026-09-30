# Tools

Tools are plain code (no AI) that agents and scripts call. Each one lives in `src/adrd_review/tools/`.

---

## PubMed tool — `tools/pubmed.py`

Downloads articles from PubMed and saves each one as a JSON file that every agent reads. It also runs PubMed searches and finds PMIDs from DOIs or titles.

It uses NCBI's free **E-utilities** service. No API key is required.

### What it can do

| Command | What it does | When we use it |
|---|---|---|
| `fetch` | Downloads title, abstract and details for a list of PMIDs and saves each to `data/corpus/<pmid>.json` | Day 2: download the 268 gold-standard articles |
| `search` | Runs a PubMed query and lists the matching PMIDs | Day 9: retrieval check (does the search find the 268?) |
| `lookup` | Finds the PMID for a DOI or an exact title | If the professor's file has DOIs/titles instead of PMIDs |

### How to run it

From the repo root:

```bash
source .venv/bin/activate

# Download articles by PMID
python -m adrd_review.tools.pubmed fetch --pmids 41503471 41984500

# Download all PMIDs from a file (.txt one per line, or .csv / .xlsx with a PMID column)
python -m adrd_review.tools.pubmed fetch --pmids-file data/gold/gold.xlsx --column PMID

# Re-download even if already saved
python -m adrd_review.tools.pubmed fetch --pmids-file ids.txt --refresh

# Search PubMed (optionally limit by publication date)
python -m adrd_review.tools.pubmed search '"dementia"[tiab] AND "screening"[tiab]' --mindate 2015 --maxdate 2024 --out ids.txt

# Find a PMID from a DOI or title
python -m adrd_review.tools.pubmed lookup --doi 10.64898/2025.12.27.25343088
python -m adrd_review.tools.pubmed lookup --title "Exact article title here"
```

From Python:

```python
from adrd_review.tools.pubmed import fetch_articles, search, lookup_pmid

articles, report = fetch_articles(["41503471", "41984500"])
print(report.summary())
print(articles["41503471"].abstract)
```

### What `fetch` prints

```
Requested:   5
Downloaded:  5
From cache:  0
Missing:     0
No abstract: 0
```

- **Missing:** PubMed returned nothing for that PMID (typo, withdrawn record, or a book chapter rather than a journal article). The command exits with an error code so this is never silently ignored.
- **No abstract:** the article exists but has no abstract (letters, editorials, some older papers). The Triage agent will flag these for full text.

### What one saved article looks like

Each file in `data/corpus/` follows the `Article` record in `src/adrd_review/schemas.py`:

| Field | Example |
|---|---|
| `pmid` | `"41503471"` |
| `title` | `"Acoustic Analysis of Primary Care Patient-provider Conversations to Screen for Cognitive Impairment."` |
| `abstract` | Full abstract. Structured abstracts keep their section labels: `"IMPORTANCE: ...\n\nOBJECTIVE: ..."` |
| `abstract_sections` | The same abstract split into `{label, text}` parts |
| `journal`, `year` | `"medRxiv : ..."`, `2025` |
| `authors` | `["Colonel, Joseph T", ...]` |
| `doi`, `pmcid` | `"10.64898/..."`, `"PMC12772652"` (PMCID means free full text may exist in PubMed Central) |
| `language`, `publication_types` | `["eng"]`, `["Journal Article", "Preprint"]` |
| `mesh_terms`, `keywords` | Subject headings and author keywords, when PubMed has them |
| `fetched_at` | When it was downloaded (UTC) |

### Built-in safeguards

- **Caching.** An article already in `data/corpus/` is read from disk, never re-downloaded (unless `--refresh`). Reruns are instant and PubMed is not hit again.
- **Rate limits.** Waits between requests to stay under NCBI's limit: 3/second without a key, 10/second with `NCBI_API_KEY` in `.env`.
- **Retries.** Network errors and server errors are retried 3 times with increasing waits.
- **Batching.** Sends at most 200 PMIDs per request, so all 268 download in 2 requests.
- **Clean PMID input.** Strips spaces, removes the `.0` Excel adds to numbers, drops blanks and duplicates, and stops with a clear error if an entry is not a number (for example a DOI in the PMID column).
- **Correct IDs.** DOI and PMCID are read from the article's own ID list, not from its reference list (which contains the cited papers' DOIs).

### Optional: NCBI API key

Not needed for 268 articles. To raise the rate limit, get a free key at https://www.ncbi.nlm.nih.gov/account/settings/ and add to `.env`:

```
NCBI_API_KEY=your-key
NCBI_EMAIL=you@example.com
```

### Tests

```bash
python -m pytest tests/test_pubmed.py            # offline tests (fixture file)
RUN_NETWORK_TESTS=1 python -m pytest tests/      # also runs one live PubMed test
```

The offline tests use `tests/fixtures/pubmed_sample.xml`, a small made-up PubMed response covering the tricky cases: structured abstracts, italic tags in titles, no abstract, unusual dates, reference-list IDs, and book records.


---

## PubMed Central full-text tool — `tools/pmc.py`

Fetches an article's full text from PubMed Central (free, no key) when it has a PMCID, and keeps the body by section (Methods, Results, …). Back matter (references, acknowledgements, funding, author contributions, declarations) is left out. A section called "Reference standard" is **kept**: only a section titled exactly "References" is dropped (this was a bug caught by the tests).

```bash
python -m adrd_review.tools.pmc 23172761          # by PMID; prints sections, size, licence
```

**In the pipeline** (run config):

| `fulltext:` | Behaviour |
|---|---|
| `never` (default) | Abstract only, like title/abstract screening by the human team |
| `when_needed` | Fetch when triage says the abstract is not enough and the article has a PMCID |
| `always` | Fetch for every article with a PMCID |

The agents then see an excerpt (default 12,000 characters, cut at a sentence end; `fulltext_max_chars:`) after the abstract, and quotes may come from it (`source: fulltext`). A failed fetch never fails the article; the outcome is recorded per article in `fulltext_info` (`used`, or why not: off / not needed / no PMCID / no PMC full text / fetch failed). Cached in `data/fulltext/`.

Which setting to use for the study depends on question 5 for the professor (were the human reviewers screening abstracts only?). On the 40 calibration articles of the placeholder, 5 have PubMed Central full text (older papers rarely do).

Tests: `tests/test_fulltext.py`.
