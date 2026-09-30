# Gold standard

The gold standard is the set of human decisions that every agent run is scored against.

> **Current status: PLACEHOLDER.** Until the professor's file arrives we use a public gold standard built from real Cochrane screening decisions (details below). Everything downstream reads the same standard format, so replacing it is a one-step swap.

---

## The standard format

One CSV file, one row per article. `src/adrd_review/gold.py` reads and validates it.

| Column | Required | Values |
|---|---|---|
| `pmid` | yes | PubMed ID |
| `screening` | yes | `include` or `exclude` (title/abstract decision) |
| `fulltext` | no | `include` or `exclude` (final decision after reading the full text) |
| one column per annotation dimension (e.g. `instrument`, `setting`) | no | codebook label IDs; several separated by `;`; empty = none |
| `<field>__<reviewer>` e.g. `screening__r1`, `instrument__r2` | no | each human reviewer's own answer (needed for the human Fleiss' κ) |
| `source` | no | free text: where the row came from |

Example:

```csv
pmid,screening,fulltext,instrument,setting,screening__r1,screening__r2,screening__r3,source
10812327,include,exclude,IQCODE;MMSE,primary_care,include,include,exclude,prof-2026
22840457,exclude,exclude,,,exclude,exclude,exclude,prof-2026
```

The loader checks everything at once and lists every problem it finds:
- missing `pmid`/`screening` columns
- PMIDs that are not numbers, and duplicate PMIDs
- decisions that are not include/exclude (it accepts `Included`/`Excluded` too)
- labels that are not in the codebook
- more than one label in a single-label dimension
- annotation labels on an excluded article

---

## The placeholder (current)

Built by `scripts/build_placeholder_gold.py` from the **CLEF 2017 Technology Assisted Review** dataset (github.com/CLEF-TAR/tar, MIT licence, pinned to commit `dbc13d0`). CLEF is the standard public benchmark for automated screening research. It contains the real title/abstract and full-text decisions made by Cochrane review teams.

We use its six Cochrane reviews of **dementia screening instruments**:

| Review | Instrument | Setting / population |
|---|---|---|
| CD010775 | MoCA | memory clinics, hospital, primary care, community |
| CD010783 | MMSE | people with MCI, followed over time |
| CD011145 | MMSE | people 65+ in community or primary care, not previously tested |
| CD010860 | Mini-Cog | community only |
| CD010771 | IQCODE | primary care |
| CD010772 | IQCODE | secondary care |

**How the 268 articles were chosen:** 90 includes and 178 excludes (same total as the human set), sampled in turn from each review so all six are represented (seed `20260930`). Only articles PubMed actually returns were kept.

**What the labels mean:**
- `screening`: include if **any** of the six reviews included it at title/abstract stage. These are real human decisions.
- `fulltext`: include if any review included it after full-text reading.
- `instrument`: the instrument(s) of the review(s) that included it.
- `setting`: the setting of the review(s) that included it. The MoCA review covered mixed settings, so it adds no setting label.

**Current contents:** 268 articles, 90 include / 178 exclude (32 included after full text). Instruments: MMSE 45, IQCODE 36, MoCA 11, Mini-Cog 7 (9 studies have two). Published 1971–2014; 9 have no abstract.

**Known limitations** (why it is only a placeholder):
1. Six separate reviews merged into one review question, so a few labels may not fit the merged codebook perfectly.
2. `instrument` and `setting` are derived from which review included the study, not coded by someone reading it. They are positives-only: a study may evaluate other instruments too.
3. No per-reviewer answers, so the human Fleiss' κ cannot be computed.
4. No exclusion reasons.

Screening results on the placeholder are meaningful. Annotation results are only a pipeline check.

### Rebuild it

```bash
python scripts/build_placeholder_gold.py
```

Writes `data/gold/gold.csv`, `data/gold/gold_meta.json` (provenance and limitations) and downloads the 268 articles to `data/corpus/`. The `data/` folder is git-ignored; the script and the pinned commit make it reproducible.

---

## The split (frozen)

`configs/splits/placeholder.json`, made with `python -m adrd_review.split`:

| Set | Articles | Include | Exclude | Used for |
|---|---|---|---|---|
| calibration | 40 | 13 | 27 | all tuning: prompts, codebook, thresholds |
| test | 228 | 77 | 151 | final results only (Day 9) |

- Stratified by screening decision, so both sets have the same include rate.
- Fixed seed `20260930`.
- The file records the SHA-256 of the gold file. A test fails if the gold file changes after the split was made.
- Running the split command again **refuses to overwrite** it. `--force` exists but invalidates earlier results.

---

## Swapping in the professor's gold standard

1. Put the file in `data/gold/` (git-ignored).
2. Replace `configs/codebook.yaml` with the human codebook (same structure; keep label IDs identical to the values in the gold file).
3. Convert the file to the standard format above (a small one-off script, since the column names will differ). If it has DOIs or titles instead of PMIDs, use `python -m adrd_review.tools.pubmed lookup`.
4. Check it: the loader lists any problems.
   ```bash
   python -c "from adrd_review.codebook import load_codebook; from adrd_review.gold import load_gold, summarize; print(summarize(load_gold('data/gold/gold.csv', load_codebook())))"
   ```
5. Download the articles: `python -m adrd_review.tools.pubmed fetch --pmids-file data/gold/gold.csv`
6. Freeze a new split: `python -m adrd_review.split --gold data/gold/gold.csv --out configs/splits/human.json --name human`
7. Keep the placeholder split file for reference; results on it stay comparable across runs.
