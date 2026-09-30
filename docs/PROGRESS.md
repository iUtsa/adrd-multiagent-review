# Progress log

What was built, when, and how it was checked. Newest first. Task numbers refer to [10_DAY_PLAN.md](10_DAY_PLAN.md).

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
