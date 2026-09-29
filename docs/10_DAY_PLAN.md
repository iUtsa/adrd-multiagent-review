# 10-Day Build Plan (Sep 29 – Oct 8, 2026)

Goal: by Day 10, have the multi-agent system running on all 268 articles, scored against the human gold standard, with a short report for the professor.

The one rule that protects the results: **tune only on the 40 calibration articles. Freeze everything on Day 8. The 228 test articles are touched only on Day 9.**

Legend: each task says **why** it matters and when it is **done**.

---

## Day 1 · Tue Sep 29 · Kickoff: data, access, setup

| # | Task | Why | Done when |
|---|---|---|---|
| 1.1 | Send the professor the open questions (§10 of ARCHITECTURE.md) | Days 2–3 depend on the answers | Message sent |
| 1.2 | Get the gold-standard file(s) and the human codebook into `data/gold/` | Everything is scored against these | Files open cleanly in pandas |
| 1.3 | Get API keys: Anthropic, Google AI Studio (free tier), OpenAI, one open-model host (e.g. OpenRouter). Put them in `.env`. Apply for research credits. | Agents cannot run without them | A test call to each provider works |
| 1.4 | Python environment: `langgraph`, `langchain-anthropic`, `langchain-google-genai`, `langchain-openai`, `pydantic`, `pandas`, `scikit-learn`, `statsmodels`, `python-dotenv`, `pytest` | Same setup every day after this | `import langgraph` works |
| 1.5 | (Optional) Install Ollama + a small Qwen model on the Mac | Free local testing | `ollama run` answers |

## Day 2 · Wed Sep 30 · Rulebook, articles, split

| # | Task | Why | Done when |
|---|---|---|---|
| 2.1 | Turn the human codebook into `configs/codebook.yaml`: every rule gets an ID, a definition and 1–2 borderline examples | Agents and humans must use the exact same rules and labels | Every label in the gold file exists in the codebook |
| 2.2 | Write `tools/pubmed.py`: fetch title, abstract and metadata for all 268 PMIDs and cache them in `data/corpus/` | Agents read these, and caching avoids re-downloading | 268 JSON files saved |
| 2.3 | Load the gold labels into one clean table (PMID, screening decision, each category, each reviewer's labels) | The scorer needs one tidy format | Table has 268 rows |
| 2.4 | Freeze the split: 40 calibration / 228 test, balanced by include/exclude, fixed random seed | Prevents tuning on the test set | Split file committed |

## Day 3 · Thu Oct 1 · Build the scorer first

| # | Task | Why | Done when |
|---|---|---|---|
| 3.1 | `eval/metrics.py`: precision, recall, F1, Cohen's κ, Fleiss' κ, WSS@95, time and cost summary | You must be able to measure before you build | Functions exist |
| 3.2 | Unit tests with small made-up examples whose answers are known | Proves the scorer is correct | `pytest` passes |
| 3.3 | Score the old 2-agent system (E0) | Checks the scorer against published numbers | Numbers match the paper |
| 3.4 | Compute the humans' own Fleiss' κ | This is the number agents will be compared with | Number recorded |

## Day 4 · Fri Oct 2 · First working pipeline (E1, one agent)

| # | Task | Why | Done when |
|---|---|---|---|
| 4.1 | `graph/state.py`: the shared record every agent reads/writes (article, votes, quotes, decisions) | Every node depends on it | Schemas validate a sample |
| 4.2 | Model factory + response cache (any model built from `panel.yaml`; every answer saved) | Swap models in config; reruns are free | Second run of the same input costs $0 |
| 4.3 | Agents: triage, screening reviewer, extraction, annotation reviewer | The basic pipeline | Each works on one article |
| 4.4 | `graph/pipeline.py`: triage → screen → extract → annotate in LangGraph, plus a batch runner | Runs many articles safely | 40 articles run end to end |
| 4.5 | Run E1 on the 40 calibration articles with a free model and score it | First real numbers | Scores printed |

## Day 5 · Sat Oct 3 · The 3-reviewer panel (E2)

| # | Task | Why | Done when |
|---|---|---|---|
| 5.1 | `configs/panel.yaml`: 3 reviewers on 3 different model families, exact versions pinned | Model diversity is where the accuracy gain comes from | File committed |
| 5.2 | `Send` fan-out to the 3 reviewers, collect votes, agreement check, weighted vote | The core of the multi-agent design | 3 votes stored per article |
| 5.3 | Package it as one reusable consensus subgraph (screening + each annotation category) | Same logic used twice | Both stages use it |
| 5.4 | Run E2 on calibration; compare the agents' Fleiss' κ with the humans' | First agent-vs-human reliability comparison | Table produced |

## Day 6 · Sun Oct 4 · Adjudicator and fact-checker (E3)

| # | Task | Why | Done when |
|---|---|---|---|
| 6.1 | Adjudicator node (strongest model, not on the panel): weighs votes and quotes, returns decision + rule + confidence | Resolves disagreements with reasons | Works on split cases |
| 6.2 | Recall guard: a split-vote exclusion always goes to a human | Never silently drop a relevant study | Rule tested |
| 6.3 | Verifier: code checks each quote really exists; small model checks the rule fits | Catches made-up evidence | Fake quote is caught in a test |
| 6.4 | Run E3 on calibration; compare with E2 | Does the adjudicator help? | Scores printed |

## Day 7 · Mon Oct 5 · Human-in-the-loop (E4) + better rules

| # | Task | Why | Done when |
|---|---|---|---|
| 7.1 | `human_review` node using `interrupt()` + SQLite checkpointer; simple command-line screen to answer | The pipeline pauses for a person on hard cases | Pause → answer → resume works |
| 7.2 | For scoring, the "human" answer on escalated items is the gold label; count how many were escalated | Measures accuracy vs. human effort without new human work | Escalation count reported |
| 7.3 | Codebook Curator reviews calibration disagreements and proposes clearer rules; professor approves → codebook v2 | Fixes the "criteria refinement" weakness | v2 committed |
| 7.4 | Re-run E1–E4 on calibration with codebook v2 | See if clearer rules helped | Scores compared |

## Day 8 · Tue Oct 6 · Debate test + FREEZE

| # | Task | Why | Done when |
|---|---|---|---|
| 8.1 | Debate node (variant D): 1–2 rounds, vote counts hidden, minority argues first, new evidence required to change a vote | Tests whether agents talking helps | Works on split cases |
| 8.2 | Run D on calibration; compare with E3; decide keep or drop | Only keep what helps | Decision written down |
| 8.3 | **FREEZE** prompts, codebook, models, thresholds; tag `v1.0-frozen` in git | Keeps the final test fair | Tag pushed |
| 8.4 | Estimate the cost of the final runs | Stay within budget | Estimate approved |

## Day 9 · Wed Oct 7 · Final runs on the test set

| # | Task | Why | Done when |
|---|---|---|---|
| 9.1 | Run E1, E2, E3, E4 and D on the 228 test articles, 3 repeats each (reuse cached reviewer answers across conditions) | The actual results | All runs finished |
| 9.2 | Record time and cost for each condition | Compare with the 47.7× result | Logged |
| 9.3 | Retrieval check: does the PubMed search find the 268 articles? | Measures the search step separately | Recall number recorded |
| 9.4 | Check for failed calls or missing fields and rerun those | Clean data | 0 failures remain |

## Day 10 · Thu Oct 8 · Analysis and report

| # | Task | Why | Done when |
|---|---|---|---|
| 10.1 | Results tables: accuracy per condition, κ vs humans, agent vs human Fleiss' κ, WSS@95, escalation rate, time and cost, run-to-run κ | The main findings | Tables saved |
| 10.2 | Error analysis: sort agent mistakes into unclear rule / keyword trap / wrong topic / missing info | Explains *why* agents fail | Sample coded |
| 10.3 | Figures: condition comparison chart, accuracy vs. human-effort curve | For slides and the paper | Figures saved |
| 10.4 | Short report + slides for the professor; push code and docs | Hand-off | Sent |

---

## If you fall behind, cut in this order

1. Debate variant D (Day 8.1–8.2)
2. Codebook Curator (Day 7.3–7.4); refine the codebook by hand instead
3. Repeats: 3 → 2

Never cut: the scorer (Day 3), the frozen split, or the freeze on Day 8.
