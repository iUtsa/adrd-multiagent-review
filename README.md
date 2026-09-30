# ADRD Multi-Agent Scoping Review

A LangGraph multi-agent system that runs an automated scoping review of PubMed-indexed ADRD (Alzheimer's disease and related dementias) screening literature. Its output is evaluated against a 3-reviewer human gold standard of 268 articles.

This project extends the earlier two-agent system (Extraction Agent + Annotation Agent, built on LangChain + Django REST). It adds:

- a **panel of 3 independent reviewer agents on different AI models**, mirroring the 3-reviewer human protocol,
- an **Adjudicator** that weighs the reviewers' evidence when they disagree, and a **human-in-the-loop** queue for low-confidence cases,
- an optional **debate variant**, tested against voting rather than assumed to help,
- a **Verifier** that checks every label against verbatim evidence,
- a **versioned codebook** with a Curator agent for iterative criteria refinement.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design, diagrams and evaluation plan, and [docs/10_DAY_PLAN.md](docs/10_DAY_PLAN.md) for the build schedule.

## Status

**All agents are built** (Days 1–8 of the [10-day plan](docs/10_DAY_PLAN.md)): PubMed tool, codebook, gold standard, frozen split, scorer, and every agent (triage, 3-reviewer panel, extraction, annotation, adjudicator, verifier, human review, Codebook Curator, debate) in one LangGraph pipeline covering E1–E4 and D. Everything runs today with a stand-in model. Models: **Gemini** (professor's licence), see [docs/MODELS.md](docs/MODELS.md). **Next: real Gemini runs** once access is set up ([docs/RUNNING_EXPERIMENTS.md](docs/RUNNING_EXPERIMENTS.md)). **The gold standard and codebook are currently a public placeholder** (Cochrane dementia-screening decisions from CLEF 2017) until the professor's data arrives; see [docs/GOLD_STANDARD.md](docs/GOLD_STANDARD.md). Progress log: [docs/PROGRESS.md](docs/PROGRESS.md).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,local]"
cp .env.example .env        # add API keys when available

python scripts/build_placeholder_gold.py   # gold standard + 268 articles into data/
python -m pytest                           # run the tests
python scripts/score_baselines.py          # score the non-AI baselines
python -m adrd_review.graph.runner configs/runs/E4_fake.yaml   # run the full proposed system (no key needed)
```

## Docs

| File | What it covers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full system design, diagrams, evaluation plan |
| [docs/10_DAY_PLAN.md](docs/10_DAY_PLAN.md) | Day-by-day build schedule |
| [docs/PROGRESS.md](docs/PROGRESS.md) | What has been built and how it was checked |
| [docs/TOOLS.md](docs/TOOLS.md) | How to use each tool (PubMed, ...) |
| [docs/GOLD_STANDARD.md](docs/GOLD_STANDARD.md) | Gold file format, the placeholder, the frozen split, how to swap in the real data |
| [docs/RUNNING_EXPERIMENTS.md](docs/RUNNING_EXPERIMENTS.md) | **The order to run everything**, one command per step |
| [docs/AGENTS.md](docs/AGENTS.md) | The agents, model factory, cache, pipeline, run configs and outputs |
| [docs/agents/](docs/agents/README.md) | **Build document for every agent** + exact prompts |
| [docs/MODELS.md](docs/MODELS.md) | Which Gemini model does what; Gemini access setup; optional Groq |
| [docs/SEARCH.md](docs/SEARCH.md) | The professor's PubMed search |
| [docs/EVALUATION.md](docs/EVALUATION.md) | How runs are scored, every metric, baseline results |
| [docs/LOCAL_MODELS.md](docs/LOCAL_MODELS.md) | Free local models with Ollama (on standby) |

## Layout

```
configs/
  codebook.yaml           review rules and labels (placeholder)
  runs/                   run configs: which model each agent uses
  search/                 the professor's PubMed query
  splits/                 frozen calibration/test splits
src/adrd_review/
  schemas.py              shared records: Article, answer forms, Vote, Decision, Usage
  llm.py                  model factory (LangChain) + response cache
  consensus.py            combine reviewers' votes; panel agreement
  human_review.py         answer paused decisions and finish a run
  curate.py               run the Codebook Curator on a calibration run
  freeze.py               freeze configs before test runs
  fake_model.py           keyword stand-in model for runs without a key
  codebook.py             load + validate the codebook
  gold.py                 load + validate the gold standard
  split.py                make + freeze the calibration/test split
  tools/pubmed.py         PubMed search / fetch / lookup
  agents/                 triage, screening, extraction, annotation, adjudicator, verifier, debate, human, curator
  graph/                  state.py, pipeline.py (LangGraph), runner.py
  eval/metrics.py         P/R/F1, Cohen & Fleiss kappa, WSS, WSS@95, cost
  eval/score.py           score a predictions file against the gold standard
data/                     gold standard, corpus, run outputs (git-ignored)
scripts/                  build_placeholder_gold.py, score_baselines.py, render_prompts.py, check_ollama.py
notebooks/                analysis
tests/                    pytest tests (+ fixtures/)
docs/                     design, plan, progress, tool docs
```
