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

Building (Day 2 of the [10-day plan](docs/10_DAY_PLAN.md)). The human review is complete. Done so far: Python setup and the PubMed tool. See [docs/PROGRESS.md](docs/PROGRESS.md).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add API keys when available

python -m pytest            # run the tests
python -m adrd_review.tools.pubmed fetch --pmids 41503471   # download an article
```

## Docs

| File | What it covers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full system design, diagrams, evaluation plan |
| [docs/10_DAY_PLAN.md](docs/10_DAY_PLAN.md) | Day-by-day build schedule |
| [docs/PROGRESS.md](docs/PROGRESS.md) | What has been built and how it was checked |
| [docs/TOOLS.md](docs/TOOLS.md) | How to use each tool (PubMed, ...) |

## Layout

```
configs/                  codebook + panel configuration
src/adrd_review/
  schemas.py              shared records (Article, ...)
  tools/pubmed.py         PubMed search / fetch / lookup
  agents/                 one module per agent
  graph/                  LangGraph state, subgraphs, pipeline
  eval/                   metrics and reports
data/                     gold standard, corpus, run outputs (git-ignored)
notebooks/                analysis
tests/                    pytest tests (+ fixtures/)
docs/                     design, plan, progress, tool docs
```
