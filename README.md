# ADRD Multi-Agent Scoping Review

A LangGraph multi-agent system that runs an automated scoping review of PubMed-indexed ADRD (Alzheimer's disease and related dementias) screening literature. Its output is evaluated against a 3-reviewer human gold standard of 268 articles.

This project extends the earlier two-agent system (Extraction Agent + Annotation Agent, built on LangChain + Django REST). It adds:

- a **panel of 3 independent reviewer agents on different AI models**, mirroring the 3-reviewer human protocol,
- an **Adjudicator** that weighs the reviewers' evidence when they disagree, and a **human-in-the-loop** queue for low-confidence cases,
- an optional **debate variant**, tested against voting rather than assumed to help,
- a **Verifier** that checks every label against verbatim evidence,
- a **versioned codebook** with a Curator agent for iterative criteria refinement.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design, diagrams and evaluation plan.

## Status

Design phase. The human review is complete; the agent side is being built.

## Layout

```
configs/           codebook + panel configuration
src/adrd_review/   agents, tools, LangGraph graphs, evaluation
data/              gold standard, corpus, run outputs (git-ignored)
notebooks/         analysis
tests/
docs/              architecture and design notes
```
