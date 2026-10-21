# Diagrams and charts

Everything here is generated, never drawn by hand, so it can be redrawn after any change.

## Architecture diagrams (Mermaid, sources in `src/`)

| Image | Shows | Source |
|---|---|---|
| [system-overview.png](system-overview.png) | the whole system as built: protocol, ingestion, the three stages, verifier, second look, professor, outputs | `src/system-overview.mmd` |
| [pipeline-as-run.png](pipeline-as-run.png) | the LangGraph path the free run takes; nodes switched off in this run are grey dashed | `src/pipeline-as-run.mmd` |
| [langgraph-full.png](langgraph-full.png) | the complete compiled graph, exported from the code (`graph.get_graph().draw_mermaid()`) | `src/langgraph_PROF_search_free.mmd` |
| [consensus-by-stage.png](consensus-by-stage.png) | one reviewer's rule marks and quote check; the panel rule at titles, abstracts and full text | `src/consensus-by-stage.mmd` |
| [agent-interaction-sequence.png](agent-interaction-sequence.png) | messages between the runner, reviewers A/B/C, verifier, second look and professor | `src/agent-interaction-sequence.mmd` |
| [model-routing.png](model-routing.png) | which model and host each reviewer uses per stage, overflow hosts, and models that were tried and dropped | `src/model-routing.mmd` |
| [run-loop.png](run-loop.png) | the unattended loop between tries, and what happens on every model call (cache, limits, overflow, LimitReached) | `src/run-loop.mmd` |
| [timeline.png](timeline.png) | the free run day by day: stages, model changes, nights the Mac slept | `src/timeline.mmd` |
| [experiment-ladder.png](experiment-ladder.png) | conditions E0–E4, D, three stages, the professor's search, second look, with status | `src/experiment-ladder.mmd` |

Colours (same meaning in every diagram and chart): blue = used / passed / included; grey dashed = built but off in this run, or tried and dropped, or not counted; orange = excluded; aqua = human review. Dashed blue outline = a paid host.

Redraw one (Node is needed; the renderer is fetched on first use):

```bash
cd docs/diagrams
npx -y @mermaid-js/mermaid-cli@11.4.2 -i src/system-overview.mmd -o system-overview.png \
    -c src/mermaid-config.json -b "#fcfcfb" -s 2 -w 1600
```

Re-export the complete graph after changing `graph/pipeline.py`:

```bash
python - <<'EOF'
from adrd_review.graph.runner import load_run_config, build_graph
from adrd_review.codebook import load_codebook
cfg = load_run_config("configs/runs/PROF_search_free.yaml")
g, _ = build_graph(cfg, load_codebook(cfg.codebook), [])
open("docs/diagrams/src/langgraph_PROF_search_free.mmd", "w").write(g.get_graph().draw_mermaid())
EOF
```

## Result charts (`results/`, from the run's data)

Made by `python -m adrd_review.eval.report_figures <run folder> --out docs/diagrams/results --billed <amount OpenRouter billed>`. Each chart has a CSV next to it with the plotted numbers. Free full text only: paid-access papers are in no chart (their number is printed as a side note).

| Chart | Shows |
|---|---|
| [stage_outcomes.png](results/stage_outcomes.png) | papers at each stage: passed / excluded / for human review / retry |
| [exclusion_reasons.png](results/exclusion_reasons.png) | the main rule behind each exclusion, per stage |
| [how_decided.png](results/how_decided.png) | unanimous / majority / no majority, per stage |
| [panel_agreement.png](results/panel_agreement.png) | Fleiss' κ and % unanimous among A, B, C, per stage |
| [reviewer_votes.png](results/reviewer_votes.png) | how often each reviewer voted exclude (titles, abstracts) or unclear (full text) |
| [model_calls.png](results/model_calls.png) | answers per stage, host and model (free vs paid) |
| [openrouter_cost.png](results/openrouter_cost.png) | recorded cost of paid calls per model, and the amount billed |
| [progress.png](results/progress.png) | papers still open after each try of the unattended loop |
| [second_look.png](results/second_look.png) | outcome of the second look (once `second_look.csv` exists) |

Chart style follows a validated palette (colour-blind safe for these three hues; every segment carries a direct label; one axis per chart).

## Earlier design

[design-v0.3/](design-v0.3/) holds the four diagrams of the design proposal (2026-09-25), before the system was built: kept for the record.
