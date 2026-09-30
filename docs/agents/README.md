# Agent build documentation

One document per agent. Every agent added to the system gets a document here, written when the agent is built, following the template below.

| Agent | Stage | Status | Build doc | Exact prompt |
|---|---|---|---|---|
| Triage | 1. Ingestion | Built (Day 4) | [triage.md](triage.md) | [prompts/triage.md](prompts/triage.md) |
| Screening reviewer | 2. Screening panel | Built (Day 4); used ×3 in the panel (Day 5) | [screening-reviewer.md](screening-reviewer.md) | [prompts/screening.md](prompts/screening.md) |
| Extraction | 3. Extraction | Built (Day 4) | [extraction.md](extraction.md) | [prompts/extraction.md](prompts/extraction.md) |
| Annotation reviewer | 4. Annotation panel | Built (Day 4); used ×3 in the panel (Day 5) | [annotation-reviewer.md](annotation-reviewer.md) | [prompts/annotation.md](prompts/annotation.md) |
| Reviewer panel + consensus | 2 and 4 | Built (Day 5) | [reviewer-panel.md](reviewer-panel.md) | uses the reviewer prompts + personas |
| Adjudicator | 3. Consensus | Planned (Day 6) | — | — |
| Verifier | 5. Verification | Planned (Day 6); quote check already runs inside each reviewer | — | — |
| Human review | 5. Verification | Planned (Day 7) | — | — |
| Codebook Curator | 0. Protocol | Planned (Day 7) | — | — |
| Debate (variant D) | 3. Consensus | Planned (Day 8) | — | — |

The `prompts/` pages are **generated** from the live code and codebook by `python scripts/render_prompts.py`, so they always show exactly what the model receives. Re-run it after changing an agent, persona or the codebook.

## Template

Every build document has these sections:

1. **Purpose**: what the agent does, in one paragraph, and which weakness of the earlier 2-agent system it addresses.
2. **Where it sits**: its place in the pipeline and what runs before and after it.
3. **Inputs**: exactly what the model is shown.
4. **Output**: the answer form, field by field.
5. **Prompt**: how it is built, with a link to the generated exact prompt.
6. **Model and settings**: default model, temperature, limits, and how to change them.
7. **Checks after the model answers**: what the code verifies, drops or flags.
8. **Failure handling**: what happens on errors, bad output or missing data.
9. **Design decisions**: choices made and why, with evidence.
10. **Tests**: which tests cover it.
11. **How to run and inspect it**: commands and where its output appears.
12. **Limitations and planned changes**.
13. **Change log**.
