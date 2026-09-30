"""Codebook Curator: proposes clearer codebook wording from real disagreements.

Reads where the agents disagreed with the gold standard on the CALIBRATION set,
and proposes clarifications or borderline examples. Proposals are written to a
file for the professor to accept or reject; nothing is applied automatically.
"""

from __future__ import annotations

from adrd_review.agents.common import render_dimensions, render_screening_rules
from adrd_review.codebook import Codebook
from adrd_review.llm import LLM
from adrd_review.schemas import CuratorAnswer, Usage

SYSTEM = """\
You maintain the codebook (the written rules) for a scoping review of dementia screening research.
AI reviewers applied the codebook below to a calibration set of articles, and some of their answers
disagreed with the human gold standard. Your job is to find where the WORDING of the rules caused
those disagreements, and propose clearer wording or borderline examples.

CURRENT CODEBOOK (version {version})

{rules}

{dimensions}

Guidance:
- Look for patterns across several disagreements, not one-off errors.
- Prefer small, precise edits: a definition, a boundary, or a borderline example.
- Do not propose changes that would simply memorise the examples.
- Cite the PMIDs each proposal addresses."""


def render_disagreements(items: list[dict]) -> str:
    blocks = []
    for d in items:
        lines = [f"PMID {d['pmid']}: {d['title']}", f"  Field: {d['field']}   Human: {d['gold']}   Agents: {d['agent']}"]
        if d.get("rules"):
            lines.append(f"  Rules the agents cited: {', '.join(d['rules'])}")
        for r in d.get("reasons", [])[:3]:
            lines.append(f"  Agent reasoning: {r}")
        if d.get("abstract"):
            lines.append(f"  Abstract (start): {d['abstract'][:600]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def run(codebook: Codebook, disagreements: list[dict], llm: LLM) -> tuple[CuratorAnswer, Usage]:
    system = SYSTEM.format(
        version=codebook.version, rules=render_screening_rules(codebook), dimensions=render_dimensions(codebook)
    )
    user = f"DISAGREEMENTS ({len(disagreements)})\n\n" + render_disagreements(disagreements)
    result = llm.structured(CuratorAnswer, system, user, role="curator")
    return result.parsed, result.usage
