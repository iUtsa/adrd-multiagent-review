"""Slide-sized (16:9) system architecture: components and how they connect, no counts or file names.

    python docs/diagrams/src/architecture_slide.py     # writes docs/diagrams/architecture-slide.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SURFACE, INK, INK2, PANEL, PANEL_EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#f4f3f0", "#c9c8c2"
BLUE, BLUE_BG, ORANGE_BG, ORANGE, AQUA_BG, AQUA = "#2a78d6", "#e3eefc", "#fde6dc", "#eb6834", "#d9f3e9", "#1baf7a"

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor(SURFACE)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")


def panel(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=PANEL, ec=PANEL_EDGE, lw=1.2))
    ax.text(x + 0.2, y + h - 0.3, title, fontsize=15, fontweight="bold", color=INK, va="top")


def box(x, y, w, h, text, fc=BLUE_BG, ec=BLUE, size=13, bold_first=True, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.6))
    lines = text.split("\n")
    for i, line in enumerate(lines):
        dy = (len(lines) - 1) / 2 - i
        ax.text(x + w / 2, y + h / 2 + dy * 0.36, line, ha="center", va="center", fontsize=size if i else size + 0.5,
                fontweight="bold" if (i == 0 and bold_first) else "normal", color=color)


def arrow(p, q, both=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=18, lw=1.8,
                                 color=INK2, connectionstyle=f"arc3,rad={rad}", shrinkA=4, shrinkB=4))


# Row 1: inputs -> pipeline -> outputs
panel(0.3, 5.9, 3.2, 2.8, "Inputs")
box(0.55, 7.0, 2.7, 0.95, "PubMed records\nthe search results", fc=SURFACE, ec=INK2)
box(0.55, 6.05, 2.7, 0.8, "Review criteria\nI1–I4, E1–E4 · age 65+", fc=SURFACE, ec=INK2, size=12)

panel(4.0, 5.9, 7.6, 2.8, "LangGraph pipeline (orchestrator)")
for i, (t, sub) in enumerate((("1 · Titles", "all 3 must exclude"), ("2 · Abstracts", "majority vote"),
                               ("3 · Full text", "majority + re-checks"))):
    x = 4.3 + i * 2.45
    box(x, 6.3, 2.1, 1.5, f"{t}\n{sub}", size=11)
    if i:
        arrow((x - 0.36, 7.05), (x - 0.02, 7.05))

panel(12.1, 5.9, 3.6, 2.8, "Outputs")
box(12.35, 7.35, 3.1, 0.7, "Included papers", fc=BLUE, ec=BLUE, color="white")
box(12.35, 6.55, 3.1, 0.7, "Human review", fc=AQUA_BG, ec=AQUA)
box(12.35, 6.0, 3.1, 0.45, "Stage tables + report", fc=SURFACE, ec=INK2, size=11, bold_first=False)

arrow((3.5, 7.3), (4.0, 7.3))
arrow((11.6, 7.3), (12.1, 7.3))

# Row 2: AI agents
panel(4.0, 3.0, 7.6, 2.4, "AI agents (independent, different models)")
for i, (t, sub) in enumerate((("Reviewer A", "neurologist"), ("Reviewer B", "ML researcher"),
                               ("Reviewer C", "methodologist"))):
    box(4.3 + i * 1.8, 3.25, 1.6, 1.35, f"{t}\n{sub}", fc="white", size=12)
box(9.7, 3.25, 1.7, 1.35, "Re-checks\n2nd & 3rd look\nage 65+", fc="white", size=11)
arrow((7.8, 5.9), (7.8, 5.4))
ax.text(7.95, 5.62, "each paper, judged blind", fontsize=11, color=INK2, va="center")

# Row 3: checks and model services
panel(4.0, 0.3, 3.7, 2.2, "Checks (code)")
box(4.25, 1.0, 3.2, 0.72, "Quote verifier\nexclusions need proof", fc=ORANGE_BG, ec=ORANGE, size=11)
box(4.25, 0.42, 3.2, 0.48, "Consensus rules", fc=ORANGE_BG, ec=ORANGE, size=11)

panel(7.9, 0.3, 3.7, 2.2, "Model services")
box(8.15, 1.0, 3.2, 0.72, "Groq · NVIDIA · OpenRouter\nfree tiers + overflow host", fc=SURFACE, ec=INK2, size=10)
box(8.15, 0.42, 3.2, 0.48, "Response cache", fc=SURFACE, ec=INK2, size=11)

arrow((5.85, 3.0), (5.85, 2.5))        # agents -> checks
arrow((9.75, 3.0), (9.75, 2.5))  # agents -> models
ax.text(9.9, 2.75, "model calls", fontsize=11, color=INK2, va="center")
arrow((4.0, 1.4), (4.0, 6.3), rad=-0.35)    # checks -> pipeline decisions
ax.text(2.55, 3.85, "decisions", fontsize=11, color=INK2, ha="center")

out = Path(__file__).resolve().parents[1] / "architecture-slide.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor=SURFACE)
print(out)
