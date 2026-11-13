"""Overview of the tuned rule (baseline). Its own logic, not the tree's: a control sequence with settings,
settings scheduled over the climate, a one-setting-at-a-time search checked on held-out buildings, and one
configuration for every room. Concept only, generic notation; blue = kept / active, grey = rejected / neutral.
argv[1] = paper fig folder."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Ellipse
OUT = Path(sys.argv[1])
plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
NAVY, ONC, GREY, MID = "#33415c", "#1f6fb2", "#ececec", "#9aa0a6"
TINT = ["#e3e6ea", "#c3c9d1", "#98a1ad"]
fig = plt.figure(figsize=(6.0, 2.7))
X0 = [0.012, 0.25, 0.462, 0.765]; W = [0.2, 0.18, 0.265, 0.225]
def panel(k, title):
    ax = fig.add_axes([X0[k], 0.02, W[k], 0.83]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.text(X0[k], 0.945, f"{k + 1}", fontsize=8, fontweight="bold", color="white", ha="left", va="center",
             bbox=dict(boxstyle="circle,pad=0.22", facecolor=NAVY, edgecolor="none"))
    fig.text(X0[k] + 0.03, 0.945, title, fontsize=6.7, fontweight="bold", ha="left", va="center")
    return ax
ROOMS = [(0.0, 0.5, 0.28, 0.5, 0), (0.28, 0.5, 0.28, 0.5, 0), (0.56, 0.5, 0.44, 0.5, 2), (0.0, 0.0, 0.4, 0.5, 2), (0.4, 0.0, 0.3, 0.5, 1), (0.7, 0.0, 0.3, 0.5, 1)]
def slider(ax, x, y, w, pos, col=ONC, lw=1.0, ms=3.2):
    ax.plot([x, x + w], [y, y], color=MID, linewidth=lw, solid_capstyle="round")
    ax.plot(x + pos * w, y, "o", color=col, markersize=ms)
def arrow(ax, p0, p1, col=NAVY, lw=0.8, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=6, linewidth=lw, color=col, connectionstyle=f"arc3,rad={rad}"))

# ---------------- 1 the rule: a control sequence that runs every step; each block has settings
ax = panel(0, "Rule")
ax.add_patch(FancyBboxPatch((0.27, 0.78), 0.46, 0.14, boxstyle="round,pad=0.006,rounding_size=0.02", facecolor=TINT[0], edgecolor=MID, linewidth=0.6))
ax.text(0.5, 0.85, "building", ha="center", va="center", fontsize=5.8)
bx = [0.14, 0.38, 0.62, 0.86]; POS = [0.3, 0.7, 0.45, 0.6]
for i, x in enumerate(bx):
    ax.add_patch(FancyBboxPatch((x - 0.09, 0.44), 0.18, 0.13, boxstyle="round,pad=0.006,rounding_size=0.02", facecolor="white", edgecolor="black", linewidth=0.6))
    ax.text(x, 0.505, r"$f_%d$" % (i + 1), ha="center", va="center", fontsize=6.2)
    slider(ax, x - 0.075, 0.36, 0.15, POS[i]); ax.text(x, 0.29, r"$\theta_%d$" % (i + 1), ha="center", va="center", fontsize=5.6)
    if i < 3: arrow(ax, (x + 0.095, 0.505), (bx[i + 1] - 0.095, 0.505))
arrow(ax, (0.26, 0.85), (0.14, 0.585), rad=0.35); arrow(ax, (0.86, 0.585), (0.74, 0.85), rad=0.35)
ax.text(0.21, 0.66, "state", ha="left", va="center", fontsize=5.4, color="#555555"); ax.text(0.79, 0.66, "commands", ha="right", va="center", fontsize=5.4, color="#555555")
ax.text(0.5, 0.15, "control sequence $f_1 \\ldots f_n$\nwith settings $\\theta$", ha="center", va="center", fontsize=5.5)

# ---------------- 2 schedule: every setting has its own value in cold, mild and hot weather
ax = panel(1, "Schedule")
cols = [0.36, 0.6, 0.84]; POSS = [(0.25, 0.7, 0.3), (0.75, 0.35, 0.75), (0.3, 0.3, 0.8)]
for x, t in zip(cols, ("cold", "mild", "hot")):
    ax.add_patch(FancyBboxPatch((x - 0.1, 0.33), 0.2, 0.56, boxstyle="round,pad=0.004,rounding_size=0.02", facecolor="#f4f5f7", edgecolor="none"))
    ax.text(x, 0.845, t, ha="center", va="center", fontsize=5.6)
for r, poss in enumerate(POSS):
    y = 0.72 - r * 0.15
    ax.text(0.2, y, r"$\theta_%d$" % (r + 1), ha="right", va="center", fontsize=6.0)
    for x, pos in zip(cols, poss):
        slider(ax, x - 0.075, y, 0.15, pos)
ax.text(0.5, 0.2, "each setting has one value\nper temperature band", ha="center", va="center", fontsize=5.4)

# ---------------- 3 search: change one setting, test it on the search set, then confirm it on the held-out set
ax = panel(2, "Search")
ax.text(0.46, 0.94, "change one setting", ha="center", va="center", fontsize=5.5)
for r, (pos, new) in enumerate(((0.3, None), (0.35, 0.7), (0.6, None))):
    y = 0.875 - r * 0.05
    ax.plot([0.25, 0.75], [y, y], color=MID, linewidth=1.0, solid_capstyle="round")
    if new is None:
        ax.plot(0.25 + pos * 0.5, y, "o", color=NAVY, markersize=3.0)
    else:
        ax.plot(0.25 + pos * 0.5, y, "o", markerfacecolor="white", markeredgecolor=MID, markersize=3.0)
        ax.plot(0.25 + new * 0.5, y, "o", color=ONC, markersize=3.4)
        arrow(ax, (0.25 + pos * 0.5 + 0.03, y), (0.25 + new * 0.5 - 0.03, y), col=ONC, lw=0.9)
def stage(y, title, question, short):
    ax.add_patch(FancyBboxPatch((0.04, y - 0.06), 0.36, 0.12, boxstyle="round,pad=0.006,rounding_size=0.02", facecolor="white", edgecolor="black", linewidth=0.6))
    ax.text(0.22, y, title, ha="center", va="center", fontsize=5.1)
    ax.add_patch(Rectangle((0.5, y + 0.022), 0.24, 0.026, facecolor=MID, edgecolor="none"))
    ax.add_patch(Rectangle((0.5, y - 0.012), 0.24 * short, 0.026, facecolor=ONC, edgecolor="none"))
    ax.text(0.62, y - 0.062, question, ha="center", va="center", fontsize=5.1)
arrow(ax, (0.22, 0.735), (0.22, 0.675), lw=0.7); ax.text(0.28, 0.705, "1. test", ha="left", va="center", fontsize=5.0, color="#555555")
stage(0.6, "search set", "cheaper?", 0.72)
arrow(ax, (0.22, 0.535), (0.22, 0.445), lw=0.7); ax.text(0.28, 0.49, "if yes: 2. confirm", ha="left", va="center", fontsize=5.0, color="#555555")
stage(0.375, "held-out set", "not worse?", 0.9)
arrow(ax, (0.22, 0.31), (0.22, 0.225), lw=0.7); ax.text(0.28, 0.268, "if yes", ha="left", va="center", fontsize=5.0, color="#555555")
for x, t, on in ((0.22, "keep", True), (0.7, "discard", False)):
    ax.add_patch(FancyBboxPatch((x - 0.13, 0.135), 0.26, 0.08, boxstyle="round,pad=0.006,rounding_size=0.02", facecolor=ONC if on else GREY, edgecolor=ONC if on else MID, linewidth=0.6))
    ax.text(x, 0.175, t, ha="center", va="center", fontsize=5.5, color="white" if on else "#555555", fontweight="bold")
ax.plot([0.84, 0.84], [0.6, 0.27], color=MID, linewidth=0.7, linestyle=(0, (2, 1.5)))
ax.plot([0.78, 0.84], [0.6, 0.6], color=MID, linewidth=0.7, linestyle=(0, (2, 1.5))); ax.plot([0.78, 0.84], [0.375, 0.375], color=MID, linewidth=0.7, linestyle=(0, (2, 1.5)))
arrow(ax, (0.84, 0.275), (0.76, 0.22), col=MID, lw=0.7); ax.text(0.93, 0.49, "if no", ha="center", va="center", fontsize=5.0, color="#555555", rotation=90)
ax.text(0.46, 0.045, "then the next setting", ha="center", va="center", fontsize=5.2, style="italic")

# ---------------- 4 tuned rule: one configuration, the same for every room
ax = panel(3, "Tuned rule")
ax.add_patch(FancyBboxPatch((0.25, 0.66), 0.5, 0.26, boxstyle="round,pad=0.006,rounding_size=0.02", facecolor="white", edgecolor="black", linewidth=0.6))
for j_, pos in enumerate((0.3, 0.7, 0.5)):
    slider(ax, 0.33, 0.86 - j_ * 0.065, 0.34, pos, lw=1.0, ms=3.4)
ax.text(0.5, 0.61, r"one configuration $\theta^{*}$", ha="center", va="center", fontsize=5.6)
xs4 = [0.14, 0.38, 0.62, 0.86]
for x in xs4:
    arrow(ax, (0.5, 0.565), (x, 0.44), lw=0.6)
    ax.add_patch(FancyBboxPatch((x - 0.095, 0.25), 0.19, 0.18, boxstyle="round,pad=0.004,rounding_size=0.02", facecolor="#f4f5f7", edgecolor=MID, linewidth=0.6))
    for j_, pos in enumerate((0.3, 0.7, 0.5)):
        slider(ax, x - 0.065, 0.385 - j_ * 0.045, 0.13, pos, lw=0.7, ms=2.2)
ax.text(0.5, 0.17, "the same in every room", ha="center", va="center", fontsize=5.5)

for i in range(3):
    fig.add_artist(FancyArrowPatch((X0[i] + W[i] + 0.004, 0.5), (X0[i + 1] - 0.006, 0.5), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8, linewidth=1.0, color=NAVY))
fig.savefig(OUT / "baseline_diagram.pdf"); fig.savefig(OUT / "baseline_diagram.png", dpi=230); plt.close(fig)
print("written")
