"""Method overview of the modelling: concept only (no values), but with the real structure of each step.
Three colours: navy, orange (stage 1), blue (stage 2). argv[1] = paper fig folder."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Polygon
OUT = Path(sys.argv[1])
plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
NAVY, S1C, S2C, GREY, MID = "#33415c", "#1f6fb2", "#1f6fb2", "#ececec", "#9aa0a6"
ONC = "#1f6fb2"
TINT = ["#e3e6ea", "#c3c9d1", "#98a1ad"]
fig = plt.figure(figsize=(6.0, 2.7))
X0 = [0.012, 0.235, 0.445, 0.69]; W = [0.185, 0.17, 0.205, 0.3]
def panel(k, title):
    ax = fig.add_axes([X0[k], 0.02, W[k], 0.83]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.text(X0[k], 0.945, f"{k + 1}", fontsize=8, fontweight="bold", color="white", ha="left", va="center",
             bbox=dict(boxstyle="circle,pad=0.22", facecolor=NAVY, edgecolor="none"))
    fig.text(X0[k] + 0.03, 0.945, title, fontsize=6.7, fontweight="bold", ha="left", va="center")
    return ax
ROOMS = [(0.0, 0.5, 0.28, 0.5, 0), (0.28, 0.5, 0.28, 0.5, 0), (0.56, 0.5, 0.44, 0.5, 2), (0.0, 0.0, 0.4, 0.5, 2), (0.4, 0.0, 0.3, 0.5, 1), (0.7, 0.0, 0.3, 0.5, 1)]
def building(ax, x, y, w, h, on=None):
    for rx, ry, rw, rh, c in ROOMS:
        ax.add_patch(Rectangle((x + rx * w, y + ry * h), rw * w, rh * h, facecolor=ONC if on == c else TINT[c], edgecolor="white", linewidth=0.8))
def node(ax, x, y, text, w, hh=0.085, fs=5.2, fc="white", ec="black", tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - hh / 2), w, hh, boxstyle="round,pad=0.006,rounding_size=0.02", facecolor=fc, edgecolor=ec, linewidth=0.6))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal")
def link(ax, x0, y0, x1, y1): ax.plot([x0, x1], [y0, y1], color="black", linewidth=0.55, zorder=0)

# ---------------- 1 experiment (version chosen by the author)
ax = panel(0, "Experiment")
building(ax, 0.12, 0.68, 0.76, 0.28)
ax.text(0.5, 0.645, "rooms grouped into classes", ha="center", va="top", fontsize=5.4)
for i, c in enumerate((0, 1, 2)):
    x = 0.03 + i * 0.33
    building(ax, x, 0.29, 0.28, 0.14, on=c)
    ax.add_patch(Rectangle((x + 0.02, 0.2), [0.12, 0.2, 0.24][i], 0.045, facecolor=MID, edgecolor="none"))
ax.text(0.5, 0.47, "switch a setting on\nfor one class", ha="center", va="bottom", fontsize=5.4)
ax.text(0.5, 0.16, "measure the cost\nof the whole building", ha="center", va="top", fontsize=5.4)

# ---------------- 2 stage 1: decision for every room class and option; some depend on the weather
ax = panel(1, "Stage 1: by class")
PAT = ["F.H..", "FH...", "F.H..", "FH.F.", "...F.", "..HF."]      # F on, H on in part of the climate range, . off
gx, gy, cw, chh = 0.24, 0.3, 0.14, 0.095
for r, row in enumerate(PAT):
    for c, v in enumerate(row):
        x, y = gx + c * cw, gy + (len(PAT) - 1 - r) * chh
        ax.add_patch(Rectangle((x, y), cw, chh, facecolor=S1C if v == "F" else GREY, edgecolor="white", linewidth=0.9))
        if v == "H":
            ax.add_patch(Polygon([(x + 0.006, y + 0.006), (x + cw - 0.006, y + 0.006), (x + 0.006, y + chh - 0.006)], closed=True, facecolor=S1C, edgecolor="none"))
ax.text(gx + 2.5 * cw, gy + 6 * chh + 0.02, "settings", ha="center", va="bottom", fontsize=5.4)
ax.text(gx - 0.04, gy + 3 * chh, "room classes", ha="right", va="center", fontsize=5.4, rotation=90)
ax.add_patch(Rectangle((0.06, 0.17), 0.08, 0.055, facecolor=S1C)); ax.text(0.17, 0.197, "on", ha="left", va="center", fontsize=5.3)
ax.add_patch(Rectangle((0.36, 0.17), 0.08, 0.055, facecolor=GREY)); ax.add_patch(Polygon([(0.36, 0.17), (0.44, 0.17), (0.36, 0.225)], closed=True, facecolor=S1C, edgecolor="none"))
ax.text(0.47, 0.197, "on in part of\nthe climate", ha="left", va="center", fontsize=4.9)
ax.text(0.5, 0.06, "mean effect over\nmany buildings", ha="center", va="center", fontsize=5.2)

# ---------------- 3 stage 2: honest tree, three levels, generic notation
ax = panel(2, "Stage 2: decision tree")
node(ax, 0.5, 0.93, r"$x \leq t_1$", 0.3, fs=6.2)
xs = [0.115, 0.37, 0.63, 0.885]; ON = [False, True, True, True]
for side, xm in enumerate((0.25, 0.75)):
    link(ax, 0.5, 0.888, xm, 0.775); node(ax, xm, 0.735, r"$x \leq t_%d$" % (side + 2), 0.3, fs=6.2)
    for s2 in range(2):
        k = 2 * side + s2
        link(ax, xm, 0.692, xs[k], 0.575)
        node(ax, xs[k], 0.53, "ON" if ON[k] else "off", 0.2, hh=0.09, fs=5.6, fc=S2C if ON[k] else GREY, ec=S2C if ON[k] else MID, tc="white" if ON[k] else "#555555", bold=True)
ax.text(0.5, 0.41, "one tree per setting", ha="center", va="center", fontsize=5.3, style="italic")
for i in range(5):
    ax.add_patch(Rectangle((0.05 + i * 0.058, 0.23), 0.042, 0.07, facecolor=MID, edgecolor="none"))
    ax.add_patch(Rectangle((0.63 + i * 0.058, 0.23), 0.042, 0.07, facecolor=NAVY, edgecolor="none"))
ax.text(0.19, 0.2, "half the buildings\nchoose the splits", ha="center", va="top", fontsize=4.8)
ax.text(0.79, 0.2, "the other half\nsets the leaves", ha="center", va="top", fontsize=4.8)

# ---------------- 4 merged tree, generic notation: features x1, x2, x3; leaves are option sets a1, a2, ...
ax = panel(3, "Merged tree")
def chips(x, y, cols):
    w = 0.034
    for i, c in enumerate(cols):
        ax.add_patch(Rectangle((x - len(cols) * w / 2 + i * w, y - 0.03), w - 0.006, 0.06, facecolor=ONC if c else GREY, edgecolor="white", linewidth=0.4))
def dots(x, y): ax.text(x, y, "\u22ee", ha="center", va="center", fontsize=8, color="#777777")
node(ax, 0.5, 0.93, r"$x_1$", 0.16, fs=6.4)
for x in (0.14, 0.5, 0.86):
    link(ax, 0.5, 0.888, x, 0.795); node(ax, x, 0.755, r"$x_2$", 0.16, fs=6.4)
for x in (0.14, 0.86):
    for dx in (-0.08, 0.08):
        link(ax, x, 0.712, x + dx, 0.635); dots(x + dx, 0.595)
link(ax, 0.5, 0.712, 0.38, 0.62); node(ax, 0.38, 0.58, r"$x_3$", 0.16, fs=6.4)
link(ax, 0.5, 0.712, 0.62, 0.635); dots(0.62, 0.595)
for k, (xk, cols) in enumerate(((0.2, [1, 0, 1, 1]), (0.38, [0, 1, 0, 0]), (0.56, [1, 1, 0, 1]))):
    link(ax, 0.38, 0.537, xk, 0.44); chips(xk, 0.395, cols)
    ax.text(xk, 0.325, r"$a_%d$" % (k + 1), ha="center", va="center", fontsize=6.0)
ax.text(0.5, 0.235, r"$x$: features,  $a$: settings of a room", ha="center", va="center", fontsize=5.3)
ax.add_patch(Rectangle((0.24, 0.115), 0.045, 0.055, facecolor=ONC)); ax.text(0.305, 0.142, "on", ha="left", va="center", fontsize=5.3)
ax.add_patch(Rectangle((0.52, 0.115), 0.045, 0.055, facecolor=GREY)); ax.text(0.585, 0.142, "off", ha="left", va="center", fontsize=5.3)
for i in range(3):
    fig.add_artist(FancyArrowPatch((X0[i] + W[i] + 0.004, 0.5), (X0[i + 1] - 0.006, 0.5), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8, linewidth=1.0, color=NAVY))
fig.savefig(OUT / "model_diagram.pdf"); fig.savefig(OUT / "model_diagram.png", dpi=230); plt.close(fig)
print("written")
