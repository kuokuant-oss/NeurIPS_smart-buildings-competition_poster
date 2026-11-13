"""Result figures of the fifth draft (corrected simulator). argv[1] = scratch/final, argv[2] = paper fig folder. Needs an6.json and trace_p2."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
S, OUT = Path(sys.argv[1]), Path(sys.argv[2]); T = S / "trace_p2"
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.unicode_minus": True})
A = json.loads((S / "an6.json").read_text())
RULE, LAYER, WARN = "#c4621c", "#1f6fb2", "#d98a3d"
PART = {"comfort": ("#c6d4ea", "#4c72b0"), "energy": ("#f3d2b8", "#dd8452"), "carbon": ("#dcdcdc", "#7f7f7f")}
GL = {"cold": "cold\n(below 5 °C)", "cool": "cool\n(5–15 °C)", "mild": "mild\n(15–25 °C)", "hot": "hot\n(above 25 °C)"}
pct = lambda v, d=1: f"{v:+.{d}f}%".replace("-", "−")
def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=220); plt.close(fig)

# ---------------- Figure: share of scenarios that are cheaper ----------------
p = A["paired"]; G = A["groups"]; order = ["cold", "cool", "mild", "hot"]
fig, (ax, axl) = plt.subplots(1, 2, figsize=(3.6, 1.25), gridspec_kw={"width_ratios": [1.0, 1.9]})
vals = [p["cheaper"], p["dearer"], p["same"]]; cols = ["#1a1a1a", "#8c8c8c", "#d9d9d9"]
labels = [f"cheaper with the learned layer: {vals[0]}", f"more expensive: {vals[1]}", f"within ±0.05%: {vals[2]}"]
w, _ = ax.pie(vals, colors=cols, startangle=90, counterclock=False, wedgeprops=dict(edgecolor="white", linewidth=0.8))
ax.set_aspect("equal")
axl.axis("off")
axl.legend(w, labels, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.5), fontsize=7.5, handlelength=1.0, handleheight=1.0, labelspacing=0.7, borderaxespad=0)
fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.05); save(fig, "scenario_share")

# ---------------- Figure: one working day in two unseen climates ----------------
sc = A["trace_scene"]
def pick_building(stations):
    best = None
    for b in ("1", "2", "4", "5", "7", "8"):
        ss = [f"building_train_{b}__{s}" for s in stations]
        if not all(s in sc for s in ss): continue
        err = sum(abs(100 * (sc[s]["cost_l"] / sc[s]["cost_r"] - 1) - A["stations"][st]["rel"]) for s, st in zip(ss, stations))
        if best is None or err < best[0]: best = (err, b)
    return best[1]
EXS = ["riyadh_jul", "beijing_jan"]
b = "4"   # the building of the day example in the text
fig, axes = plt.subplots(2, 2, figsize=(5.5, 2.5), gridspec_kw={"height_ratios": [1.25, 1.0]}, sharex=True)
for e, stn in enumerate(EXS):
    s = f"building_train_{b}__{stn}"; D = {}
    for m, name in (("r", "pr_opt2"), ("l", "gt_x8")):
        z = np.load(T / f"{name}__{s}.npz"); j = json.loads((T / f"{name}__{s}.json").read_text()); wts = j["weights"]; sw = sum(wts)
        D[m] = dict(zone=z["zone"] - 273.15, band=z["band"] - 273.15, comf=-wts[0] * z["regret"] / sw, en=wts[1] * z["energy"] / sw, ca=wts[2] * z["carbon"] / sw, oat=j["oat_daily"])
    n = len(D["r"]["zone"]); off = 215                                     # episodes start at 06:05; step 215 is 00:00 of the next day
    days = [(k, slice(off + k * 288, off + (k + 1) * 288)) for k in range((n - off) // 288)]
    work = [(k, sl) for k, sl in days if (D["r"]["band"][sl, 0] > 20.5).sum() > 100]
    tot = lambda m, sl: (D[m]["comf"][sl] + D[m]["en"][sl] + D[m]["ca"][sl]).mean()
    target = sc[s]["cost_l"] / sc[s]["cost_r"]
    k, sl = min(work, key=lambda t: abs(tot("l", t[1]) / tot("r", t[1]) - target))
    hrs = np.arange(288) / 12.0
    ax = axes[0, e]
    occ = D["r"]["band"][sl, 0] > 20.5; h_on = hrs[occ]
    ax.add_patch(Rectangle((h_on.min(), 20.85), h_on.max() - h_on.min(), 3.0, facecolor="#e6f0de", edgecolor="#86b06c", linewidth=0.5, zorder=0))
    for m, col in (("r", RULE), ("l", LAYER)):
        zt = D[m]["zone"][sl]
        ax.fill_between(hrs, np.percentile(zt, 5, axis=1), np.percentile(zt, 95, axis=1), color=col, alpha=0.22, linewidth=0, zorder=2)
        ax.plot(hrs, np.median(zt, axis=1), color=col, linewidth=1.2, zorder=3 if m == "l" else 4, linestyle="-" if m == "l" else (0, (4, 1.5)))
    dr = 100 * (tot("l", sl) / tot("r", sl) - 1)
    name = {"riyadh_jul": "Riyadh, July", "beijing_jan": "Beijing, January"}[stn]
    ax.set_title(f"({'ab'[e]}) {name}: cost of the day {pct(dr)}", fontsize=7.5, loc="left")
    if e == 0: ax.set_ylabel("Room temp. ($^\\circ$C)", fontsize=6.8)
    ax2 = axes[1, e]
    for m, col in (("r", RULE), ("l", LAYER)):
        c = (D[m]["comf"][sl] + D[m]["en"][sl] + D[m]["ca"][sl]) * 1e3
        ax2.plot(hrs, c, color=col, linewidth=1.0, linestyle="-" if m == "l" else (0, (4, 1.5)), zorder=3 if m == "l" else 4)
        ax2.fill_between(hrs, 0, D[m]["comf"][sl] * 1e3, color=col, alpha=0.3, linewidth=0)
    ax2.set_xlim(0, 24); ax2.set_xticks([0, 6, 12, 18, 24]); ax2.set_xlabel("Hour of day"); ax2.set_ylim(0, None)
    if e == 0: ax2.set_ylabel("Cost per step\n($\\times10^{-3}$)", fontsize=6.8)
    print(f"day example {s}: day index {k}, cost rule {tot('r', sl):.6f} layer {tot('l', sl):.6f} ({dr:+.1f}%); scenario change {100*(target-1):+.1f}%;"
          f" comfort r/l {D['r']['comf'][sl].mean()*1e3:.3f}/{D['l']['comf'][sl].mean()*1e3:.3f}, energy {D['r']['en'][sl].mean()*1e3:.3f}/{D['l']['en'][sl].mean()*1e3:.3f},"
          f" carbon {D['r']['ca'][sl].mean()*1e3:.3f}/{D['l']['ca'][sl].mean()*1e3:.3f}; outdoor daily mean {D['l']['oat'][min(k + 1, len(D['l']['oat']) - 1)]}")
hh = [Line2D([], [], color=RULE, linewidth=1.3, linestyle=(0, (4, 1.5))), (Rectangle((0, 0), 1, 1, facecolor=RULE, alpha=0.3), Rectangle((0, 0), 1, 1, facecolor=LAYER, alpha=0.3)), Line2D([], [], color=LAYER, linewidth=1.3),
      Rectangle((0, 0), 1, 1, facecolor="#e6f0de", edgecolor="#86b06c", linewidth=0.5)]
fig.legend(hh, ["tuned rule", "5th–95th percentile of rooms (top); comfort term (bottom)", "rule + layer", "comfort band while occupied"], ncol=2, frameon=False,
           loc="upper center", bbox_to_anchor=(0.5, 1.005), fontsize=6.4, handlelength=1.6, columnspacing=1.6, labelspacing=0.3, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)})
fig.subplots_adjust(left=0.115, right=0.99, top=0.80, bottom=0.14, wspace=0.16, hspace=0.22); save(fig, "day_example")

# ---------------- Figure: time outside the comfort band ----------------
TG, W5 = A["trace_groups"], A["worst_fifth"]
LEG = [Rectangle((0, 0), 1, 1, facecolor=RULE), Rectangle((0, 0), 1, 1, facecolor=LAYER), Rectangle((0, 0), 1, 1, facecolor="#777777"),
       Rectangle((0, 0), 1, 1, facecolor="#777777", alpha=0.45, hatch="////")]
fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.65), gridspec_kw={"width_ratios": [2.2, 1.25]})
ax = axes[0]; og = [g for g in order if g in TG]; x = np.arange(len(og)); bw = 0.36
for i, col in enumerate((RULE, LAYER)):
    ab = np.array([TG[g]["above"][i] for g in og]); be = np.array([TG[g]["below"][i] for g in og])
    ax.bar(x + (i - 0.5) * bw, ab, bw, color=col, edgecolor="white", linewidth=0.3)
    ax.bar(x + (i - 0.5) * bw, be, bw, bottom=ab, color=col, alpha=0.45, edgecolor="white", linewidth=0.3, hatch="////")
    for xi, t in zip(x, ab + be): ax.text(xi + (i - 0.5) * bw, t + 0.12, f"{t:.1f}", ha="center", va="bottom", fontsize=6.2)
ax.set_xticks(x); ax.set_xticklabels([GL[g] for g in og], fontsize=6.6); ax.set_ylabel("Occupied room-hours\noutside the band (%)", fontsize=7)
ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
ax.set_title("(a) by mean outdoor temperature of the scenario", fontsize=7.5, loc="left")
ax.legend(LEG[2:], ["above the band", "below the band"], ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.42, 1.03), fontsize=6.6, handlelength=1.0, columnspacing=1.2)
ax = axes[1]; x = np.arange(2)
for i, col in enumerate((RULE, LAYER)):
    v = [W5["worst"][i], W5["rest"][i]]
    ax.bar(x + (i - 0.5) * bw, v, bw, color=col, edgecolor="white", linewidth=0.3)
    for xi, t in zip(x, v): ax.text(xi + (i - 0.5) * bw, t + 0.25, f"{t:.1f}", ha="center", va="bottom", fontsize=6.2)
ax.set_xticks(x); ax.set_xticklabels(["fifth of rooms with\nthe most time\noutside the band", "other rooms"], fontsize=6.6)
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
ax.set_title("(b) by room", fontsize=7.5, loc="left")
ax.legend(LEG[:2], ["tuned rule", "rule + layer"], ncol=1, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.02), fontsize=6.6, handlelength=1.0, labelspacing=0.25)
ax.set_ylabel("Outside the band (%)", fontsize=7)
fig.subplots_adjust(left=0.115, right=0.995, top=0.88, bottom=0.27, wspace=0.27); save(fig, "comfort_rooms")
print("figures written")
