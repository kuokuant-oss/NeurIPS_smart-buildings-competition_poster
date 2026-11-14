"""Writes numbers.tex (macros used by main.tex) from an6.json. argv[1] = scratch/final, argv[2] = paper folder, argv[3] = optional file with the day text."""
import json, sys, io
from pathlib import Path
S, P = Path(sys.argv[1]), Path(sys.argv[2])
A = json.loads((S / "an6.json").read_text())
L = []
from math import floor, log10
def se2(v): return f"{v:.{1 - int(floor(log10(v)))}f}"
def m(name, val): L.append("\\newcommand{\\%s}{%s}" % (name, val))
s = A["sets"]["all"]
if "default" in s:
    m("DefCost", f"{s['default'][0]:.6f}"); m("DefRel", f"{-s['tuned_vs_default']:.1f}"); m("DefCheaper", str(s["tuned_cheaper"]))
else:
    m("DefCost", "PENDING"); m("DefRel", "PENDING"); m("DefCheaper", "PENDING")
t = A.get("trace")
if t:
    m("OutR", f"{t['out'][0]:.1f}"); m("OutL", f"{t['out'][1]:.1f}"); m("OutLower", f"{t['out'][2]} of the {t['n']}")
    m("AboveR", f"{t['above'][0]:.1f}"); m("AboveL", f"{t['above'][1]:.1f}"); m("BelowR", f"{t['below'][0]:.1f}"); m("BelowL", f"{t['below'][1]:.1f}")
    m("BigR", f"{t['out1'][0]:.1f}"); m("BigL", f"{t['out1'][1]:.1f}")
    w = A["worst_fifth"]
    m("WfShare", f"{w['share_of_outside_time']:.0f}"); m("WfR", f"{w['worst'][0]:.1f}"); m("WfL", f"{w['worst'][1]:.1f}"); m("RestR", f"{w['rest'][0]:.1f}"); m("RestL", f"{w['rest'][1]:.1f}")
pp = A["paired"]
m("ValN", str(A["n"])); m("ValNSeen", str(A["n_seen"])); m("ValNUnseen", str(A["n"] - A["n_seen"])); m("ValBld", str(A["n_buildings"]))
m("ValRel", f"{-A['sets']['all']['rel']:.1f}"); m("ValCheaper", str(pp["cheaper"])); m("ValDearer", str(pp["dearer"]))
m("PairSE", f"{pp['se_scen']:.1f}"); m("PairSEB", f"{pp['se_bld']:.1f}")
if "pr_room" in A.get("other", {}): m("RoomRel", f"{-A['other']['pr_room']['vs_rule']['rel']:.1f}")
O = A.get("other", {})
m("ValBldLower", str(pp["bld_lower"]))
def cell(n, k, bold=False):
    if n not in O: return "(pending)"
    v, e = O[n][k]; t = "%.6f" % v
    return ("\\textbf{%s}" % t if bold else t) + " $\\pm$ " + se2(e)
def ccell(n):
    if n not in O: return "(pending)"
    t = "%.6f" % O[n]["all"][0]
    return "\\textbf{%s}" % t if best["all"] == n else t
def chg(n):
    if n not in ("pr_room", "gt_x8") or n not in O: return "--"
    v = O[n]["vs_rule"]; return "$%+.1f \\pm %.1f\\%%$" % (v["rel"], v["se_bld"])
HR = [("org_bang_bang", "Bang-bang", "--"), ("org_fixed_setpoint", "Fixed setpoint", "--"), ("org_pid", "PID", "--"), ("org_pid_water", "PID + water reset", "--"), None,
      ("pr_opt2", "Tuned rule", "0.01071"), ("pr_room", "Tuned rule + per-room selection", "--"), ("gt_x8", "Rule + layer", "\\textbf{0.01056}")]
best = {k: min((O[r[0]][k][0], r[0]) for r in HR if r and r[0] in O)[1] for k in ("seen", "unseen", "all")}
L.append("\\newcommand{\\HeadRows}{%\n" + "\n".join("    \\midrule" if r is None else "    %s & %s & %s & %s \\\\" % (r[1], ccell(r[0]), chg(r[0]), r[2]) for r in HR) + "}")
def cost(n): return ("%.6f" % O[n]["all"][0]) if n in O else "(pending)"
def worse(n): return ("%d of %d" % (O[n]["vs_rule"]["dearer"], A["n"])) if n in O else "(pending)"
BR = [("org_bang_bang", "Bang-bang", "--"), ("org_fixed_setpoint", "Fixed setpoint", "--"), ("org_pid", "PID", "--"), ("org_pid_water", "PID + water reset", "--"), None, ("g36_default", "Guideline rule, default settings", "0.01441")]
L.append("\\newcommand{\\BaseRows}{%\n" + "\n".join("    \\midrule" if r is None else "    %s & %s & %s & %s \\\\" % (r[1], cost(r[0]), worse(r[0]), r[2]) for r in BR)
         + "\n    Tuned rule (baseline) & \\textbf{%s} & -- & \\textbf{0.01071} \\\\}" % cost("pr_opt2"))
ia, ib = cost("gm_tree_m1"), cost("gt_t2b")
if "(" not in ia + ib: ia, ib = (ia, "\\textbf{%s}" % ib) if float(ib) < float(ia) else ("\\textbf{%s}" % ia, ib)
L.append("\\newcommand{\\IntfRows}{%\n    Cost of individual rooms & " + ia + " & 0.01124 \\\\\n    Cost of the whole building & " + ib + " & \\textbf{0.01113} \\\\}")
NICE = {"atlanta": "Atlanta", "denver": "Denver", "fairbanks": "Fairbanks", "phoenix": "Phoenix", "singapore": "Singapore", "beijing_jan": "Beijing, January",
        "beijing_jul": "Beijing, July", "dubai_jan": "Dubai, January", "dubai_jul": "Dubai, July", "london_jan": "London, January", "london_jul": "London, July",
        "riyadh_jan": "Riyadh, January", "riyadh_jul": "Riyadh, July", "seattle_jan": "Seattle, January", "seattle_jul": "Seattle, July"}
rows = []
for k, v in A["stations"].items():
    oat = f"{v['oat']:.1f}".replace("-", "$-$") if not v["seen"] else (f"{v['oat_min']:.0f} to {v['oat_max']:.0f}").replace("-", "$-$")
    rows.append(f"    {NICE[k]} & {'seen' if v['seen'] else 'unseen'} & {oat} & {', '.join(v['groups'])} & {v['rule']:.6f} & {v['layer']:.6f} \\\\")
tab = ("\\begin{tabular}{llcccc}\n    \\toprule\n    \\textbf{Station and month} & \\textbf{Climate} & \\textbf{Outdoor ($^\\circ$C)} & \\textbf{Groups} & \\textbf{Tuned rule} & \\textbf{Rule + layer} \\\\\n    \\midrule\n"
       + "\n".join(rows) + "\n    \\bottomrule\n  \\end{tabular}")
L.append("\\newcommand{\\ClimateTable}{%\n" + tab + "}")
for k, nm in (("seen", "Seen"), ("unseen", "Unseen"), ("all", "All")):
    v = A["sets"][k]; m("Chg" + nm, ("$%+.1f \\pm %.1f\\%%$" % (v["rel"], v["se"])).replace("$-", "$-"))
for k, nm in (("seen", "Seen"), ("unseen", "Unseen"), ("all", "All")):
    v = A["sets"][k]
    m("Rule" + nm, "%.6f $\\pm$ %s" % (v["rule"][0], se2(v["sem_rule"]))); m("Layer" + nm, "\\textbf{%.6f} $\\pm$ %s" % (v["layer"][0], se2(v["sem_layer"])))
for k, nm in (("seen", "Seen"), ("unseen", "Unseen"), ("all", "All")):
    v = A["sets"][k]
    pass
GN = {"cold": "Cold (below 5~$^\\circ$C)", "cool": "Cool (5--15~$^\\circ$C)", "mild": "Mild (15--25~$^\\circ$C)", "hot": "Hot (above 25~$^\\circ$C)"}
gr = []
for g in ("cold", "cool", "mild", "hot"):
    v = A["groups"][g]; a, b = v["rule"], v["layer"]
    dd = [("$%+.1f$" % (100 * (b[i] - a[i]) / a[0])).replace("+0.0", "0.0").replace("-0.0", "0.0") for i in (1, 2, 3)]
    best = (lambda x, y: ("\\textbf{%.6f}" % x) if x < y else "%.6f" % x)
    gr.append("    %s & %d & %s & %s & %d & %s & %s & %s \\\\" % (GN[g], v["n"], best(a[0], b[0]), best(b[0], a[0]), v["cheaper"], dd[0], dd[1], dd[2]))
L.append("\\newcommand{\\GroupRows}{%\n" + "\n".join(gr) + "}")
day = Path(sys.argv[3]).read_text(encoding="utf-8").strip() if len(sys.argv) > 3 else "PENDING"
L.append("\\newcommand{\\DayText}{%\n" + day + "}")
io.open(P / "numbers.tex", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n"); print("\n".join(L)[:1800])
