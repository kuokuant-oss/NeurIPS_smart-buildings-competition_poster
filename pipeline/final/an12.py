"""Numbers for the fifth draft, corrected simulator. argv[1] = scratch/final. Writes an6.json.
Cost-level numbers come from runs_p2; behaviour numbers from trace_p2 (scenarios with both traces)."""
import json, os, re, sys
from pathlib import Path
import numpy as np
S = Path(sys.argv[1]); T = S / "trace_p2"
RUNS = ("runs_p2", "runs_v2") if len(sys.argv) < 3 else tuple(sys.argv[2].split(","))
def rows(n, runs=None):
    d = {}
    for run in (runs or RUNS):
        for t in ("off", "new"):
            f = S / run / f"{t}-{n}.json"
            if f.is_file():
                d.update({f"{t}:{k}": v for k, v in json.loads(f.read_text())["rows"].items() if "minneap" not in k.lower()})
    return d
R, L, G = rows("pr_opt2"), rows("gt_x8"), rows("g36_default")
keys = sorted(set(R) & set(L)); print("scenarios", len(keys), "seen", sum(k.startswith("off:") for k in keys))
st = lambda k: re.sub(r"_\d+$", "", k.split("/")[1]); bd = lambda k: k.split("/")[0].split(":")[1].replace("building_train_", "").replace("building_", "")
P3 = ("comfort_part", "energy_part", "carbon_part")
part = lambda r, p: -r[p] / r["steps"]
GROUPS = [("cold", -99, 5), ("cool", 5, 15), ("mild", 15, 25), ("hot", 25, 99)]
oat = {k: L[k]["oat_mean_c"] for k in keys}
grp = lambda k: next(g for g, a, b in GROUPS if a <= oat[k] < b)
out = {"groups": {}, "stations": {}, "sets": {}, "scene": {}}
def agg(ks, D):
    return [float(np.mean([D[k]["cost"] for k in ks]))] + [float(np.mean([part(D[k], p) for k in ks])) for p in P3]
print("== sets (corrected simulator)")
for label, ks in (("seen", [k for k in keys if k.startswith("off:")]), ("unseen", [k for k in keys if k.startswith("new:")]), ("all", keys)):
    a, b = agg(ks, R), agg(ks, L)
    _d = np.array([L[k]["cost"] - R[k]["cost"] for k in ks])
    if not ks: continue
    out["sets"][label] = {"n": len(ks), "rule": a, "layer": b, "rel": 100 * (b[0] / a[0] - 1), "se": float(100 * _d.std(ddof=1) / np.sqrt(len(ks)) / a[0]),
                          "sem_rule": float(np.std([R[k]["cost"] for k in ks], ddof=1) / np.sqrt(len(ks))), "sem_layer": float(np.std([L[k]["cost"] for k in ks], ddof=1) / np.sqrt(len(ks)))}
    if ks and all(k in G for k in ks):
        g = agg(ks, G); out["sets"][label]["default"] = g
        out["sets"][label]["tuned_vs_default"] = 100 * (a[0] / g[0] - 1); out["sets"][label]["tuned_cheaper"] = int(sum(R[k]["cost"] < G[k]["cost"] for k in ks))
    print(label, len(ks), {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out["sets"][label].items() if k not in ("rule", "layer", "default")},
          "rule", round(a[0], 6), "layer", round(b[0], 6), "default", round(out["sets"][label].get("default", [float("nan")])[0], 6))
    print("    parts change (pts of rule cost): comfort %.2f energy %.2f carbon %.2f" % tuple(100 * (b[i] - a[i]) / a[0] for i in (1, 2, 3)))
rel = np.array([100 * (L[k]["cost"] / R[k]["cost"] - 1) for k in keys])
d = np.array([L[k]["cost"] - R[k]["cost"] for k in keys]); rm = np.mean([R[k]["cost"] for k in keys])
bl = sorted({bd(k) for k in keys}, key=int)
bm = np.array([np.mean([d[i] for i, k in enumerate(keys) if bd(k) == b]) for b in bl])
out["paired"] = {"bld_lower": int((bm < 0).sum()), "se_scen": float(100 * d.std(ddof=1) / np.sqrt(len(keys)) / rm), "se_bld": float(100 * bm.std(ddof=1) / np.sqrt(len(bl)) / rm),
                 "cheaper": int((rel < -0.05).sum()), "dearer": int((rel > 0.05).sum()), "same": int((abs(rel) <= 0.05).sum()),
                 "median": float(np.median(rel)), "q1": float(np.percentile(rel, 25)), "q3": float(np.percentile(rel, 75)),
                 "sd_scen_pts": float(100 * d.std(ddof=1) / rm)}
print("paired", out["paired"])
for tag in ("off", "new"):
    kk = [i for i, k in enumerate(keys) if k.startswith(tag)]
    print("  ", tag, "cheaper", int((rel[kk] < -0.05).sum()), "dearer", int((rel[kk] > 0.05).sum()), "of", len(kk))
print("== temperature groups")
for g, a_, b_ in GROUPS:
    ks = [k for k in keys if grp(k) == g]; a, b = agg(ks, R), agg(ks, L)
    out["groups"][g] = {"n": len(ks), "seen": sum(k.startswith("off:") for k in ks), "unseen": sum(k.startswith("new:") for k in ks), "rule": a, "layer": b,
                        "rel": 100 * (b[0] / a[0] - 1), "comfort_share": 100 * a[1] / a[0], "cheaper": int(sum(L[k]["cost"] < R[k]["cost"] * 0.9995 for k in ks)),
                        "members": sorted({st(k) for k in ks})}
    o = out["groups"][g]
    print(f"{g:5s} n={o['n']} (seen {o['seen']}, unseen {o['unseen']}) rule {1e3*a[0]:.3f} layer {1e3*b[0]:.3f} {o['rel']:+.1f}%  comfort share {o['comfort_share']:.0f}%  cheaper {o['cheaper']}"
          f"  d comfort {100*(b[1]-a[1])/a[0]:+.1f} energy {100*(b[2]-a[2])/a[0]:+.1f} carbon {100*(b[3]-a[3])/a[0]:+.1f}  {o['members']}")
print("== station-months (appendix)")
for s in sorted({st(k) for k in keys}, key=lambda s: np.mean([oat[k] for k in keys if st(k) == s])):
    ks = [k for k in keys if st(k) == s]; a, b = agg(ks, R), agg(ks, L)
    out["stations"][s] = {"seen": ks[0].startswith("off:"), "oat": float(np.mean([oat[k] for k in ks])), "oat_min": min(oat[k] for k in ks), "oat_max": max(oat[k] for k in ks),
                          "rule": a[0], "layer": b[0], "rel": 100 * (b[0] / a[0] - 1), "groups": sorted({grp(k) for k in ks}, key=[g[0] for g in GROUPS].index)}
    print(s, out["stations"][s])
print("== buildings")
for b in bl:
    ks = [k for k in keys if bd(k) == b]; a, c = agg(ks, R), agg(ks, L); print("  B" + b, f"{100*(c[0]/a[0]-1):+.1f}%")
share = np.array([part(R[k], "comfort_part") / R[k]["cost"] for k in keys])
sp = lambda a, b: float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
out["spearman_share_rel"] = sp(share, rel); print("spearman comfort share vs change", round(out["spearman_share_rel"], 2))
# ---------------- traces ----------------
scn = sorted({f.stem.split("__", 1)[1] for f in T.glob("pr_opt2__*.npz")} & {f.stem.split("__", 1)[1] for f in T.glob("gt_x8__*.npz")})
print("== traces: scenarios with both", len(scn))
tr = {}
tot = {m: dict(out=0.0, n=0.0) for m in ("r", "l")}
wf = dict(worst_out_r=0.0, worst_out_l=0.0, worst_n=0.0, rest_out_r=0.0, rest_out_l=0.0, rest_n=0.0)
lor = []
for s in scn:
    b, stn = s.split("__"); key = next(k for k in keys if k.endswith(f"{b}/{stn}"))
    rec = {}
    for m, name in (("r", "pr_opt2"), ("l", "gt_x8")):
        z = np.load(T / f"{name}__{s}.npz"); zone, band = z["zone"], z["band"]
        mode = band[:, 0] > 293.5
        zo, lo, hi = zone[mode], band[mode, 0:1], band[mode, 1:2]
        below, above = zo < lo - 1e-6, zo > hi + 1e-6
        dev = np.clip(lo - zo, 0, None) + np.clip(zo - hi, 0, None)
        rec[m] = dict(out=float((below | above).mean()), below=float(below.mean()), above=float(above.mean()), out1=float((dev > 1).mean()),
                      room=(below | above).mean(0), nstep=int(mode.sum()), cost=float(-z["reward"].mean()), flow=float(z["flow"][mode].sum(1).mean()) if z["flow"].size else float("nan"))
    rr, rl = rec["r"]["room"], rec["l"]["room"]; n = len(rr); k5 = max(1, int(round(n / 5)))
    order = np.argsort(-(rr + rl) / 2); w, o = order[:k5], order[k5:]
    ns = rec["r"]["nstep"]
    wf["worst_out_r"] += rr[w].sum() * ns; wf["worst_out_l"] += rl[w].sum() * ns; wf["worst_n"] += k5 * ns
    wf["rest_out_r"] += rr[o].sum() * ns; wf["rest_out_l"] += rl[o].sum() * ns; wf["rest_n"] += (n - k5) * ns
    c = np.cumsum(rr[order]) / max(rr.sum(), 1e-12); lor.append(np.interp(np.linspace(0, 1, 21), np.arange(1, n + 1) / n, c))
    tr[s] = {"key": key, "grp": grp(key), "oat": oat[key], **{f"{a}_{m}": rec[m][a] for m in rec for a in ("out", "below", "above", "out1", "cost", "flow")}}
if scn:
    A = lambda f: np.array([tr[s][f] for s in scn])
    print(f"check cost from traces: rule {A('cost_r').mean():.6f} layer {A('cost_l').mean():.6f} ({100*(A('cost_l').mean()/A('cost_r').mean()-1):+.2f}%)")
    for f in ("out", "above", "below", "out1"):
        print(f"  share {f}: rule {100*A(f+'_r').mean():.2f}% layer {100*A(f+'_l').mean():.2f}%  lower in {(A(f+'_l') < A(f+'_r') - 1e-9).sum()}/{len(scn)}  higher in {(A(f+'_l') > A(f+'_r') + 1e-9).sum()}")
    print(f"  airflow while occupied: {100*(np.nanmean(A('flow_l'))/np.nanmean(A('flow_r'))-1):+.1f}%")
    out["trace"] = {"n": len(scn), **{f: [float(100 * A(f + "_r").mean()), float(100 * A(f + "_l").mean()), int((A(f + "_l") < A(f + "_r") - 1e-9).sum())] for f in ("out", "above", "below", "out1")},
                    "flow_rel": float(100 * (np.nanmean(A("flow_l")) / np.nanmean(A("flow_r")) - 1))}
    out["trace_groups"] = {}
    for g, _, _ in GROUPS:
        ss = [s for s in scn if tr[s]["grp"] == g]
        if ss:
            o = {f: [float(100 * np.mean([tr[s][f + "_r"] for s in ss])), float(100 * np.mean([tr[s][f + "_l"] for s in ss]))] for f in ("out", "above", "below")}
            o["n"] = len(ss); o["lower"] = int(sum(tr[s]["out_l"] < tr[s]["out_r"] - 1e-9 for s in ss)); out["trace_groups"][g] = o; print("  ", g, o)
    tot_out = wf["worst_out_r"] + wf["rest_out_r"]
    out["worst_fifth"] = {"share_of_outside_time": 100 * wf["worst_out_r"] / tot_out, "worst": [100 * wf["worst_out_r"] / wf["worst_n"], 100 * wf["worst_out_l"] / wf["worst_n"]],
                          "rest": [100 * wf["rest_out_r"] / wf["rest_n"], 100 * wf["rest_out_l"] / wf["rest_n"]],
                          "share_of_reduction": 100 * (wf["worst_out_r"] - wf["worst_out_l"]) / (tot_out - wf["worst_out_l"] - wf["rest_out_l"]),
                          "lorenz": np.mean(lor, 0).tolist()}
    print("worst fifth", {k: (np.round(v, 2).tolist() if k != "lorenz" else "...") for k, v in out["worst_fifth"].items()})
    out["trace_scene"] = tr
# ---------------- other controllers ----------------
out["n_buildings"] = len(bl); out["n"] = len(keys); out["n_seen"] = sum(k.startswith("off:") for k in keys)
out["other"] = {}
se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v)))
for n in ("org_bang_bang", "org_fixed_setpoint", "org_pid", "org_pid_water", "g36_default", "pr_room", "gm_tree_m1", "gt_t2b", "pr_opt2", "gt_x8"):
    D = rows(n)
    if not all(k in D for k in keys):
        print("MISSING", n, sum(k in D for k in keys), "of", len(keys)); continue
    o = {}
    for label, ks in (("seen", [k for k in keys if k.startswith("off:")]), ("unseen", [k for k in keys if k.startswith("new:")]), ("all", keys)):
        v = [D[k]["cost"] for k in ks]; o[label] = [float(np.mean(v)), se(v)]
    dd = np.array([D[k]["cost"] - R[k]["cost"] for k in keys]); rr = np.array([D[k]["cost"] / R[k]["cost"] - 1 for k in keys])
    bmm = np.array([np.mean([dd[i] for i, k in enumerate(keys) if bd(k) == b]) for b in bl])
    o["vs_rule"] = {"rel": float(100 * dd.mean() / rm), "se_scen": float(100 * dd.std(ddof=1) / np.sqrt(len(keys)) / rm), "se_bld": float(100 * bmm.std(ddof=1) / np.sqrt(len(bl)) / rm),
                    "cheaper": int((rr < -0.0005).sum()), "dearer": int((rr > 0.0005).sum())}
    out["other"][n] = o
    print(f"{n:20s} seen {o['seen'][0]:.6f}+-{o['seen'][1]:.6f} unseen {o['unseen'][0]:.6f}+-{o['unseen'][1]:.6f} all {o['all'][0]:.6f}+-{o['all'][1]:.6f} | vs tuned rule {o['vs_rule']}")
if "gm_tree_m1" in out["other"] and "gt_t2b" in out["other"]:
    A_, B_ = rows("gm_tree_m1"), rows("gt_t2b"); d2 = np.array([B_[k]["cost"] - A_[k]["cost"] for k in keys])
    print("t2b vs m1: %+.2f%% (SE %.2f%%)" % (100 * d2.mean() / np.mean([A_[k]["cost"] for k in keys]), 100 * d2.std(ddof=1) / np.sqrt(len(keys)) / np.mean([A_[k]["cost"] for k in keys])))
for tag, name in (("off", "seen"), ("new", "unseen")):
    kk = [i for i, k in enumerate(keys) if k.startswith(tag)]
    if kk:
        dk = d[kk]; bk = sorted({bd(keys[i]) for i in kk}); bmk = np.array([np.mean([d[i] for i in kk if bd(keys[i]) == b]) for b in bk]); rk = np.mean([R[keys[i]]["cost"] for i in kk])
        out["sets"][name]["se_bld"] = float(100 * bmk.std(ddof=1) / np.sqrt(len(bk)) / rk); print(name, "rel", round(out["sets"][name]["rel"], 2), "se scen", round(out["sets"][name]["se"], 2), "se bld", round(out["sets"][name]["se_bld"], 2))
(S / (sys.argv[3] if len(sys.argv) > 3 else "an12.json")).write_text(json.dumps(out, indent=1, default=float))
print("written")
