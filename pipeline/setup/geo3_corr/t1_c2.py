"""Correction T1-2 (2026-11-01), from diagnosis-T1-2.json and the per-scenario closed-loop comparison.

What diagnosis 2 showed:
- The totals model stays calibrated on the design scenarios: predicted −4.24%, realised −4.38%, slope 1.00.
- gt_t1b beats gm_tree_m1 there by −0.33 points. That is −2.59 in Fairbanks (small exterior rooms off in the cold) and +1.20 in Atlanta.
- By cell group the loss is medium exterior rooms (+0.14), and the closed loop lost +1.0 to +1.7 points in Atlanta and Singapore.

The cause is a flaw in correction 1. Its threshold rules were chosen by the training mean, with the one-standard-error guard only on the side that turns a factor ON. So min_m_ext was switched off in warm months on a mean of +0.16 ± 0.37: noise, with tails from −2.8 to +4.9. By station it is negative everywhere.

Fix 1, symmetric evidence. Start from the global one-standard-error choice. A climate exception (a side of one threshold of the scenario's mean outside air, at least 10 scenarios) is taken only when that side's mean has the opposite sign to the global choice by more than one standard error. The most significant threshold wins.

Fix 2, the wall-heat form. gm_tree_m1's remaining edge (inner Fairbanks/Denver) sits in long-wall large rooms, where it uses a MILD wall-length heating, (wall / 30 m)^0.5. The T1 design tested only the strong form, (wall / 20 m)^1: costly in the cold, slightly saving in the warm. So three variants are recorded on the 80 design scenarios:
- A: the symmetric rules alone;
- B: A plus the mild form for large long-wall rooms in every climate bin where A sets no wall heat;
- C: the mild form instead of any strong form.

The variant with the lowest measured mean cost goes to the closed loop."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

W30 = {"zone_heat_wall_ref_m": 30.0, "zone_heat_wall_power": 0.5}
LARGE_LONG = ("l|long|0", "x|long|0")
MIN_SIDE = 10


def symmetric_rules(gs, per, factors, oat):
    glob = gs.choose(gs.summarise(per, factors), factors)
    rules = []
    for i, f in enumerate(factors):
        specs = [s for s in per if per[s]["total"][i] is not None]
        xs = np.array([oat[s] for s in specs])
        best = None
        for q in np.unique(np.percentile(xs, np.arange(15, 90, 7.5))):
            for side in ("<=", ">"):
                sel = [s for s in specs if (oat[s] <= q) == (side == "<=")]
                if len(sel) < MIN_SIDE:
                    continue
                v = np.array([per[s]["total"][i] for s in sel])
                m, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
                t = (m / se) if se > 0 else 0.0
                flips = (glob[i] > 0 and m - se > 0) or (glob[i] < 0 and m + se < 0)
                if flips and (best is None or abs(t) > abs(best[3])):
                    best = (side, float(q), m, t)
        rules.append({"factor": f["name"], "global": glob[i], "exception": best})
    return rules


def levels_at(rules, x):
    lv = []
    for r in rules:
        e = r["exception"]
        inside = e is not None and ((x <= e[1]) if e[0] == "<=" else (x > e[1]))
        lv.append(-r["global"] if inside else r["global"])
    return lv


def build(ctx):
    gs, per, factors = ctx.gs, ctx.per, ctx.factors
    tr = {s: gs.scenario_traits(s, ctx.feats, ctx.dirs) for s in per}
    oat = {s: tr[s]["oat_mean"] for s in per}
    rules = symmetric_rules(gs, per, factors, oat)
    cuts = sorted({r["exception"][1] for r in rules if r["exception"]})
    reps = ([cuts[0] - 1.0] + [(a + b) / 2 for a, b in zip(cuts, cuts[1:])] + [cuts[-1] + 1.0]) if cuts else [0.0]
    base_maps = [gs.compose(ctx.base_map, factors, levels_at(rules, x), 1) for x in reps]

    def with_mild(maps, replace_strong):
        out = []
        for m in maps:
            m = {k: dict(v) for k, v in m.items()}
            for k in LARGE_LONG:
                if replace_strong or not m[k].get("zone_heat_wall_ref_m"):
                    m[k].update(W30)
            out.append(m)
        return out

    def as_map(maps):
        return {"oat_cuts": cuts, "maps": maps} if cuts else maps[0]

    variants = {"A": as_map(base_maps), "B": as_map(with_mild(base_maps, False)), "C": as_map(with_mild(base_maps, True))}
    described = "; ".join(f"{r['factor']} {'on' if r['global'] > 0 else 'off'}"
                          + (f" except {r['exception'][0]} {r['exception'][1]:.1f} C (mean {100 * r['exception'][2]:+.2f}%, "
                             f"t {r['exception'][3]:+.1f})" if r["exception"] else "") for r in rules)
    ctx.log("EVENT", f"T1 correction 2 symmetric rules: {described}; climate bins at {[round(c, 2) for c in cuts]}")
    models = {k: ctx.model_for(f"gt_t1c_{k}", v, None, ctx.base_model) for k, v in variants.items()}
    if getattr(ctx, "dry_run", False):
        return {"variants": variants, "rules": rules}
    ctx.record(list(models.values()))
    m1 = {s: ctx.rs.find_record([ctx.gs.DIAG], "gm_tree_m1", s) for s in per}
    base = {s: ctx.rs.find_record(ctx.dirs, "gm_fixed0", s) for s in per}
    res = {}
    for k in variants:
        rel, vs_m1 = [], []
        for s in per:
            r = ctx.rs.find_record(ctx.dirs, f"gt_t1c_{k}", s)
            if r is None:
                continue
            rel.append(r["summary"]["cost"] / base[s]["summary"]["cost"] - 1.0)
            if m1[s] is not None:
                vs_m1.append(r["summary"]["cost"] / m1[s]["summary"]["cost"] - 1.0)
        res[k] = {"vs base %": 100 * float(np.mean(rel)), "vs gm_tree_m1 %": 100 * float(np.mean(vs_m1)),
                  "se vs m1": 100 * float(np.std(vs_m1, ddof=1) / np.sqrt(len(vs_m1))), "n": len(rel)}
    pick = min(res, key=lambda k: res[k]["vs base %"])
    ctx.log("EVENT", "T1 correction 2 variants on the design scenarios: " + "; ".join(
        f"{k} {v['vs base %']:+.2f}% vs night1, {v['vs gm_tree_m1 %']:+.2f}±{v['se vs m1']:.2f} vs gm_tree_m1"
        for k, v in res.items()) + f"; picked {pick}")
    ctx.stage.state.setdefault("notes", {})["T1-2"] = {"rules": described, "variants": res, "picked": pick}
    levels_by_spec = {s: levels_at(rules, oat[s]) for s in per} if pick == "A" else None
    return {"label": "c", "map": variants[pick], "levels_by_spec": levels_by_spec,
            "predicted": res[pick]["vs base %"] / 100}
