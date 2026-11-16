"""Correction T1-1 (2026-11-01), from diagnosis-T1-1.json and the two checks beside this file:
- gt_t1a equals gm_tree_m1 there, so one global choice per factor is exhausted.
- The gap to the per-scenario oracle is mostly noise: 12 runs for 11 coefficients leave 1 residual degree of freedom per scenario.

The correction chooses each factor per climate. Per factor, take the best of:
- off;
- on;
- on below one threshold of the scenario's mean outside air;
- on above one threshold;

chosen on all 80 scenarios. The chosen side must still be below zero by one standard error, else the factor stays off. The thresholds cut the scenarios into climate bins, giving one cell map per bin. When deployed, the controller picks the bin from the running mean outside air since the episode start (b2rl.py "oat_mode": "episode"), which is the same quantity as the training label."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_trait_check as tc


def build(ctx):
    gs, per, factors, feats = ctx.gs, ctx.per, ctx.factors, ctx.feats
    specs = sorted(per)
    tr = {s: gs.scenario_traits(s, feats, ctx.dirs) for s in specs}
    rules = []
    for i, f in enumerate(factors):
        _, name, fn = tc.best_rule(per, i, tr, specs, ("oat_mean",))
        on = [s for s in specs if fn(s) and per[s]["total"][i] is not None]
        vals = [per[s]["total"][i] for s in on]
        ok = len(vals) >= 4 and np.mean(vals) + np.std(vals, ddof=1) / np.sqrt(len(vals)) < 0
        if name == "off" or not ok:
            rules.append(("off", None, None))
        elif name == "on":
            rules.append(("on", None, None))
        else:
            op, q = ("<=", float(name.split("<=")[1])) if "<=" in name else (">", float(name.split(">")[1]))
            rules.append((op, q, None))
    cuts = sorted({q for op, q, _ in rules if q is not None})

    def levels_at(oat):
        lv = []
        for op, q, _ in rules:
            lv.append(1 if op == "on" or (op == "<=" and oat <= q) or (op == ">" and oat > q) else -1)
        return lv

    reps = [cuts[0] - 1.0] + [(a + b) / 2 for a, b in zip(cuts, cuts[1:])] + [cuts[-1] + 1.0] if cuts else [0.0]
    maps = [gs.compose(ctx.base_map, factors, levels_at(x), 1) for x in reps]
    cell_map = {"oat_cuts": cuts, "maps": maps} if cuts else maps[0]
    levels_by_spec = {s: levels_at(tr[s]["oat_mean"]) for s in specs}
    predicted = float(np.mean([gs.predicted(per, levels_by_spec[s], [s]) for s in specs]))
    described = ", ".join(f"{f['name']} {op}{'' if q is None else f' {q:.1f} C'}" for f, (op, q, _) in zip(factors, rules))
    ctx.log("EVENT", f"T1 correction 1 rules (scenario mean outside air): {described}; climate bins at {cuts}; "
                     f"predicted {100 * predicted:+.2f}% on the design scenarios")
    (ctx.stage.state.get("notes") or ctx.stage.state.setdefault("notes", {}))["T1-1"] = described
    return {"label": "b", "map": cell_map, "levels_by_spec": levels_by_spec, "predicted": predicted}


if __name__ == "__main__":
    print(json.dumps({"module": __file__}))
