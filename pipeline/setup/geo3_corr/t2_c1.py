"""Correction T2-1 (2026-11-02), from diagnosis-T2-1.json and t2_check.py.

What diagnosis 1 showed:
- INNER, which missed by 0.13 points, is where Denver and Fairbanks weigh most: they are the costliest scenarios.

t2_check.py shows the factors that work in the cool climates:

The correction applies the symmetric-evidence climate rule of correction T1-2 to the T2 factors: a climate exception needs the opposite sign by one standard error. Its bins are laid over the T1c base's climate bins, and every merged bin gets the base sub-map of its climate plus the T2 levels of its climate."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_c2


def build(ctx):
    gs, per, factors = ctx.gs, ctx.per, ctx.factors
    tr = {s: gs.scenario_traits(s, ctx.feats, ctx.dirs) for s in per}
    oat = {s: tr[s]["oat_mean"] for s in per}
    rules = t1_c2.symmetric_rules(gs, per, factors, oat)
    base = ctx.base_map
    base_cuts = base["oat_cuts"] if gs.is_climate(base) else []
    new_cuts = [r["exception"][1] for r in rules if r["exception"]]
    cuts = sorted(set(base_cuts) | set(new_cuts))
    reps = ([cuts[0] - 1.0] + [(a + b) / 2 for a, b in zip(cuts, cuts[1:])] + [cuts[-1] + 1.0]) if cuts else [0.0]
    maps = []
    for x in reps:
        sub = base["maps"][sum(x > c for c in base_cuts)] if gs.is_climate(base) else base
        maps.append(gs.compose(sub, factors, t1_c2.levels_at(rules, x), 1))
    cell_map = {"oat_cuts": cuts, "maps": maps} if cuts else maps[0]
    levels_by_spec = {s: t1_c2.levels_at(rules, oat[s]) for s in per}
    predicted = float(np.mean([gs.predicted(per, levels_by_spec[s], [s]) for s in per]))
    described = "; ".join(f"{r['factor']} {'on' if r['global'] > 0 else 'off'}"
                          + (f" except {r['exception'][0]} {r['exception'][1]:.1f} C (mean {100 * r['exception'][2]:+.2f}%, "
                             f"t {r['exception'][3]:+.1f})" if r["exception"] else "") for r in rules)
    ctx.log("EVENT", f"T2 correction 1 symmetric rules: {described}; climate bins at {[round(c, 2) for c in cuts]}; "
                     f"predicted {100 * predicted:+.2f}% vs the T1c base on the design scenarios")
    ctx.stage.state.setdefault("notes", {})["T2-1"] = described
    return {"label": "b", "map": cell_map, "levels_by_spec": levels_by_spec, "predicted": predicted}
