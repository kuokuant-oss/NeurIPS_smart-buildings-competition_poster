"""Pipeline 4 machinery (2026-11-02, docs/plan/geo-room-stages-4.md): per-room option factors (U1) and period-of-day
factors (U2) learned from building totals of balanced designs, a shallow honest tree on the per-scenario factor
effects (three variants and the main-effects rule compared by leave-buildings-out on the real buildings), and the
policy a decision turns into. Builds on group_stage.py (12 cells, Plackett-Burman design, climate maps).

A factor = one option (a dict of per-room switches) switched on for a group of cells, in all climate bins of the base
map [and only in some periods of the day]. A factor's per-scenario effect (share of the base-map cost when it goes
off -> on) comes from that scenario's 8- or 12-run design. Tree "units" are (scenario, factor) pairs with the features
area (the group's mean area band 0..3), wall (mean wall class 0..2) and oat (the scenario's mean outside air, which
is what the policy's episode running mean converges to); one tree per option type.

Used by pipeline/setup/geo_pipeline4.py."""
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_stage as gs  # noqa: E402
import room_stage as rs  # noqa: E402

OUT4 = ROOT / "data/runs/b2rl/geo4/stage4"
SM, LX, ALL = ("s", "m"), ("l", "x"), gs.AREAS
EXT, NONLONG = ("perim", "long"), ("core", "perim")
V62 = {"zone_min_ra_lps_m2": 0.3, "zone_min_rp_lps_m2": 0.125, "zone_min_oa_divide": True}
AREA_IDX, WALL_IDX = {a: i for i, a in enumerate(gs.AREAS)}, {w: i for i, w in enumerate(gs.WALLS)}
GRID = np.arange(-25.0, 45.0001, 0.25)

PB8 = [[1, 1, 1, -1, 1, -1, -1]]
for _ in range(6):
    PB8.append(PB8[-1][-1:] + PB8[-1][:-1])
PB8.append([-1] * 7)                                      # all off = the base map


def cells(areas, walls):
    return [[a, w] for a in areas for w in walls]


def F(name, kind, areas, walls, delta, periods=None):
    return {"name": name, "type": kind, "cells": cells(areas, walls), "delta": delta,
            "periods": periods,
            "area": float(np.mean([AREA_IDX[a] for a in areas])), "wall": float(np.mean([WALL_IDX[w] for w in walls]))}


def margin(kind, k):
    return {f"room_{kind}_margin_k": k}


# U1: options beyond pipeline 3's on the gt_t2b map (levels: setpoint margin +0.8 / +0.4 / -0.4, heating airflow maximum
# 0.3 / 0.6, heating discharge 11 K, the code minimum airflow where gt_t2b leaves it off)
U1 = [F("cool08_SMl", "cool+0.8", SM, ["long"], margin("cool", 0.8)),
      F("cool08_LXl", "cool+0.8", LX, ["long"], margin("cool", 0.8)),
      F("cool04_SMn", "cool+0.4", SM, NONLONG, margin("cool", 0.4)),
      F("cool04_LXn", "cool+0.4", LX, NONLONG, margin("cool", 0.4)),
      F("heat08_SMe", "heat+0.8", SM, EXT, margin("heat", 0.8)),
      F("heat08_LXe", "heat+0.8", LX, EXT, margin("heat", 0.8)),
      F("heatrelax04_core", "heat-0.4", ALL, ["core"], margin("heat", -0.4)),
      F("ratio03_SM", "ratio0.3", SM, gs.WALLS, {"room_heat_max_ratio": 0.3}),
      F("ratio06_LXe", "ratio0.6", LX, EXT, {"room_heat_max_ratio": 0.6}),
      F("disch11_SM", "disch11", SM, gs.WALLS, {"room_discharge_offset_k": 11.0}),
      F("min_SMe", "min", SM, EXT, V62)]

# U2: option x period (1 warm-up, 2 in between, 3 late)
U2 = [F("W_heat08_SMe", "W_heat+0.8", SM, EXT, margin("heat", 0.8), [1]),
      F("W_heat08_LXe", "W_heat+0.8", LX, EXT, margin("heat", 0.8), [1]),
      F("W_ratio06_ext", "W_ratio0.6", ALL, EXT, {"room_heat_max_ratio": 0.6}, [1]),
      F("W_cool08_all", "W_cool+0.8", ALL, gs.WALLS, margin("cool", 0.8), [1]),
      F("M_cool04_SM", "M_cool+0.4", SM, gs.WALLS, margin("cool", 0.4), [2]),
      F("M_cool04_LX", "M_cool+0.4", LX, gs.WALLS, margin("cool", 0.4), [2]),
      F("L_cool_relax04_SM", "L_relax", SM, gs.WALLS, margin("cool", -0.4), [3]),
      F("L_heat_relax04_SM", "L_relax", SM, gs.WALLS, margin("heat", -0.4), [3]),
      F("L_ratio03_all", "L_ratio0.3", ALL, gs.WALLS, {"room_heat_max_ratio": 0.3}, [3]),
      F("L_relax08_LX", "L_relax08", LX, gs.WALLS, {**margin("cool", -0.8), **margin("heat", -0.8)}, [3]),
      F("M_heat08_ext", "M_heat+0.8", ALL, EXT, margin("heat", 0.8), [2])]

# U2c: warm-up only, the larger levels (+-0.8 K, airflow 0.6 / 0.3), 7 factors (a 12-run design; factors that set the same
# switch never share cells and periods, so that no two columns conflict)
U2C = [F("W_heat08_SMe", "W_heat+0.8", SM, EXT, margin("heat", 0.8), [1]),
       F("W_heat08_LXe", "W_heat+0.8", LX, EXT, margin("heat", 0.8), [1]),
       F("W_ratio06_ext", "W_ratio0.6", ALL, EXT, {"room_heat_max_ratio": 0.6}, [1]),
       F("W_cool08_SM", "W_cool+0.8", SM, gs.WALLS, margin("cool", 0.8), [1]),
       F("W_cool08_LX", "W_cool+0.8", LX, gs.WALLS, margin("cool", 0.8), [1]),
       F("W_heatrelax08_core", "W_relax08", ALL, ["core"], margin("heat", -0.8), [1]),
       F("W_ratio03_core", "W_ratio0.3", ALL, ["core"], {"room_heat_max_ratio": 0.3}, [1])]


# ---------------------------------------------------------------- scenario sets
def scenario_sets():
    sets = json.loads((ROOT / "configs/b2rl/eval-sets.json").read_text(encoding="utf-8"))
    fresh = json.loads((ROOT / "data/runs/b2rl/geo2/pipeline.json").read_text(encoding="utf-8"))["fresh80"]
    real = list(sets[rs.SEARCH])
    first = {}
    for s in fresh[::2] + fresh[1::2]:                    # one scenario per generated building (prefer the half that
        first.setdefault(s.split("/")[1], s)              # pipeline 3 recorded), the rest are the later replicates
    gen1 = list(first.values())
    gen2 = [s for s in fresh if s not in gen1]
    return real, gen1, gen2


def conflicts(factors):
    """Pairs of factors that set the same switch on a common cell in a common period (their columns would conflict)."""
    out = []
    for i, a in enumerate(factors):
        for j in range(i + 1, len(factors)):
            b = factors[j]
            if set(a["delta"]) & set(b["delta"]) and any(c in b["cells"] for c in a["cells"]) and (
                    a["periods"] is None or b["periods"] is None or set(a["periods"]) & set(b["periods"])):
                out.append((a["name"], b["name"]))
    return out


# ---------------------------------------------------------------- designs and policies
def _key(cell, b):
    return f"{cell[0]}|{cell[1]}|{b}"


def pb_rows(k):
    """Always the 12-run design: the 8-run one is saturated for 7 factors, its main effects are aliased with two-factor
    interactions (seen in the second smoke)."""
    if k <= 11:
        return gs.PB12
    raise ValueError(k)


def compose_one(base, factors, on, nper):
    out = {}
    for cell in gs.CELLS:
        for b in range(nper):
            opt = dict(base.get(_key(cell, b), base.get(_key(cell, 0), {})))
            for fac, o in zip(factors, on):
                if o and list(cell) in fac["cells"] and (fac.get("periods") is None or b in fac["periods"]):
                    opt.update(fac["delta"])
            out[_key(cell, b)] = opt
    return out


def compose_design(base_map, factors, levels, nper):
    on = [v > 0 for v in levels]
    if gs.is_climate(base_map):
        return {"oat_cuts": list(base_map["oat_cuts"]),
                "maps": [compose_one(m, factors, on, nper) for m in base_map["maps"]]}
    return compose_one(base_map, factors, on, nper)


def design_runs(prefix, base_map, factors, nper):
    k = len(factors)
    return [(f"{prefix}{r:02d}", row[:k], compose_design(base_map, factors, row[:k], nper))
            for r, row in enumerate(pb_rows(k))]


def bin_reps(cuts):
    reps = []
    for j in range(len(cuts) + 1):
        if j == 0:
            reps.append(cuts[0] - 1.0 if cuts else 10.0)
        elif j == len(cuts):
            reps.append(cuts[-1] + 1.0)
        else:
            reps.append((cuts[j - 1] + cuts[j]) / 2.0)
    return reps


def compose_decided(base_map, factors, dec, cuts_new, nper):
    """A cell map in which factor i is on exactly where dec(i, oat) is true: the base map's climate bins refined by the
    decision's own cuts (a plain map when there are none)."""
    base = base_map if gs.is_climate(base_map) else {"oat_cuts": [], "maps": [base_map]}
    cuts = sorted({round(float(c), 6) for c in base["oat_cuts"]} | {round(float(c), 6) for c in cuts_new})
    maps = []
    for r in bin_reps(cuts):
        j0 = sum(r > c for c in base["oat_cuts"])
        maps.append(compose_one(base["maps"][j0], factors, [bool(dec(i, r)) for i in range(len(factors))], nper))
    return {"oat_cuts": cuts, "maps": maps} if cuts else maps[0]


def _cell_tree4(cell_map, nper, options):
    def period_node(cell):
        leaves = [{"leaf": options.index(cell_map[_key(cell, b)])} for b in range(nper)]
        node = leaves[-1]
        for b in range(nper - 2, -1, -1):
            node = {"feature": "period", "threshold": b + 0.5, "left": leaves[b], "right": node}
        return node

    def wall_node(a):
        return {"feature": "wall", "threshold": 0.1, "left": period_node((a, "core")),
                "right": {"feature": "wall", "threshold": 20.0, "left": period_node((a, "perim")),
                          "right": period_node((a, "long"))}}

    tree = wall_node(gs.AREAS[-1])
    for i in range(len(gs.AREA_EDGES) - 1, -1, -1):
        tree = {"feature": "area", "threshold": gs.AREA_EDGES[i], "left": wall_node(gs.AREAS[i]), "right": tree}
    return tree


def policy_of(cell_map, nper=1, periods=None):
    """Room-policy tree over area, wall [and period] for a (climate) cell map; "periods" is the period_of config."""
    maps = cell_map["maps"] if gs.is_climate(cell_map) else [cell_map]
    options = [{}]
    for m in maps:
        for k in sorted(m):
            if m[k] not in options:
                options.append(m[k])
    trees = [_cell_tree4(m, nper, options) for m in maps]
    extra = {"periods": dict(periods)} if nper > 1 else {}
    if not gs.is_climate(cell_map):
        return {"mode": "tree", "tree": trees[0], "options": options, **extra}
    cuts = cell_map["oat_cuts"]
    tree = trees[-1]
    for i in range(len(cuts) - 1, -1, -1):
        tree = {"feature": "oat", "threshold": float(cuts[i]), "left": trees[i], "right": tree}
    return {"mode": "tree", "tree": tree, "options": options, "oat_mode": "episode", **extra}


# ---------------------------------------------------------------- effects
def estimate(specs, feats, factors, runs, dirs, base_names):
    """Per scenario: {"eff": per-factor main effect (None when no room of the scenario is in the factor's cells),
    "oat": the scenario's mean outside air}. base_names: record names that may stand for the all-off row."""
    per = {}
    for spec in specs:
        recs = []
        for name, levels, _ in runs:
            rec = None
            if all(v < 0 for v in levels):
                for bn in base_names:
                    rec = rs.find_record(dirs, bn, spec)
                    if rec is not None:
                        break
            if rec is None:
                rec = rs.find_record(dirs, name, spec)
            if rec is None:
                break
            recs.append((levels, rec))
        if len(recs) < len(runs):
            continue
        base = [r for lv, r in recs if all(v < 0 for v in lv)][0]
        norm = base["summary"]["cost"]
        X = np.array([[1.0] + list(lv) for lv, _ in recs])
        y = np.array([r["summary"]["cost"] / norm - 1.0 for _, r in recs])
        pres = gs.present(factors, gs.room_cells(feats, spec))
        cols = [0] + [i + 1 for i, ok in enumerate(pres) if ok]
        beta, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
        eff = [None] * len(factors)
        for j, c in enumerate(cols[1:], start=1):
            eff[c - 1] = 2.0 * float(beta[j])
        per[spec] = {"eff": eff, "oat": float(np.mean(base["hourly"]["outside_t"]))}
    return per


def embed(per, n, index_of):
    """Effects of a design on a factor subset -> vectors over the full list of n factors (index_of: subset -> full)."""
    out = {}
    for s, v in per.items():
        eff = [None] * n
        for j, e in enumerate(v["eff"]):
            eff[index_of[j]] = e
        out[s] = {"eff": eff, "oat": v["oat"]}
    return out


def summarise(per, factors):
    rows = {}
    for i, fac in enumerate(factors):
        vals = [v["eff"][i] for v in per.values() if v["eff"][i] is not None]
        rows[fac["name"]] = ({"mean": float(np.mean(vals)), "se": float(np.std(vals, ddof=1) / math.sqrt(len(vals))),
                              "n": len(vals)} if len(vals) >= 2 else {"mean": 0.0, "se": float("inf"), "n": len(vals)})
    return rows


def screen(per, factors, active, folds=3):
    """Stability screen (U1b): a factor stays when its mean effect is negative in at least 2 of 3 disjoint thirds of the
    buildings."""
    buildings = sorted({s.split("/")[1] for s in per})
    rng = np.random.default_rng(1)
    fold = {b: i % folds for i, b in enumerate(rng.permutation(buildings))}
    keep, table = [], {}
    for i in active:
        neg = []
        for f in range(folds):
            vals = [v["eff"][i] for s, v in per.items() if fold[s.split("/")[1]] == f and v["eff"][i] is not None]
            neg.append(bool(vals) and float(np.mean(vals)) < 0.0)
        table[factors[i]["name"]] = neg
        if sum(neg) >= 2:
            keep.append(i)
    return keep, table


# ---------------------------------------------------------------- tree units
class Units:
    """(scenario, factor) units of the active factors: S scenario id, B building id, I factor, Y effect, X features."""

    def __init__(self, per, factors, active, specs=None):
        self.factors, self.active = factors, list(active)
        self.scen = [s for s in (specs or sorted(per)) if s in per]
        sid = {s: k for k, s in enumerate(self.scen)}
        bld = sorted({s.split("/")[1] for s in self.scen})
        bid = {b: k for k, b in enumerate(bld)}
        self.types = sorted({factors[i]["type"] for i in self.active})
        rows = []
        for s in self.scen:
            for i in self.active:
                e = per[s]["eff"][i]
                if e is not None:
                    rows.append((sid[s], bid[s.split("/")[1]], i, e, factors[i]["area"], factors[i]["wall"], per[s]["oat"]))
        arr = np.array(rows, dtype=float).reshape(-1, 7)
        self.S, self.B, self.I = arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2].astype(int)
        self.Y, self.X = arr[:, 3], arr[:, 4:7]
        self.T = np.array([self.types.index(factors[i]["type"]) for i in self.I], dtype=int)
        self.sid, self.bid = sid, bid

    def scen_mask(self, scenarios):
        return np.isin(self.S, [self.sid[s] for s in scenarios if s in self.sid])


def _leaf_stats(U, idx, min_leaf):
    if len(idx) == 0:
        return {"n": 0, "mean": 0.0, "se": float("inf"), "on": False}
    uniq, inv = np.unique(U.S[idx], return_inverse=True)
    m = np.bincount(inv, U.Y[idx]) / np.bincount(inv)
    n = len(uniq)
    se = float(np.std(m, ddof=1) / math.sqrt(n)) if n > 1 else float("inf")
    mean = float(m.mean())
    return {"n": int(n), "mean": mean, "se": se, "on": bool(n >= min_leaf and mean + se < 0.0)}


def _best_split(U, idx, min_n, allowed):
    y, s = U.Y[idx], U.S[idx]
    base_sse = float(((y - y.mean()) ** 2).sum())
    best = None
    for j in allowed:
        x = U.X[idx, j]
        order = np.argsort(x, kind="stable")
        xs, ys, ss = x[order], y[order], s[order]
        n = len(ys)
        if n < 2:
            continue
        cs, cs2 = np.cumsum(ys), np.cumsum(ys * ys)
        pl, pr = np.empty(n, dtype=int), np.empty(n, dtype=int)
        seen = set()
        for k in range(n):
            seen.add(ss[k])
            pl[k] = len(seen)
        seen = set()
        for k in range(n - 1, -1, -1):
            seen.add(ss[k])
            pr[k] = len(seen)
        k = np.arange(n - 1)
        nl = k + 1.0
        nr = n - nl
        sse = (cs2[k] - cs[k] ** 2 / nl) + ((cs2[-1] - cs2[k]) - (cs[-1] - cs[k]) ** 2 / nr)
        gain = base_sse - sse
        gain[~((xs[k] < xs[k + 1]) & (pl[k] >= min_n) & (pr[k + 1] >= min_n))] = -np.inf
        if not np.isfinite(gain).any():
            continue
        b = int(np.argmax(gain))
        if gain[b] > 1e-12 and (best is None or gain[b] > best[0]):
            best = (float(gain[b]), j, float((xs[b] + xs[b + 1]) / 2.0))
    return best


def _grow(U, idx, levels, min_n):
    """levels: allowed feature indices per depth level (its length is the depth)."""
    if not levels or len(idx) < 2:
        return {"leaf": True}
    split = _best_split(U, idx, min_n, levels[0])
    if split is None:
        return {"leaf": True}
    _, j, t = split
    m = U.X[idx, j] <= t
    return {"f": j, "t": t, "l": _grow(U, idx[m], levels[1:], min_n), "r": _grow(U, idx[~m], levels[1:], min_n)}


def _fill(node, U, idx, min_leaf):
    if "f" not in node:
        node["stat"] = _leaf_stats(U, idx, min_leaf)
        return
    m = U.X[idx, node["f"]] <= node["t"]
    _fill(node["l"], U, idx[m], min_leaf)
    _fill(node["r"], U, idx[~m], min_leaf)


def _walk(node, X, key):
    """Leaf statistic `key` ("on" / "mean") of every row of X."""
    if "f" not in node:
        return np.full(len(X), float(node["stat"][key]) if key == "mean" else node["stat"][key])
    out = np.zeros(len(X), dtype=float)
    m = X[:, node["f"]] <= node["t"]
    if m.any():
        out[m] = _walk(node["l"], X[m], key)
    if (~m).any():
        out[~m] = _walk(node["r"], X[~m], key)
    return out


DEFAULT_LEVELS = [[0, 1, 2], [0, 1, 2]]
OAT_FIRST = [[2], [0, 1]]          # U1c: the first split on the episode running-mean outside air, then the cells


def honest_tree(U, idx, min_leaf, levels, seed):
    """Structure from a random half of the buildings, leaf means from the other half."""
    b = np.unique(U.B[idx])
    rng = np.random.default_rng(seed)
    half = set(rng.permutation(b)[: max(1, len(b) // 2)].tolist())
    in_a = np.array([x in half for x in U.B[idx]])
    node = _grow(U, idx[in_a], levels, max(3, min_leaf // 2))
    _fill(node, U, idx[~in_a], min_leaf)
    return node


class Fit:
    """A chosen policy as a decision per (factor, outside air): kind me / tree / bag / reg."""

    def __init__(self, kind, U, models, info=None):
        self.kind, self.U, self.models, self.info = kind, U, models, info or {}

    def on(self, X, I):
        """Whether each unit (feature row X, factor index I) is switched on."""
        X, I = np.asarray(X, dtype=float).reshape(-1, 3), np.asarray(I, dtype=int)
        if self.kind == "me":
            return np.array([bool(self.models.get(int(i), False)) for i in I])
        out = np.zeros(len(X), dtype=bool)
        types = np.array([self.U.types.index(self.U.factors[int(i)]["type"]) for i in I])
        for t in np.unique(types):
            m = types == t
            model = self.models.get(int(t))
            if model is None:
                continue
            if self.kind == "bag":
                votes = np.mean([_walk(n, X[m], "on") for n in model], axis=0)
                out[m] = votes >= 0.5
            else:
                out[m] = _walk(model, X[m], "on") > 0.5
        return out


def fit_kind(kind, U, scenarios, levels=None, min_leaf=6, bags=50, seed=0):
    levels = levels or DEFAULT_LEVELS
    mask = U.scen_mask(scenarios)
    if kind == "me":
        models = {}
        for i in U.active:
            vals = []
            for s in np.unique(U.S[mask & (U.I == i)]):
                vals.append(float(U.Y[mask & (U.I == i) & (U.S == s)].mean()))
            models[i] = len(vals) >= 2 and float(np.mean(vals)) + float(np.std(vals, ddof=1) / math.sqrt(len(vals))) < 0.0
        return Fit("me", U, models)
    models, info = {}, {}
    for t in range(len(U.types)):
        idx = np.where(mask & (U.T == t))[0]
        if len(idx) < 4:
            continue
        if kind == "tree":
            models[t] = honest_tree(U, idx, min_leaf, levels, seed + t)
        elif kind == "bag":
            b = np.unique(U.B[idx])
            rng = np.random.default_rng(seed + 100 + t)
            trees = []
            for r in range(bags):
                draw = rng.choice(b, size=len(b), replace=True)
                sub = idx[np.isin(U.B[idx], np.unique(draw))]
                trees.append(honest_tree(U, sub, min_leaf, levels, seed + 1000 * (r + 1) + t))
            models[t] = trees
        elif kind == "reg":
            b = np.unique(U.B[idx])
            fold = {x: k % 3 for k, x in enumerate(np.random.default_rng(seed + 7).permutation(b))}
            fold_of = np.array([fold[x] for x in U.B[idx]])
            score = {}
            for ml in (6, 10, 16, 24):
                tot = []
                for f in range(3):
                    node = honest_tree(U, idx[fold_of != f], ml, levels, seed + 31 * f + t)
                    test = idx[fold_of == f]
                    on = _walk(node, U.X[test], "on") > 0.5
                    tot.append(float(np.sum(U.Y[test] * on)) / max(1, len(np.unique(U.S[test]))))
                score[ml] = float(np.mean(tot))
            best = min(score, key=lambda m: (round(score[m], 9), -m))
            models[t] = honest_tree(U, idx, best, levels, seed + t)
            info[U.types[t]] = {"min_leaf": best, "inner": {str(k): round(v, 6) for k, v in score.items()}}
        else:
            raise ValueError(kind)
    return Fit(kind, U, models, info)


def predicted_total(U, fit, scenarios):
    """Mean over the scenarios of the sum of the effects of the units the fit switches on (additive)."""
    ids = [U.sid[s] for s in scenarios if s in U.sid]
    if not ids:
        return 0.0
    mask = np.isin(U.S, ids)
    on = fit.on(U.X[mask], U.I[mask])
    return float(np.sum(U.Y[mask] * on)) / len(ids)


KINDS = ("me", "tree", "bag", "reg")


def lobo_real(U, real, kinds=KINDS, levels=None, folds=5, seed=0):
    """Leave-buildings-out over the real buildings only: the test scenarios are real, the training scenarios are the
    other folds' real ones plus (variant "all") every generated one; variant "real" trains on real scenarios only
    (the effect of adding the generated buildings)."""
    real = [s for s in real if s in U.sid]
    gen = [s for s in U.scen if s not in set(real)]
    buildings = sorted({s.split("/")[1] for s in real})
    rng = np.random.default_rng(seed)
    fold = {b: i % folds for i, b in enumerate(rng.permutation(buildings))}
    acc = {k: {"all": [], "real": []} for k in kinds}
    acc["none"] = {"all": [], "real": []}
    n_test = 0
    for f in range(folds):
        test = [s for s in real if fold[s.split("/")[1]] == f]
        train_real = [s for s in real if fold[s.split("/")[1]] != f]
        if not test:
            continue
        n_test += len(test)
        for label, train in (("all", train_real + gen), ("real", train_real)):
            for kind in kinds:
                fit = fit_kind(kind, U, train, levels, seed=seed + f)
                acc[kind][label].append(predicted_total(U, fit, test) * len(test))
    out = {k: {lab: float(np.sum(v)) / n_test for lab, v in d.items() if v} for k, d in acc.items() if k != "none"}
    out["none"] = {"all": 0.0, "real": 0.0}
    ids = [U.sid[s] for s in real]
    out["oracle"] = {"all": float(np.mean([float(np.minimum(U.Y[U.S == k], 0.0).sum()) for k in ids]))}
    return out


# ---------------------------------------------------------------- the decision as a policy
def decisions(fit, factors, active):
    """dec[i] = on-vector over GRID for factor i (active ones only)."""
    return {i: fit.on(np.column_stack([np.full(len(GRID), factors[i]["area"]), np.full(len(GRID), factors[i]["wall"]),
                                       GRID]), np.full(len(GRID), i)) for i in active}


def decision_cuts(dec):
    cuts = set()
    for v in dec.values():
        for g in np.where(v[1:] != v[:-1])[0]:
            cuts.add(round(float((GRID[g] + GRID[g + 1]) / 2.0), 6))
    return sorted(cuts)


def dec_function(dec, n):
    def f(i, r):
        if i not in dec:
            return False
        g = int(np.clip(round((r - GRID[0]) / 0.25), 0, len(GRID) - 1))
        return bool(dec[i][g])
    return f


def describe(dec, factors):
    """factor -> the outside-air ranges where it is on, as text."""
    out = {}
    for i, v in dec.items():
        spans, start = [], None
        for g, on in enumerate(v):
            if on and start is None:
                start = g
            if (not on or g == len(v) - 1) and start is not None:
                end = g if on else g - 1
                spans.append([None if start == 0 else float(GRID[start]), None if end == len(v) - 1 else float(GRID[end])])
                start = None
        out[factors[i]["name"]] = spans
    return out
