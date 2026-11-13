"""Group-factor stages, learned from building totals (2026-11-01, docs/plan/geo-room-stages-3.md). Follows the
diagnosis of the per-room tree (data/runs/b2rl/geo/diag/): per-room cost shares overstate the saving (the cooling
coil part about 2x; a room's airflow also changes its neighbours' and AHU mates' bills), the totals of the existing
runs are confounded, and the tree's outside-air feature meant one thing in training and another when deployed.

So: rooms fall in 12 cells (area <= 95 / 95-133 / 133-250 / > 250 m2 x no exterior wall / wall <= 20 m / > 20 m);
a factor switches one setting (a per-room option delta) on for a set of cells, for all of their rooms together; a
12-run Plackett-Burman design over up to 11 factors is recorded on every design scenario, and each factor's effect is
estimated from the building totals of that scenario's runs (main effects; the all-off row is the stage's base map).
A candidate is a cell map (cell [x time block] -> option dict), expressed as a room-policy tree over area / wall
[/ block], so rooms.py and the submission need no change. No outside-air feature.

Used by pipeline/setup/geo_pipeline3.py."""
import json
from pathlib import Path
import statistics as st
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import room_stage as rs  # noqa: E402

GEO = ROOT / "data/runs/b2rl/geo"
OUT3 = GEO / "stage3"
DIAG = GEO / "diag"
PARTS = ("comfort_cost", "fan_cost", "coil_cost", "gas_cost", "pump_cost")
AREA_EDGES = (95.0, 133.0, 250.0)
AREAS = ("s", "m", "l", "x")
WALLS = ("core", "perim", "long")
CELLS = [(a, w) for a in AREAS for w in WALLS]
PB12 = [[1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1]]
for _ in range(10):
    PB12.append(PB12[-1][-1:] + PB12[-1][:-1])
PB12.append([-1] * 11)                                    # row 12: every factor off = the base map


def cell_of(area, wall):
    a = AREAS[sum(area > e for e in AREA_EDGES)]
    w = "core" if wall <= 0.1 else ("perim" if wall <= 20.0 else "long")
    return a, w


def room_cells(feats, spec):
    f = feats[spec]
    return [cell_of(a, w) for a, w in zip(f["area"], f["wall"])]


def _key(cell, block):
    return f"{cell[0]}|{cell[1]}|{block}"


def is_climate(cell_map):
    """A climate map (correction T1-1): {"oat_cuts": [deg C, ...], "maps": [cell map per climate bin]}; the bin is
    chosen by the running mean outside air since the episode start (b2rl.py room policy "oat_mode": "episode")."""
    return isinstance(cell_map, dict) and "oat_cuts" in cell_map


def is_empty(cell_map):
    if is_climate(cell_map):
        return all(is_empty(m) for m in cell_map["maps"])
    return all(not v for v in cell_map.values())


def compose(base_map, factors, levels, nblocks=1):
    """Cell map (key 'a|w|block' -> option dict): the base map plus the delta of every factor that is on."""
    if is_climate(base_map):
        return {"oat_cuts": list(base_map["oat_cuts"]),
                "maps": [compose(m, factors, levels, nblocks) for m in base_map["maps"]]}
    out = {}
    for cell in CELLS:
        for b in range(nblocks):
            opt = dict(base_map.get(_key(cell, b), base_map.get(_key(cell, 0), {})))
            for fac, on in zip(factors, levels):
                if on > 0 and list(cell) in [list(c) for c in fac["cells"]] and \
                        (fac.get("blocks") is None or b in fac["blocks"]):
                    opt.update(fac["delta"])
            out[_key(cell, b)] = opt
    return out


def policy_of(cell_map, blocks=None):
    """Room-policy tree over area, wall [and block] for a cell map; a climate map adds a root chain over the running
    mean outside air ("oat_mode": "episode")."""
    maps = cell_map["maps"] if is_climate(cell_map) else [cell_map]
    options = [{}]
    for m in maps:
        for k in sorted(m):
            if m[k] not in options:
                options.append(m[k])
    trees = [_cell_tree(m, blocks, options) for m in maps]
    if not is_climate(cell_map):
        return {"mode": "tree", "tree": trees[0], "options": options, **({"blocks": list(blocks)} if blocks else {})}
    cuts = cell_map["oat_cuts"]
    tree = trees[-1]
    for i in range(len(cuts) - 1, -1, -1):
        tree = {"feature": "oat", "threshold": float(cuts[i]), "left": trees[i], "right": tree}
    return {"mode": "tree", "tree": tree, "options": options, "oat_mode": "episode",
            **({"blocks": list(blocks)} if blocks else {})}


def _cell_tree(cell_map, blocks, options):
    nb = len(blocks) if blocks else 1

    def block_node(cell):
        leaves = [{"leaf": options.index(cell_map[_key(cell, b)])} for b in range(nb)]
        node = leaves[-1]
        for b in range(nb - 2, -1, -1):
            node = {"feature": "block", "threshold": b + 0.5, "left": leaves[b], "right": node}
        return node

    def wall_node(a):
        return {"feature": "wall", "threshold": 0.1, "left": block_node((a, "core")),
                "right": {"feature": "wall", "threshold": 20.0, "left": block_node((a, "perim")),
                          "right": block_node((a, "long"))}}

    tree = wall_node(AREAS[-1])
    for i in range(len(AREA_EDGES) - 1, -1, -1):
        tree = {"feature": "area", "threshold": AREA_EDGES[i], "left": wall_node(AREAS[i]), "right": tree}
    return tree


def design_runs(prefix, base_map, factors, nblocks=1):
    """The 12 Plackett-Burman rows as (run name, levels, cell map)."""
    k = len(factors)
    assert k <= 11
    return [(f"{prefix}{r:02d}", row[:k], compose(base_map, factors, row[:k], nblocks)) for r, row in enumerate(PB12)]


def present(factors, cells, nblocks=1):
    """Whether each factor touches at least one room of the scenario."""
    have = {tuple(c) for c in cells}
    return [any(tuple(c) in have for c in fac["cells"]) for fac in factors]


def estimate(specs, feats, factors, runs, dirs, base_run=None):
    """Per-scenario main effects (cost change as a share of the base-map run when the factor goes off -> on), in
    total and by part. runs: design_runs output; base_run: an existing record name to use for row 12 (all off)."""
    per = {}
    for spec in specs:
        recs = []
        for name, levels, _ in runs:
            rec = rs.find_record(dirs, base_run if (base_run and all(v < 0 for v in levels)) else name, spec)
            if rec is None:
                break
            recs.append((levels, rec))
        if len(recs) < len(runs):
            continue
        base = [r for lv, r in recs if all(v < 0 for v in lv)][0]
        norm = base["summary"]["cost"]
        X = np.array([[1.0] + list(lv) for lv, _ in recs])
        ys = {"total": np.array([r["summary"]["cost"] / norm - 1.0 for _, r in recs])}
        for p in PARTS:
            b = float(np.sum(base["zone_summary"][p]))
            ys[p] = np.array([(float(np.sum(r["zone_summary"][p])) - b) / norm for _, r in recs])
        pres = present(factors, room_cells(feats, spec))
        cols = [0] + [i + 1 for i, ok in enumerate(pres) if ok]
        out = {}
        for key, y in ys.items():
            beta, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
            eff = [None] * len(factors)
            for j, c in enumerate(cols[1:], start=1):
                eff[c - 1] = 2.0 * float(beta[j])
            out[key] = eff
        per[spec] = out
    return per


def summarise(per, factors, key="total", specs=None):
    rows = {}
    for i, fac in enumerate(factors):
        vals = [per[s][key][i] for s in (specs or per) if s in per and per[s][key][i] is not None]
        if len(vals) >= 2:
            rows[fac["name"]] = {"mean": float(np.mean(vals)), "se": float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
                                 "n": len(vals)}
        else:
            rows[fac["name"]] = {"mean": 0.0, "se": float("inf"), "n": len(vals)}
    return rows


def choose(summary, factors, t=1.0):
    """Main rule: a factor goes on when its mean effect is below zero by more than t standard errors."""
    return [1 if summary[f["name"]]["mean"] + t * summary[f["name"]]["se"] < 0 else -1 for f in factors]


def predicted(per, levels, specs=None, key="total"):
    """Mean over scenarios of the sum of the chosen factors' own effects (vs the base map)."""
    vals = []
    for s in specs or per:
        if s in per:
            vals.append(sum(e for e, on in zip(per[s][key], levels) if on > 0 and e is not None))
    return float(np.mean(vals)) if vals else 0.0


def lobo(per, factors, t=1.0, folds=5):
    """Leave-buildings-out: choose on the other buildings, predict on the held-out ones."""
    specs = sorted(per)
    buildings = sorted({s.split("/")[1] for s in specs})
    rng = np.random.default_rng(0)
    fold = {b: i % folds for i, b in enumerate(rng.permutation(buildings))}
    preds, oracle = [], []
    for i in range(folds):
        train = [s for s in specs if fold[s.split("/")[1]] != i]
        test = [s for s in specs if fold[s.split("/")[1]] == i]
        if not test:
            continue
        lv = choose(summarise(per, factors, specs=train), factors, t)
        preds += [predicted(per, lv, [s]) for s in test]
        oracle += [sum(min(e, 0.0) for e in per[s]["total"] if e is not None) for s in test]
    return {"lobo_predicted": float(np.mean(preds)), "per_scenario_oracle": float(np.mean(oracle))}


def scenario_traits(spec, feats, dirs):
    base = rs.find_record(dirs, "gm_fixed0", spec)
    cells = room_cells(feats, spec)
    return {"station": spec.split("/")[2].rstrip("_0123456789"),
            "oat_mean": float(np.mean(base["hourly"]["outside_t"])),
            "small_share": float(np.mean([c[0] == "s" for c in cells])),
            "long_share": float(np.mean([c[1] == "long" for c in cells])),
            "ahu_size": float(np.mean(feats[spec]["ahu_size"])),
            "rooms": len(cells)}


def cell_costs(rec, spec, feats, norm):
    """Each cell's summed allocated room cost (share of norm), for the where-did-it-change view."""
    f = feats[spec]
    where = {z: j for j, z in enumerate(rec["zones"])}
    out = {}
    for r, cell in zip(f["rooms"], room_cells(feats, spec)):
        j = where[f"room_{r}"]
        zs = rec["zone_summary"]
        out[f"{cell[0]}|{cell[1]}"] = out.get(f"{cell[0]}|{cell[1]}", 0.0) + \
            (zs["comfort_cost"][j] + zs["energy_cost"][j]) / norm
    return out


def eval_rows(name, subset):
    path = GEO / subset / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["rows"] if path.is_file() else {}


def by_station(cand, base, subset):
    """Closed-loop change of cand vs base (points of the base cost) per station of the set."""
    import geo_pipeline as gp
    a, b = eval_rows(cand, subset), eval_rows(base, subset)
    out = {}
    for s in sorted(set(a) & set(b)):
        stn = s.split("/")[2].rstrip("_0123456789")
        out.setdefault(stn, []).append(100 * (gp.cost(a[s]) / gp.cost(b[s]) - 1))
    return {k: [round(st.mean(v), 2), len(v)] for k, v in out.items()}
