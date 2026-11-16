"""Analysis behind correction T1-1, part 2 (2026-11-01): the climate split recovers about 0.6 of the 2.6 points
between the global choice (-3.36%) and the per-scenario oracle (-5.95%). Which scenario-level physical quantities
(known to the controller: the running mean outside air; building traits from the floor plan) explain each factor's
sign? Per factor, the best single-threshold rule (off / on / on below or above a threshold of one trait) chosen on the
training buildings, judged leave-buildings-out; compared with the global and the climate-only rules.
python3 -B pipeline/launch/launch.py pipeline/setup/geo3_corr/t1_trait_check.py"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "pipeline/setup"))
import geo_pipeline3 as gp3
import group_stage as gs
import room_stage as rs

TRAITS = ("oat_mean", "small_share", "long_share", "core_share", "ahu_size", "rooms")
MIN_SIDE = 10


def rules(vals_by_spec, trait_by_spec, specs, allowed):
    """Candidate rules for one factor: (name, function spec -> on)."""
    out = [("off", lambda s: False), ("on", lambda s: True)]
    for tname in allowed:
        xs = np.array([trait_by_spec[s][tname] for s in specs])
        for q in np.unique(np.percentile(xs, np.arange(15, 90, 7.5))):
            if min(np.sum(xs <= q), np.sum(xs > q)) < MIN_SIDE:
                continue
            out.append((f"{tname}<={q:.2f}", lambda s, t=tname, q=q: trait_by_spec[s][t] <= q))
            out.append((f"{tname}>{q:.2f}", lambda s, t=tname, q=q: trait_by_spec[s][t] > q))
    return out


def best_rule(per, i, trait_by_spec, specs, allowed):
    best = None
    for name, fn in rules(None, trait_by_spec, specs, allowed):
        v = np.mean([per[s]["total"][i] if (fn(s) and per[s]["total"][i] is not None) else 0.0 for s in specs])
        if best is None or v < best[0] - 1e-12:
            best = (v, name, fn)
    return best


def main():
    state = json.loads(gp3.STATE.read_text(encoding="utf-8"))
    specs = state["specs"]
    feats = json.loads(rs.FEATURE_CACHE.read_text(encoding="utf-8"))
    factors, _ = gp3.stage_factors("T1")
    runs = gs.design_runs("gt_t1_pb", {}, factors)
    per = gs.estimate(specs, feats, factors, runs, gp3.DIRS, "gm_fixed0")
    tr = {}
    for s in per:
        t = gs.scenario_traits(s, feats, gp3.DIRS)
        t["core_share"] = float(np.mean([c[1] == "core" for c in gs.room_cells(feats, s)]))
        tr[s] = t
    buildings = sorted({s.split("/")[1] for s in per})
    rng = np.random.default_rng(0)
    fold = {b: i % 5 for i, b in enumerate(rng.permutation(buildings))}
    report = {}
    for label, allowed in (("climate only", ("oat_mean",)), ("building traits only", TRAITS[1:]), ("all", TRAITS)):
        preds = []
        for k in range(5):
            train = [s for s in per if fold[s.split("/")[1]] != k]
            test = [s for s in per if fold[s.split("/")[1]] == k]
            fns = [best_rule(per, i, tr, train, allowed)[2] for i in range(len(factors))]
            for s in test:
                preds.append(sum(per[s]["total"][i] for i, fn in enumerate(fns)
                                 if fn(s) and per[s]["total"][i] is not None))
        full = {f["name"]: best_rule(per, i, tr, list(per), allowed)[1] for i, f in enumerate(factors)}
        report[label] = {"lobo %": round(100 * float(np.mean(preds)), 2), "rules on all 80": full}
        print(f"{label:22s} leave-buildings-out {100 * np.mean(preds):+.2f}%")
        print("   ", full)
    (gp3.STATE_DIR / "t1_trait_check.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
