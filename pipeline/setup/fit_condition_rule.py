"""Fit the condition-based rule from the step-0 planner's recordings (2026-10-30; user: "Condition-based rule").

Data: data/runs/b2rl/headroom/<tag>/train__*.json written by pipeline/setup/headroom.py in plan mode: at every decision the
reward sum of every option over its window (+ the plain continuation) and the observable features (layout v3, the
current-parameter features left out). For each kind (daily, block) and each option c other than keep, a regression tree
predicts the advantage totals[c] - totals[keep] from the features; the rule takes the option with the largest predicted
advantage if it is above a threshold, otherwise keep.

Model choice (depth, minimum leaf size, threshold) by leave-one-scenario-out folds, scored by the advantage the rule
would really have realized on the held-out decisions (the recorded totals of the option it picks: wrong choices count
with their true, possibly negative, advantage) — a one-step estimate against the policy alone; the closed-loop test
comes after (headroom.py --rule on the held-out scenarios). Writes data/runs/b2rl/headroom/rules/<name>.json (trees as
plain arrays with feature names) and prints the trees.

Round 2 (2026-10-30, stage 1 "前瞻老師和條件式規則的下一輪"): a third argument "forest" fits a random forest per
option (100 trees, a third of the features per split) instead of one tree, on the same data and the same
leave-one-scenario-out choice of depth, leaf size and threshold; headroom.py's tree_value takes the list of trees.

  python3 -B pipeline/launch/launch.py pipeline/setup/fit_condition_rule.py <tag> <name> [forest]"""
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_text

ROOT = Path(__file__).resolve().parents[2]
tag, name = sys.argv[1], sys.argv[2]
FLAGS = set(sys.argv[3:])
FOREST = "forest" in FLAGS
# "geo" (2026-10-30, stage 2B "geometry as ML features"): the building's descriptors of
# data/runs/stage2/descriptors.json (run-time ones only: plan, zones, AHUs; no materials) join every decision's
# features as geo_<name>; the folds then leave one BUILDING out (not one scenario), so a building's other weather
# cannot hand its geometry's answer to the held-out fold. "buildings" alone: the same folds without the geometry.
GEO = "geo" in FLAGS
BY_BUILDING = GEO or "buildings" in FLAGS
GEO_NAMES = ("floors", "ahus", "rooms", "rooms_per_ahu", "area_cells", "room_area_mean", "perimeter_share",
             "exposure_mean", "exposure_area_weighted", "corner_share", "compactness", "aspect")
folder = ROOT / "data/runs/b2rl/headroom" / tag
descriptors = json.loads((ROOT / "data/runs/stage2/descriptors.json").read_text(encoding="utf-8"))["buildings"] \
    if GEO else {}
decisions = []
for path in sorted(folder.glob("train__*.json")):
    row = json.loads(path.read_text(encoding="utf-8"))
    geo = {}
    if GEO:
        pool, building, _ = row["scenario"].split("/")
        geo = {f"geo_{k}": float(descriptors[f"{pool}/{building}"][k]) for k in GEO_NAMES}
    decisions += [{**d, "features": {**d["features"], **geo}, "scenario": row["scenario"],
                   "group": row["scenario"].split("/")[1] if BY_BUILDING else row["scenario"]}
                  for d in row["decisions"] if "totals" in d]
print(f"{len(decisions)} decisions from {len({d['scenario'] for d in decisions})} scenarios, "
      f"{len({d['group'] for d in decisions})} folds; flags {sorted(FLAGS)}")
GRID = [(depth, leaf) for depth in ((4, 6) if FOREST else (1, 2, 3)) for leaf in (20, 50)]
THRESHOLDS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)


def regressor(depth, leaf):
    if FOREST:
        return RandomForestRegressor(n_estimators=100, max_depth=depth, min_samples_leaf=leaf, max_features=1 / 3,
                                     random_state=0, n_jobs=4)
    return DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=0)


def tree_json(tree, names):
    t = tree.tree_
    return {"left": t.children_left.tolist(), "right": t.children_right.tolist(),
            "feature": [names[i] if i >= 0 else "" for i in t.feature], "threshold": t.threshold.tolist(),
            "value": [float(v[0][0]) for v in t.value]}


rule = {"_note": f"condition-based rule fitted by pipeline/setup/fit_condition_rule.py from data/runs/b2rl/headroom/{tag} "
                 f"(2026-10-30)", "tag": tag}
for kind in ("block", "daily"):
    sub = [d for d in decisions if d["kind"] == kind]
    names = [n for n in sub[0]["features"] if not n.startswith("parameter_")]
    options = [c for c in sub[0]["totals"] if c != "keep"]
    X = np.array([[d["features"][n] for n in names] for d in sub])
    A = np.array([[d["totals"][c] - d["totals"]["keep"] for c in options] for d in sub])
    scen = np.array([d["group"] for d in sub])                  # the fold key: scenario, or building (geo)
    oracle = float(np.maximum(A.max(axis=1), 0.0).sum())
    print(f"\n== {kind}: {len(sub)} decisions, options {options}; oracle one-step gain {oracle:.2f} (reward units)")
    best_cfg, best_gain = None, -np.inf
    for depth, leaf in GRID:
        pred = np.zeros_like(A)
        for s in sorted(set(scen)):
            train, test = scen != s, scen == s
            for j in range(len(options)):
                model = regressor(depth, leaf)
                model.fit(X[train], A[train, j])
                pred[test, j] = model.predict(X[test])
        pick = pred.argmax(axis=1)
        top = pred[np.arange(len(sub)), pick]
        realized = A[np.arange(len(sub)), pick]
        for tau in THRESHOLDS:
            take = top > tau
            gain = float(realized[take].sum())
            harm = float(np.minimum(realized[take], 0.0).sum())
            line = (f"  depth {depth} leaf {leaf:2d} threshold {tau:4.2f}: realized {gain:7.2f} "
                    f"({100 * gain / oracle:5.1f}% of oracle), deviates {int(take.sum()):4d}/{len(sub)}, "
                    f"harm {harm:6.2f}")
            if gain > best_gain:
                best_cfg, best_gain = (depth, leaf, tau), gain
            print(line)
    depth, leaf, tau = best_cfg
    print(f"  -> chosen depth {depth}, leaf {leaf}, threshold {tau} (realized {best_gain:.2f}, "
          f"{100 * best_gain / oracle:.1f}% of oracle)")
    trees = {}
    for j, option in enumerate(options):
        model = regressor(depth, leaf)
        model.fit(X, A[:, j])
        if FOREST:
            trees[option] = [tree_json(est, names) for est in model.estimators_]
            continue
        trees[option] = tree_json(model, names)
        print(f"  tree for {option}:\n    " + export_text(model, feature_names=names, decimals=3).replace("\n", "\n    "))
    rule[kind] = {"threshold": tau, "depth": depth, "min_samples_leaf": leaf, "cv_realized": best_gain,
                  "cv_oracle": oracle, "trees": trees}
rule["flags"] = sorted(FLAGS)
if GEO:
    rule["geometry_features"] = list(GEO_NAMES)            # headroom.py rule mode joins them from descriptors.json
out = folder.parent / "rules" / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rule, indent=1) + "\n", encoding="utf-8")
print(f"\nwrote {out.relative_to(ROOT)}")
