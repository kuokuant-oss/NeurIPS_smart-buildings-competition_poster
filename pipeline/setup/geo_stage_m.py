"""Stage M of the per-room geometry stages (2026-10-31, docs/plan/geo-room-stages.md): a per-room option chosen by a tree
on each room's physical quantities and the 24 h outside air, trained on control cost (user: per-room rather than
per-building features; invariant physical quantities; train on the cost of the choice, notes/27 §4.2, §4.5).

1. Data: on search 40 (train_breakdown40), K runs of the base with a "latin" room policy (g36/rooms.py): in run j each
   room takes option (hash(room) + j) % K, so across the K runs every room meets every option while its neighbours
   carry different ones. Each run is recorded with the dashboard recorder (pipeline/setup/dash_record.py episode, per-room
   comfort share + per-room energy split) into data/runs/b2rl/geo/stage_m/records/.
2. Table: one row per (scenario, room): the room's physical quantities (g36/rooms.py room_features, exactly what the
   policy computes), the scenario's mean outside air in occupied hours, and the room's cost under each option
   (comfort share + energy share, divided by the scenario's base-run cost so that every scenario weighs the same).
3. Models, compared by leave-one-building-out regret (5 building folds): a decision-focused policy tree (splits chosen
   to minimise the summed cost of each leaf's best option; leaves need >= MIN_LEAF rooms from >= MIN_GROUPS
   buildings), per-option random-forest cost predictions with argmin (predict-then-optimise), and the single best
   option. The policy tree fitted on all rows becomes the room policy ("tree" mode, re-evaluated hourly with the 24 h
   outside air).
4. Closed loop: held-out 60 and S4 first 30 (fast simulator) against night1_dcv07 and against the base. Passes when
   held-out 60 is >= 0.5 points better than the base and S4 first 30 is not more than 0.3 points worse than the base.
5. Corrections when it does not pass: (1) "mix" runs around the tree (half the rooms follow the tree, half take the
   latin option), added to the data, then refit; (2) a regularised tree (depth 2, larger leaves, features wall, area,
   core, oat). A passing candidate gets INNER 29 and pipeline/setup/geo_gate.py in the background.
Writes data/runs/b2rl/geo/stage_m/result.json; logs to data/runs/b2rl/geo/pipeline.log.

python3 -B pipeline/launch/launch.py pipeline/setup/geo_stage_m.py <base model json> <base name> [--workers 12]
"""
import argparse
import copy
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
GEO = ROOT / "data/runs/b2rl/geo"
OUT = GEO / "stage_m"
SEARCH, HOLDOUT, S4 = "train_breakdown40", "train_ceiling_rest60", "s4_first"
SEED = 7
ROOM_KEYS = ("zone_min_ra_lps_m2", "zone_min_rp_lps_m2", "zone_heat_wall_ref_m", "zone_heat_wall_power",
             "zone_min_oa_divide")
# every per-room minimum here is the code-compliant one (zone_min_oa_divide; geo_pipeline.py eligible)
V62 = {"zone_min_ra_lps_m2": 0.3, "zone_min_rp_lps_m2": 0.125, "zone_min_oa_divide": True}
EXTRA_OPTIONS = [V62,
                 {"zone_heat_wall_ref_m": 30.0, "zone_heat_wall_power": 0.5},
                 {**V62, "zone_heat_wall_ref_m": 30.0, "zone_heat_wall_power": 0.5},
                 {"zone_heat_wall_ref_m": 20.0, "zone_heat_wall_power": 1.0}]
TREE_FEATURES = ("area", "wall", "wall_per_area", "sides", "compact", "core", "ahu_size", "oat")
SMALL_FEATURES = ("area", "wall", "core", "oat")


def log(kind, text):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {kind} stage M: {text}"
    print(line, flush=True)
    with (GEO / "pipeline.log").open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ models of the runs
def room_model(base, name, policy):
    model = copy.deepcopy(base)
    model["room_policy"] = policy
    model.setdefault("source", {})["room_policy"] = name
    path = OUT / "models" / f"model-{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model) + "\n", encoding="utf-8")
    return path


def _record_init():
    import dash_record
    dash_record.OUT = OUT                       # records and buildings go to stage_m/, not the dashboard


def _record(job):
    import dash_record
    try:
        return dash_record.episode(job)[:3]
    except Exception as error:  # noqa: BLE001 - report and continue
        return job[0], job[1], {"error": repr(error)}


def record_runs(models, specs, workers):
    """Dashboard-style records of every (model, scenario) not yet recorded."""
    jobs = []
    for model in models:
        name = Path(model).stem.replace("model-", "")
        jobs += [(str(Path(model).relative_to(ROOT)), s) for s in specs
                 if not (OUT / "records" / f"{name}__{s.replace('/', '__')}.json").is_file()]
    if not jobs:
        return
    log("RESULT", f"recording {len(jobs)} episodes")
    failed = 0
    with mp.get_context("spawn").Pool(min(workers, len(jobs)), initializer=_record_init, maxtasksperchild=1) as pool:
        for policy, spec, summary in pool.imap_unordered(_record, jobs):
            if "error" in summary:
                failed += 1
                log("RESULT", f"record failed {policy} {spec}: {summary['error'][-300:]}")
    if failed:
        raise RuntimeError(f"{failed} recordings failed")


# ------------------------------------------------------------------ features and the table
def _features(spec):
    os.environ["SB_FASTSIM"] = "rom"
    from smart_building.learned.b2rl_env import setup_simulator, spec_dir
    setup_simulator(ROOT)
    from smart_buildings import evaluate
    from smart_building.g36.building import BuildingIndex
    from smart_building.g36 import rooms
    env = evaluate.load_hybrid_environment(spec_dir(spec) / "config.gin")
    env.metrics_path = None
    meta = evaluate.create_building_meta_data(env, spec_dir(spec))
    index = BuildingIndex(meta, env.action_spec())
    feats = rooms.room_features(meta, index)
    env.close()
    return spec, {"rooms": [int(v.split("_")[-1]) for v in index.vav_ids], **{k: v.tolist() for k, v in feats.items()}}


def features(specs, workers):
    path = OUT / "features.json"
    have = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    todo = [s for s in specs if s not in have]
    if todo:
        with mp.get_context("spawn").Pool(min(workers, len(todo)), maxtasksperchild=1) as pool:
            for spec, f in pool.imap_unordered(_features, todo):
                have[spec] = f
        path.write_text(json.dumps(have) + "\n", encoding="utf-8")
    return have


def table(runs, specs, feats, options, base_run):
    """Rows (scenario, room): features dict, per-option cost (normalised by the scenario's base-run cost), building."""
    from smart_building.g36 import rooms
    k = len(options)
    X, C, groups, scen = {f: [] for f in TREE_FEATURES}, [], [], []
    for spec in specs:
        f = feats[spec]
        base_rec = json.loads((OUT / "records" / f"{base_run}__{spec.replace('/', '__')}.json").read_text(encoding="utf-8"))
        norm = base_rec["summary"]["cost"]
        occ = np.asarray(base_rec["hourly"]["occupancy"]) > 0
        oat = float(np.mean(np.asarray(base_rec["hourly"]["outside_t"])[occ])) if occ.any() else \
            float(np.mean(base_rec["hourly"]["outside_t"]))
        sums = np.zeros((len(f["rooms"]), k))
        counts = np.zeros((len(f["rooms"]), k))
        for run_name, policy in runs:
            rec_path = OUT / "records" / f"{run_name}__{spec.replace('/', '__')}.json"
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
            zs = rec["zone_summary"]
            where = {z: j for j, z in enumerate(rec["zones"])}
            arrays = {name: np.asarray(v) for name, v in f.items() if name != "rooms"}
            chosen = rooms.option_index(policy, arrays, f["rooms"], oat)
            for z, number in enumerate(f["rooms"]):
                j = where[f"room_{number}"]
                sums[z, chosen[z]] += (zs["comfort_cost"][j] + zs["energy_cost"][j]) / norm
                counts[z, chosen[z]] += 1
        ok = counts.min(axis=1) > 0
        for z in np.flatnonzero(ok):
            for name in TREE_FEATURES:
                X[name].append(oat if name == "oat" else f[name][z])
            C.append(sums[z] / counts[z])
            groups.append(spec.split("/")[1])
            scen.append(spec)
    return {n: np.asarray(v, dtype=float) for n, v in X.items()}, np.asarray(C), np.asarray(groups), np.asarray(scen)


# ------------------------------------------------------------------ the decision-focused policy tree
def fit_tree(X, C, groups, names, depth, min_leaf, min_groups):
    def node(rows, d):
        totals = C[rows].sum(axis=0)
        best_option, best_cost = int(np.argmin(totals)), float(totals.min())
        if d == 0 or len(rows) < 2 * min_leaf:
            return {"leaf": best_option}, best_cost
        best = None
        for name in names:
            values = X[name][rows]
            for t in np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9))):
                left = rows[values <= t]
                right = rows[values > t]
                if len(left) < min_leaf or len(right) < min_leaf:
                    continue
                if len(set(groups[left])) < min_groups or len(set(groups[right])) < min_groups:
                    continue
                cost = C[left].sum(axis=0).min() + C[right].sum(axis=0).min()
                if best is None or cost < best[0]:
                    best = (cost, name, float(t), left, right)
        if best is None or best[0] >= best_cost - 1e-9:
            return {"leaf": best_option}, best_cost
        left_node, lc = node(best[3], d - 1)
        right_node, rc = node(best[4], d - 1)
        return {"feature": best[1], "threshold": best[2], "left": left_node, "right": right_node}, lc + rc

    return node(np.arange(len(C)), depth)[0]


def predict_tree(tree, X, rows):
    from smart_building.g36.rooms import tree_choice
    return np.array([tree_choice(tree, {n: X[n][i] for n in X}) for i in rows])


def lobo(X, C, groups, method, folds=5):
    """Leave-buildings-out: summed cost of the chosen options on the held-out buildings, over all folds."""
    buildings = np.unique(groups)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(buildings)
    total = 0.0
    for f in range(folds):
        test_b = set(order[f::folds])
        test = np.flatnonzero([g in test_b for g in groups])
        train = np.flatnonzero([g not in test_b for g in groups])
        chosen = method(train, test)
        total += float(C[test, chosen].sum())
    return total


def methods(X, C, groups, depth, min_leaf, min_groups, names):
    def single(train, test):
        return np.full(len(test), int(np.argmin(C[train].sum(axis=0))))

    def base(train, test):
        return np.zeros(len(test), dtype=int)

    def tree(train, test):
        t = fit_tree({n: X[n][train] for n in X}, C[train], groups[train], names, depth, min_leaf, min_groups)
        return predict_tree(t, X, test)

    def forest(train, test):
        from sklearn.ensemble import RandomForestRegressor
        A = np.stack([X[n] for n in names], axis=1)
        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=min_leaf // 2, max_features=0.5,
                                      random_state=SEED, n_jobs=4).fit(A[train], C[train])
        return np.argmin(model.predict(A[test]), axis=1)

    return {"base option": base, "single best option": single, "policy tree": tree, "forest argmin": forest}


# ------------------------------------------------------------------ closed loop
def closed_loop(name, model, base_name, workers):
    import geo_pipeline as gp
    res = {s: gp.evaluate(name, model, s, workers) for s in (HOLDOUT, S4)}
    base_model = ROOT / f"data/runs/b2rl/search/models/model-{base_name}.json"      # its rows are reused if present
    ref = {s: 0.0 if base_name == "night1_dcv07" else gp.evaluate(base_name, base_model, s, workers)["mean"]
           for s in (HOLDOUT, S4)}
    gain = {s: res[s]["mean"] - ref[s] for s in res}
    passed = gain[HOLDOUT] <= -0.5 and gain[S4] <= 0.3
    log("EVENT", f"{name} closed loop: held-out 60 {res[HOLDOUT]['mean']:+.2f}% / S4 first 30 {res[S4]['mean']:+.2f}% vs "
                 f"night1_dcv07; against the base {base_name} {gain[HOLDOUT]:+.2f} / {gain[S4]:+.2f} points; "
                 f"{'PASSED' if passed else 'below the expectation'}")
    return res, gain, passed


def launch_gate(name, model, workers):
    import geo_pipeline as gp
    gp.evaluate(name, model, "inner", workers)
    gate_log = GEO / f"gate-{name}.log"
    subprocess.Popen(["python3", "-B", str(ROOT / "pipeline/setup/geo_gate.py"), name, str(model.relative_to(ROOT))],
                     cwd=ROOT, stdout=open(gate_log, "a", encoding="utf-8"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    log("EVENT", f"gate for {name} started in the background; log {gate_log.relative_to(ROOT)}")


def fit_and_report(tag, X, C, groups, names, depth, min_leaf, min_groups):
    scores = {k: lobo(X, C, groups, m) for k, m in methods(X, C, groups, depth, min_leaf, min_groups, names).items()}
    base = scores["base option"]
    log("EVENT", f"{tag} leave-buildings-out cost (rooms {len(C)}, buildings {len(set(groups))}): "
                 + ", ".join(f"{k} {100 * (v / base - 1):+.2f}%" for k, v in scores.items()) + " vs the base option")
    tree = fit_tree(X, C, groups, names, depth, min_leaf, min_groups)
    return tree, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_model")
    parser.add_argument("base_name")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / args.base_model).read_text(encoding="utf-8"))
    option0 = {k: v for k, v in base.get("rule_config", {}).items() if k in ROOM_KEYS}
    options = [option0] + [o for o in EXTRA_OPTIONS if o != option0]
    k = len(options)
    specs = json.loads((ROOT / "configs/b2rl/eval-sets.json").read_text(encoding="utf-8"))[SEARCH]
    result = {"base": args.base_name, "options": options, "attempts": [], "passed": None, "candidate": None}
    log("EVENT", f"starts from {args.base_name} with {k} per-room options; data on {SEARCH} ({k} latin runs)")

    latin = [(f"gm_latin{j}", {"mode": "latin", "seed": SEED, "shift": j, "options": options}) for j in range(k)]
    fixed0 = ("gm_fixed0", {"mode": "fixed", "option": 0, "options": options})
    runs = latin
    models = [room_model(base, n, p) for n, p in latin + [fixed0]]
    record_runs(models, specs, args.workers)
    feats = features(specs, args.workers)
    X, C, groups, _ = table(runs, specs, feats, options, "gm_fixed0")

    attempts = [("m1", TREE_FEATURES, 3, 60, 6, None), ("m2", TREE_FEATURES, 3, 60, 6, "mix"),
                ("m3", SMALL_FEATURES, 2, 150, 10, None)]
    for tag, names, depth, min_leaf, min_groups, extra in attempts:
        if extra == "mix":                      # first correction: data around the tree itself
            mix = [(f"gm_mix{j}", {"mode": "mix", "seed": SEED, "shift": j, "tree": tree, "options": options})
                   for j in range(k)]
            record_runs([room_model(base, n, p) for n, p in mix], specs, args.workers)
            runs = runs + mix
            X, C, groups, _ = table(runs, specs, feats, options, "gm_fixed0")
        tree, scores = fit_and_report(tag, X, C, groups, names, depth, min_leaf, min_groups)
        name = f"gm_tree_{tag}"
        policy = {"mode": "tree", "tree": tree, "options": options}
        model = room_model(base, name, policy)
        final = ROOT / f"data/runs/b2rl/search/models/model-{name}.json"     # the gate and submit tools look here
        final.write_text(model.read_text(encoding="utf-8"), encoding="utf-8")
        log("EVENT", f"{tag} tree: {json.dumps(tree)[:500]}")
        res, gain, passed = closed_loop(name, final, args.base_name, args.workers)
        result["attempts"].append({"tag": tag, "tree": tree, "lobo": scores, "closed_loop": res, "against_base": gain,
                                   "passed": passed})
        (OUT / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        if passed:
            result["candidate"] = name
            launch_gate(name, final, args.workers)
            result["gate_started"] = True
            break
    (OUT / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    log("EVENT", f"ends: candidate {result['candidate']}")


if __name__ == "__main__":
    main()
