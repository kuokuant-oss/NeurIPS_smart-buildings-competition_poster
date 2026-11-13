"""One per-room stage (2026-11-01, docs/plan/geo-room-stages-2.md): record room-policy runs, build the per-room cost
table (optionally per time block), fit the decision-focused policy tree, check it in closed loop against the current
base on held-out 60, INNER 29 and S4 first 30, and start the Kaggle gate when it passes. Shared by
pipeline/setup/geo_pipeline2.py; the tree, the leave-buildings-out comparison and the gate come from geo_stage_m.py.

Pass (item 5 of the plan): held-out 60 and INNER 29 each >= 0.5 points better than the base, S4 first 30 not more than
0.5 points worse. Costs per room: comfort share + energy share from the dashboard recorder, divided by the scenario's
base-run cost; per time block from the hourly per-zone arrays (zone_comfort_1e8, zone_energy_1e8)."""
import copy
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_stage_m as gsm  # noqa: E402

GEO = ROOT / "data/runs/b2rl/geo"
SEARCH, HOLDOUT, S4, INNER = "train_breakdown40", "train_ceiling_rest60", "s4_first", "inner"
FEATURE_CACHE = GEO / "room_features.json"
MODELS = ROOT / "data/runs/b2rl/search/models"


def log(kind, text):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {kind} stage2: {text}"
    print(line, flush=True)
    with (GEO / "pipeline.log").open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def room_model(out, base, name, policy):
    model = copy.deepcopy(base)
    model.pop("room_policy", None)
    model["room_policy"] = policy
    model.setdefault("source", {})["room_policy"] = name
    path = out / "models" / f"model-{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model) + "\n", encoding="utf-8")
    return path


def _init(out):
    import dash_record
    dash_record.OUT = Path(out)


def _record(job):
    import dash_record
    try:
        return dash_record.episode(job)[:3]
    except Exception as error:  # noqa: BLE001 - report and continue
        return job[0], job[1], {"error": repr(error)}


def record(out, runs, specs, workers, also=()):
    """Dashboard-style records of every (run model, scenario) not yet in out/records or in the records of `also`."""
    jobs = []
    for model in runs:
        name = Path(model).stem.replace("model-", "")
        jobs += [(str(Path(model).relative_to(ROOT)), s) for s in specs
                 if not any((d / "records" / f"{name}__{s.replace('/', '__')}.json").is_file()
                            for d in (out, *also))]
    if not jobs:
        return
    log("RESULT", f"{out.name}: recording {len(jobs)} episodes")
    failed = 0
    with mp.get_context("spawn").Pool(min(workers, len(jobs)), initializer=_init, initargs=(str(out),),
                                      maxtasksperchild=1) as pool:
        for policy, spec, summary in pool.imap_unordered(_record, jobs):
            if "error" in summary:
                failed += 1
                log("RESULT", f"record failed {policy} {spec}: {summary['error'][-300:]}")
    if failed > max(2, len(jobs) // 50):
        raise RuntimeError(f"{failed} of {len(jobs)} recordings failed")


def features(specs, workers):
    have = json.loads(FEATURE_CACHE.read_text(encoding="utf-8")) if FEATURE_CACHE.is_file() else {}
    todo = [s for s in specs if s not in have]
    if todo:
        with mp.get_context("spawn").Pool(min(workers, len(todo)), maxtasksperchild=1) as pool:
            for spec, f in pool.imap_unordered(gsm._features, todo):
                have[spec] = f
        FEATURE_CACHE.write_text(json.dumps(have) + "\n", encoding="utf-8")
    return have


def find_record(dirs, run_name, spec):
    for d in dirs:
        path = d / "records" / f"{run_name}__{spec.replace('/', '__')}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def table(dirs, runs, specs, feats, options, base_run, blocks=None):
    """Rows (scenario, room[, block]): features, per-option normalised cost, building."""
    from smart_building.g36 import rooms
    k = len(options)
    names = list(gsm.TREE_FEATURES) + (["block"] if blocks else [])
    X, C, groups = {f: [] for f in names}, [], []
    nb = len(blocks) if blocks else 1
    for spec in specs:
        f = feats[spec]
        base_rec = find_record(dirs, base_run, spec)
        if base_rec is None:
            continue
        norm = base_rec["summary"]["cost"]
        occ = np.asarray(base_rec["hourly"]["occupancy"]) > 0
        out_t = np.asarray(base_rec["hourly"]["outside_t"])
        oat = float(np.mean(out_t[occ])) if occ.any() else float(np.mean(out_t))
        arrays = {name: np.asarray(v) for name, v in f.items() if name != "rooms"}
        n_room = len(f["rooms"])
        sums, counts = np.zeros((n_room, nb, k)), np.zeros((n_room, nb, k))
        for run_name, policy in runs:
            rec = find_record(dirs, run_name, spec)
            if rec is None:
                continue
            where = {z: j for j, z in enumerate(rec["zones"])}
            cols = [where[f"room_{number}"] for number in f["rooms"]]
            if blocks:
                start = int(rec["start_clock"][11:13])
                comfort = np.asarray(rec["hourly"]["zone_comfort_1e8"], dtype=float)[cols] / 1e8
                energy = np.asarray(rec["hourly"]["zone_energy_1e8"], dtype=float)[cols] / 1e8
                hours = comfort.shape[1]
                per_hour = (comfort + energy) * 12.0 / rec["steps"] / norm          # share of the episode mean
                block_of_hour = np.array([rooms.block_of({"blocks": blocks}, (start + h) % 24) for h in range(hours)])
                for b, start_hour in enumerate(blocks):
                    chosen = rooms.option_index(policy, arrays, f["rooms"], oat, start_hour)
                    cost_b = per_hour[:, block_of_hour == b].sum(axis=1)
                    sums[np.arange(n_room), b, chosen] += cost_b
                    counts[np.arange(n_room), b, chosen] += 1
            else:
                zs = rec["zone_summary"]
                chosen = rooms.option_index(policy, arrays, f["rooms"], oat)
                cost = np.array([(zs["comfort_cost"][j] + zs["energy_cost"][j]) / norm for j in cols])
                sums[np.arange(n_room), 0, chosen] += cost
                counts[np.arange(n_room), 0, chosen] += 1
        for z in range(n_room):
            for b in range(nb):
                if counts[z, b].min() == 0:
                    continue
                for name in names:
                    X[name].append(oat if name == "oat" else float(b) if name == "block" else f[name][z])
                C.append(sums[z, b] / counts[z, b])
                groups.append(spec.split("/")[1])
    return {n: np.asarray(v, dtype=float) for n, v in X.items()}, np.asarray(C), np.asarray(groups)


def closed_loop(name, model, base_name, workers):
    import geo_pipeline as gp
    base_model = MODELS / f"model-{base_name}.json"
    res, gain = {}, {}
    for s in (HOLDOUT, INNER, S4):
        res[s] = gp.evaluate(name, model, s, workers)
        ref = 0.0 if base_name == "night1_dcv07" else gp.evaluate(base_name, base_model, s, workers)["mean"]
        gain[s] = res[s]["mean"] - ref
    passed = gain[HOLDOUT] <= -0.5 and gain[INNER] <= -0.5 and gain[S4] <= 0.5
    log("EVENT", f"{name} closed loop vs night1_dcv07: held-out 60 {res[HOLDOUT]['mean']:+.2f}%, INNER 29 "
                 f"{res[INNER]['mean']:+.2f}%, S4 first 30 {res[S4]['mean']:+.2f}%; against the base {base_name} "
                 f"{gain[HOLDOUT]:+.2f} / {gain[INNER]:+.2f} / {gain[S4]:+.2f} points; "
                 f"{'PASSED' if passed else 'below the expectation'}")
    return res, gain, passed


def attempt(tag, base, base_name, dirs, runs, specs, feats, options, blocks, spec, workers):
    """Fit one tree (spec: features, depth, min_leaf, min_groups), check it, start the gate when it passes."""
    X, C, groups = table(dirs, runs, specs, feats, options, "gm_fixed0", blocks)
    names = tuple(spec["features"]) + (("block",) if blocks and "block" not in spec["features"] else ())
    scores = {k: gsm.lobo(X, C, groups, m) for k, m in gsm.methods(X, C, groups, spec["depth"], spec["min_leaf"],
                                                                   spec["min_groups"], names).items()}
    b0 = scores["base option"]
    log("EVENT", f"{tag} leave-buildings-out (rows {len(C)}, buildings {len(set(groups))}): "
                 + ", ".join(f"{k} {100 * (v / b0 - 1):+.2f}%" for k, v in scores.items()) + " vs the base option")
    tree = gsm.fit_tree(X, C, groups, names, spec["depth"], spec["min_leaf"], spec["min_groups"])
    name = f"gm_{tag}"
    policy = {"mode": "tree", "tree": tree, "options": options, **({"blocks": blocks} if blocks else {})}
    model = room_model(GEO / "stage2", base, name, policy)
    final = MODELS / f"model-{name}.json"
    final.write_text(model.read_text(encoding="utf-8"), encoding="utf-8")
    log("EVENT", f"{tag} tree: {json.dumps(tree)[:500]}")
    res, gain, passed = closed_loop(name, final, base_name, workers)
    if passed:
        gsm.launch_gate(name, final, workers)
    return {"tag": tag, "name": name, "tree": tree, "lobo": scores, "closed_loop": res, "against_base": gain,
            "passed": passed}
