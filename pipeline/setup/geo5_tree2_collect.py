"""Stage 戊, two levels, T1: the data (docs/plan/tree-two-level-20261106.md 甲; lit-two-level-20261103.md §4 steps 1-2).

- Whole plot (building layer): the full 2^3 of three time / total-airflow options that are not clearly costlier
  on the 188 training scenarios (o545_3 hourly planner, o522_1 night pre-cool, o526_3 cold airflow reduction), on
  the paper rule gt_r5pa2: 8 rows (row 0 = the rule).
- Sub plot (room layer): three room options randomised in two stages inside every AHU (share 0 / 1/2 / 1, then which
  rooms; geo5_tree.two_stage_assign), laid over the rule's own room tree (geo5_tree.overlay_model): cooling target
  0.4 K toward the band edge (the room version of o527_1), heating maximum airflow ratio 0.3, ASHRAE 62.1 zone minimum.
- Every scenario of the tree pool (188: real, the 80 generated of pipeline 4, 60 new seeds, 8 unused TRAIN r1),
  8 rows x 2 randomisation replicates = 3008 episodes; the assignment of each episode in tree2/assign/<name>.json.
- Records in data/runs/b2rl/geo5/tree2 (own pipeline.json / pipeline.log); wakes like pipeline 5 (exit 3).

python3 -B pipeline/launch/launch.py pipeline/setup/geo5_tree2_collect.py [--workers 11] [--plan]
"""
import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo5_lib as L  # noqa: E402
import geo5_options as O  # noqa: E402
import geo5_record as R  # noqa: E402
import geo5_tree as T  # noqa: E402

G = ROOT / "data/runs/b2rl/geo5"
OUT = G / "tree2"
RULE = "gt_r5pa2"
WHOLE = ["o545_3", "o522_1", "o526_3"]
DELTAS = [{"name": "cool_edge", "add": {"room_cool_margin_k": -0.4, "room_cool_relax_k": 0.4}},
          {"name": "heat_ratio_0.3", "set": {"room_heat_max_ratio": 0.3}},
          {"name": "zone_min_62_1", "set": {"zone_min_ra_lps_m2": 0.3, "zone_min_rp_lps_m2": 0.125,
                                            "zone_min_oa_divide": True}}]
REPS = 2


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def feature_job(spec):
    try:
        return spec, T.unit_features(spec), None
    except Exception as error:  # noqa: BLE001
        import traceback
        return spec, None, traceback.format_exc()[-800:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--plan", action="store_true")
    a = p.parse_args()
    spec_opts = json.load(open(G / "options-reviewed.json"))
    by = {lv["id"]: lv for lv in spec_opts["options"]}
    base = json.load(open(ROOT / spec_opts["base"]["model"]))
    tree_state = json.load(open(G / "tree/pipeline.json"))
    rule_ids = tree_state["rule_levels"]
    rule = json.load(open(G / "tree/models" / f"model-{RULE}.json"))
    if O.behaviour_sha(O.build(base, [by[i] for i in rule_ids], RULE)) != O.behaviour_sha(rule):
        raise SystemExit("the rule rebuilt from its levels is not gt_r5pa2")
    for w in WHOLE:
        clash = O.conflicts([by[i] for i in rule_ids] + [by[w]])
        if clash:
            raise SystemExit(f"{w} conflicts with the rule: {clash}")
    rows = [list(bits) for bits in itertools.product((0, 1), repeat=len(WHOLE))]
    whole = []
    for r, bits in enumerate(rows):
        on = [w for w, b in zip(WHOLE, bits) if b]
        m = rule if not on else O.build(base, [by[i] for i in rule_ids + on], f"t2w{r}")
        whole.append({"row": r, "bits": bits, "on": on, "model": m})
    specs = tree_state["specs"]
    print(f"whole plot {WHOLE}: {len(rows)} rows; room options {[d['name'] for d in DELTAS]}; {len(specs)} scenarios "
          f"x {len(rows)} rows x {REPS} replicates = {len(specs) * len(rows) * REPS} episodes")
    if a.plan:
        return 0
    ctx = L.Ctx(OUT, tag="tree2")
    ctx.state.pop("done", None)
    ctx.state.update(rule=RULE, rule_levels=rule_ids, whole=WHOLE, rows=rows, deltas=DELTAS, reps=REPS, specs=specs,
                     started=L.now())
    ctx.save()
    fpath = OUT / "unit_features.json"
    feats = L.read_json(fpath) or {}
    todo = sorted({s for s in specs if s.split("/")[1] not in feats})
    first = {}
    for s in todo:
        first.setdefault(s.split("/")[1], s)
    if first:
        ctx.log("EVENT", f"tree2: room features of {len(first)} buildings")
        with mp.get_context("spawn").Pool(min(8, len(first)), maxtasksperchild=1) as pool:
            for s, f, err in pool.imap_unordered(feature_job, list(first.values())):
                if err:
                    ctx.write_wake(L.Wake("crash", f"room features {s}: {err}"), "tree2.collect", 1, False)
                    return 3
                feats[s.split("/")[1]] = f
        L.atomic_json(fpath, feats)
    models, assign = OUT / "models", OUT / "assign"
    assign.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, s in enumerate(specs):
        feat = feats[s.split("/")[1]]
        for w in whole:
            for k in range(REPS):
                name = f"t2r{w['row']}k{k}s{i:03d}"
                path = models / f"model-{name}.json"
                apath = assign / f"{name}.json"
                if not path.is_file() or not apath.is_file():
                    asg = T.two_stage_assign(feat, len(DELTAS), seed_of(s, w["row"], k))
                    m = T.overlay_model(w["model"], DELTAS, asg)
                    m["source"]["geo5_tree2"] = {"spec": s, "row": w["row"], "rep": k, "whole_on": w["on"]}
                    O.write_model(m, path)
                    L.atomic_json(apath, {"spec": s, "row": w["row"], "bits": w["bits"], "rep": k, **asg})
                jobs.append((i, w["row"], k, ctx.rel(path), s))
    # spread memory: consecutive episodes on different scenarios and rows
    n = len(rows) * REPS
    jobs.sort(key=lambda j: (j[0] // n, (j[1] * REPS + j[2] + j[0]) % n, j[0]))
    pairs = [(j[3], j[4]) for j in jobs]
    ctx.log("EVENT", f"tree2 data: {len(pairs)} episodes ({len(specs)} scenarios x {len(rows)} whole-plot rows x "
                     f"{REPS} room randomisations), workers {a.workers}")
    try:
        R.record(ctx, "tree2 data", pairs, OUT, a.workers)
    except L.Wake as w:
        ctx.write_wake(w, "tree2.collect", 1, False)
        ctx.log("WAKE", f"tree2 data: {w.reason}: {w.text} -> {ctx.rel(ctx.wake_path)}")
        ctx.save()
        return 3
    ctx.state["done"] = L.now()
    ctx.save()
    ctx.log("EVENT", "tree2 data: done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
