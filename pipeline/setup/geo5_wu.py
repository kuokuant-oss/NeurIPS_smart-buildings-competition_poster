"""Stage 戊 (the tree on the paper rule gt_r5pa2), data and effects (2026-11-06; spec docs/plan/tree-stage-wu-20261106.md).

Round 4's machinery (group_stage4.py: F, compose_one / compose_design, policy_of, conflicts, estimate) on the paper
rule, with round 4's preset-free changes only:
- the base map is round 4's gt_u1b cell map (geo4 pipeline.json build_on); policy_of(it) must be gt_r5pa2's room
  policy bit for bit;
- 3 building-wide columns (whole plot: o538_2, o522_1, o545_3, rule levels switched on for the whole building) next to
  6 room factors (a round-4 setting x a cell group, extension only, in every climate bin of the base map);
- 12-run Plackett-Burman + its fold-over (24 runs; row 12 = all off = gt_r5pa2, its records reused from geo5/tree);
- effects from the nominal whole-building total cost of each scenario's 24 runs.

Collect: python3 -B pipeline/launch/launch.py pipeline/setup/geo5_wu.py collect [--workers 11]
Checks only: python3 -B pipeline/launch/launch.py pipeline/setup/geo5_wu.py check [--smoke-steps 288]"""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo5_lib as L  # noqa: E402
import geo5_options as O  # noqa: E402
import geo5_record as R  # noqa: E402
import group_stage as gs  # noqa: E402
import group_stage4 as g4  # noqa: E402

G = ROOT / "data/runs/b2rl/geo5"
OUT = G / "wu"
TREE = G / "tree"
RULE = "gt_r5pa2"
SM, LX, ALL = g4.SM, g4.LX, g4.ALL
EXT, NONLONG = g4.EXT, g4.NONLONG

WHOLE = ["o538_2", "o522_1", "o545_3"]
ROOM = [g4.F("ratio03_SMe", "ratio0.3", SM, EXT, {"room_heat_max_ratio": 0.3}),
        g4.F("ratio03_SMc", "ratio0.3", SM, ["core"], {"room_heat_max_ratio": 0.3}),
        g4.F("heatrelax04_core", "heat-0.4", ALL, ["core"], g4.margin("heat", -0.4)),
        g4.F("cool04_SMn", "cool+0.4", SM, NONLONG, g4.margin("cool", 0.4)),
        g4.F("cool04_LXn", "cool+0.4", LX, NONLONG, g4.margin("cool", 0.4)),
        g4.F("ratio06_LXe", "ratio0.6", LX, EXT, {"room_heat_max_ratio": 0.6})]


def whole_factor(level):
    return {"name": level, "type": f"bldg:{level}", "cells": [], "delta": {}, "periods": None, "whole": True,
            "area": float("nan"), "wall": float("nan")}


FACTORS = [whole_factor(w) for w in WHOLE] + ROOM
NW = len(WHOLE)


def rows24():
    k = len(FACTORS)
    first = [r[:k] for r in gs.PB12]
    return [(f"wur{i + 1:02d}", r) for i, r in enumerate(first)] + \
           [(f"wuf{i + 1:02d}", [-v for v in r]) for i, r in enumerate(first)]


def base_map():
    return json.loads((ROOT / "data/runs/b2rl/geo4/pipeline.json").read_text(encoding="utf-8"))["build_on"]


def rule_parts():
    spec = json.loads((G / "options-reviewed.json").read_text(encoding="utf-8"))
    by = {lv["id"]: lv for lv in spec["options"]}
    base = json.loads((ROOT / spec["base"]["model"]).read_text(encoding="utf-8"))
    rule_ids = json.loads((TREE / "pipeline.json").read_text(encoding="utf-8"))["rule_levels"]
    return by, base, rule_ids


def build_run(name, levels, by, base, rule_ids, bmap):
    """The model of one design row: rule + the whole-plot levels that are on, room policy = policy_of(composed map)."""
    on_whole = [WHOLE[i] for i in range(NW) if levels[i] > 0]
    model = O.build(base, [by[i] for i in rule_ids + on_whole], name)
    cmap = g4.compose_design(bmap, ROOM, levels[NW:], 1)
    model["room_policy"] = g4.policy_of(cmap)
    model["source"]["geo5_wu"] = {"levels": list(levels), "whole_on": on_whole,
                                  "room_on": [ROOM[j]["name"] for j in range(len(ROOM)) if levels[NW + j] > 0]}
    return model, cmap


def plan():
    by, base, rule_ids = rule_parts()
    bmap = base_map()
    rule = json.loads((TREE / "models" / f"model-{RULE}.json").read_text(encoding="utf-8"))
    runs = []
    for name, lv in rows24():
        off = all(v < 0 for v in lv)
        model, cmap = build_run(RULE if off else name, lv, by, base, rule_ids, bmap)
        runs.append({"name": RULE if off else name, "levels": lv, "model": model, "map": cmap,
                     "sha": O.behaviour_sha(model)})
    specs = json.loads((TREE / "pipeline.json").read_text(encoding="utf-8"))["specs"]
    sets = json.loads((ROOT / "configs/b2rl/eval-sets.json").read_text(encoding="utf-8"))
    real = [s for s in specs if s in set(sets["train_breakdown40"])]
    fresh = json.loads((ROOT / "data/runs/b2rl/geo2/pipeline.json").read_text(encoding="utf-8"))["fresh80"]
    order = real + [s for s in specs if s in set(fresh) and s not in set(real)] + \
        [s for s in specs if s not in set(real) and s not in set(fresh)]
    assert sorted(order) == sorted(specs)
    return {"by": by, "base": base, "rule_ids": rule_ids, "bmap": bmap, "rule": rule, "runs": runs, "specs": order,
            "real": real, "sets": sets}


# ---------------------------------------------------------------- checks
def checks(P, feats=None):
    """Raise on any failed check; return a dict of what was checked."""
    out = {}
    rule = P["rule"]
    off = [r for r in P["runs"] if r["name"] == RULE]
    assert len(off) == 1
    if off[0]["model"]["room_policy"] != rule["room_policy"]:
        raise SystemExit("policy_of(round 4 base map) is not gt_r5pa2's room policy")
    if off[0]["sha"] != O.behaviour_sha(rule):
        raise SystemExit("the all-off row does not behave as gt_r5pa2")
    out["all_off_is_rule"] = True
    shas = [r["sha"] for r in P["runs"]]
    if len(set(shas)) != len(shas):
        raise SystemExit("two design rows behave the same")
    out["distinct_rows"] = len(shas)
    if g4.conflicts(ROOM):
        raise SystemExit(f"room factors conflict: {g4.conflicts(ROOM)}")
    clash = O.conflicts([P["by"][i] for i in P["rule_ids"] + WHOLE])
    if clash:
        raise SystemExit(f"whole-plot levels conflict with the rule: {clash}")
    for w in WHOLE:
        if P["by"][w]["category"] != "physics":
            raise SystemExit(f"{w} is not a physics level")
    X = np.array([r["levels"] for r in P["runs"]], float)
    xtx = X.T @ X
    if not np.allclose(xtx, 24 * np.eye(len(FACTORS))) or not np.allclose(X.sum(0), 0):
        raise SystemExit("design columns not orthogonal / balanced")
    two = np.array([[r[i] * r[j] for i in range(len(FACTORS)) for j in range(i + 1, len(FACTORS))] for r in X])
    if not np.allclose(X.T @ two, 0):
        raise SystemExit("main effects aliased with two-factor interactions in the fold-over")
    out["design"] = "24 runs, columns orthogonal and balanced, main effects clear of two-factor interactions"
    # every row, every climate bin: each room's chosen option = the base option of its cell + the deltas of the factors on
    from smart_building.g36 import rooms
    if feats:
        bad = n = 0
        for r in P["runs"]:
            cmap = r["map"]
            cuts = cmap["oat_cuts"]
            for spec, f in feats.items():
                area, wall = np.asarray(f["area"], float), np.asarray(f["wall"], float)
                cells = [gs.cell_of(a, w) for a, w in zip(area, wall)]
                for j, rep in enumerate(g4.bin_reps(cuts)):
                    got = rooms.option_index(r["model"]["room_policy"], {"area": area, "wall": wall}, f["rooms"], rep)
                    opts = r["model"]["room_policy"]["options"]
                    base_j = sum(rep > c for c in P["bmap"]["oat_cuts"])
                    for z, c in enumerate(cells):
                        want = dict(P["bmap"]["maps"][base_j].get(g4._key(c, 0), {}))
                        for k, fac in enumerate(ROOM):
                            if r["levels"][NW + k] > 0 and list(c) in fac["cells"]:
                                want.update(fac["delta"])
                        n += 1
                        bad += opts[got[z]] != want
        if bad:
            raise SystemExit(f"{bad} of {n} room choices differ from the design")
        out["room_choices_checked"] = n
    for r in P["runs"]:
        rc = r["model"]["rule_config"]
        for i, w in enumerate(WHOLE):
            keys = P["by"][w]["build"]["rule_config"]
            on = all(rc.get(k) == v for k, v in keys.items())
            if on != (r["levels"][i] > 0):
                raise SystemExit(f"{r['name']}: whole-plot level {w} on={on} but design {r['levels'][i]}")
    out["whole_plot_checked"] = True
    return out


# ---------------------------------------------------------------- effects
def find_cost(name, spec):
    for d in (OUT, TREE):
        p = d / "records" / R.key(name, spec)
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def estimate(P, feats, specs=None):
    """Per scenario: per-factor main effect from the 24 runs' nominal totals (None when no room of the scenario is in
    the factor's cells), the rule's cost (the weight), the episode mean outside air, the per-run relative costs."""
    per = {}
    X = np.array([[1.0] + list(r["levels"]) for r in P["runs"]])
    for spec in specs or P["specs"]:
        recs = [find_cost(r["name"], spec) for r in P["runs"]]
        if any(x is None for x in recs):
            continue
        base = recs[[i for i, r in enumerate(P["runs"]) if r["name"] == RULE][0]]
        norm = base["summary"]["cost"]
        y = np.array([x["summary"]["cost"] / norm - 1.0 for x in recs])
        cells = gs.room_cells(feats, spec)
        have = {tuple(c) for c in cells}
        pres = [True] * NW + [any(tuple(c) in have for c in fac["cells"]) for fac in ROOM]
        cols = [0] + [i + 1 for i, ok in enumerate(pres) if ok]
        beta, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
        eff = [None] * len(FACTORS)
        for j, c in enumerate(cols[1:], start=1):
            eff[c - 1] = 2.0 * float(beta[j])
        per[spec] = {"eff": eff, "oat": float(np.mean(base["hourly"]["outside_t"])), "cost": float(norm),
                     "y": y.tolist(), "resid": float(np.sqrt(np.mean((y - X[:, cols] @ beta) ** 2)))}
    return per


# ---------------------------------------------------------------- collect
def collect(a):
    P = plan()
    ctx = L.Ctx(OUT, tag="wu")
    import geo5_tree as T
    fpath = G / "tree2/unit_features.json"
    ufeat = json.loads(fpath.read_text(encoding="utf-8"))
    sample = {}
    for s in P["specs"]:
        b = s.split("/")[1]
        if b in ufeat and len(sample) < 24 and b not in {x.split("/")[1] for x in sample}:
            sample[s] = ufeat[b]
    info = checks(P, sample)
    print("checks:", json.dumps(info))
    missing = [s for s in P["specs"] if s.split("/")[1] not in ufeat]
    if missing:
        raise SystemExit(f"no room features for {len(missing)} scenarios")
    if a.cmd == "check" and not a.smoke_steps:
        return 0
    models = OUT / "models"
    for r in P["runs"]:
        O.write_model(r["model"], models / f"model-{r['name']}.json")
    if a.cmd == "check":
        pick = [r for r in P["runs"] if r["name"] in ("wur01", "wuf12")]
        pairs = [(ctx.rel(models / f"model-{r['name']}.json"), P["specs"][0]) for r in pick]
        R.record(ctx, "wu smoke", pairs, OUT, 2, max_steps=a.smoke_steps)
        print("smoke recorded:", [str(R.record_path(OUT, r["name"], P["specs"][0], a.smoke_steps)) for r in pick])
        return 0
    ctx.state.pop("done", None)
    ctx.state.update(rule=RULE, factors=[f["name"] for f in FACTORS], whole=WHOLE,
                     room=[{k: f[k] for k in ("name", "type", "cells", "delta")} for f in ROOM],
                     runs=[{"name": r["name"], "levels": r["levels"], "sha": r["sha"]} for r in P["runs"]],
                     specs=P["specs"], real=P["real"], checks=info, started=L.now())
    ctx.save()
    for s in P["specs"]:                      # the rule's own records (row 12) from the building-wide tree data
        for kind in ("records", "billing"):
            src, dst = TREE / kind / R.key(RULE, s), OUT / kind / R.key(RULE, s)
            if src.is_file() and not dst.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
    todo_runs = [r for r in P["runs"] if r["name"] != RULE]
    n = len(todo_runs)
    pairs = []
    for b in range(0, len(P["specs"]), n):     # Latin-square blocks: consecutive episodes differ in run and scenario
        block = P["specs"][b:b + n]
        for k in range(n * n):
            r, j = k % n, (k % n + k // n) % n
            if j < len(block):
                pairs.append((ctx.rel(models / f"model-{todo_runs[r]['name']}.json"), block[j]))
    if len(set(pairs)) != n * len(P["specs"]):
        raise SystemExit("the episode order lost or repeated a pair")
    ctx.log("EVENT", f"wu data: {len(FACTORS)} columns ({WHOLE} + {[f['name'] for f in ROOM]}), 24 runs (row 12 = "
                     f"{RULE}, reused) x {len(P['specs'])} scenarios = {len(pairs)} episodes, workers {a.workers}")
    try:
        R.record(ctx, "wu data", pairs, OUT, a.workers)
    except L.Wake as w:
        ctx.write_wake(w, "wu.collect", 1, False)
        ctx.log("WAKE", f"wu data: {w.reason}: {w.text} -> {ctx.rel(ctx.wake_path)}")
        ctx.save()
        return 3
    ctx.state["done"] = L.now()
    ctx.save()
    ctx.log("EVENT", "wu data: done")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("check", "collect"))
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--smoke-steps", type=int, default=0)
    a = p.parse_args()
    return collect(a)


if __name__ == "__main__":
    sys.exit(main())
