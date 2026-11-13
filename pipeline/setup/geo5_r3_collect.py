"""Stage 戊之三 data (2026-11-09; spec docs/plan/tree-round3-20261109.md §2-3).
- Base: gt_wub's rule (its four building-wide climate gates) with the stage base room map (gt_r5pa2's room policy, all
  room factors off) = "gt_r3b".
- Whole plot: the full 2^3 of three options that lost on average in the 11/04 screen but may win in some climates
  (the tree decides): O13 steady lag compensation (zone_temp_lead 0.5, steady, clip 0.3 K), o509_2 warm-up at full
  heating airflow, o508_2 pre-occupancy setback 2 K deeper. 8 rows x 2 room randomisations = 16 episodes / scenario.
- Sub plot (point 5, two-stage randomisation as geo5_tree.two_stage_assign, but each option only among its eligible
  rooms): per AHU a share 0 / 1/2 / 1 of the eligible rooms, then which: room_margin_reserve 1 (all rooms),
  room_stagger 1 (all rooms), heating airflow ratio 0.3 (small / medium rooms with exterior wall), 0.6 (large rooms
  with exterior wall), cooling target 0.4 K inside (large rooms without a long exterior wall).
- Scenarios: the TRAIN-pool scenarios of the tree pool (128) + the cold set of r3/sets.json (15) + the new gen6 cold
  buildings when they exist (point 3 and 6); the rule gt_r5pa2 is recorded where missing (the cost weight).
Models / assignments per episode under data/runs/b2rl/geo5/r3/{models,assign}; records in r3/records.
python3 -B pipeline/launch/launch.py pipeline/setup/geo5_r3_collect.py [--plan] [--workers 11] [--no-gen6]"""
import argparse
import copy
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo5_lib as L  # noqa: E402
import geo5_options as O  # noqa: E402
import geo5_record as R  # noqa: E402
import geo5_tree as T  # noqa: E402
import group_stage as gs  # noqa: E402
import group_stage4 as g4  # noqa: E402

G = ROOT / "data/runs/b2rl/geo5"
OUT = G / "r3"
WUB = G / "wu/models/model-gt_wub.json"
RULE = G / "tree/models/model-gt_r5pa2.json"
WHOLE = [("o13s", {"zone_temp_lead": 0.5, "zone_temp_lead_steady": True, "zone_temp_lead_max_k": 0.3}),
         ("o509_2", {"warmup_heat_max_ratio": 1.0}),
         ("o508_2", {"setback_heat_drop_k": 2.0})]
DELTAS = [{"name": "reserve", "set": {"room_margin_reserve": 1.0}, "cells": None},
          {"name": "stagger", "set": {"room_stagger": 1.0}, "cells": None},
          {"name": "ratio03_SMe", "set": {"room_heat_max_ratio": 0.3}, "cells": g4.cells(g4.SM, g4.EXT)},
          {"name": "ratio06_LXe", "set": {"room_heat_max_ratio": 0.6}, "cells": g4.cells(g4.LX, g4.EXT)},
          {"name": "cool04_LXn", "set": {"room_cool_margin_k": 0.4}, "cells": g4.cells(g4.LX, g4.NONLONG)}]
REPS = 2
FRACTIONS = (0.0, 0.5, 1.0)


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def assign(feat, seed):
    """Per AHU and option: a share of the option's eligible rooms, then which ones. Returns codes and shares."""
    rng = np.random.default_rng(seed)
    rooms, ahu = np.asarray(feat["rooms"]), np.asarray(feat["ahu"])
    cells = [tuple(gs.cell_of(a, w)) for a, w in zip(feat["area"], feat["wall"])]
    codes = np.zeros(len(rooms), dtype=int)
    share = {}
    for j, d in enumerate(DELTAS):
        ok = np.array([d["cells"] is None or list(c) in d["cells"] for c in cells])
        share[d["name"]] = {}
        for g in np.unique(ahu):
            idx = np.nonzero((ahu == g) & ok)[0]
            f = float(rng.choice(FRACTIONS))
            k = int(round(f * len(idx)))
            on = rng.choice(idx, size=k, replace=False) if k else np.array([], dtype=int)
            codes[on] |= 1 << j
            share[d["name"]][int(g)] = f
    return {"codes": {str(int(r)): int(c) for r, c in zip(rooms, codes) if c}, "share": share}


def specs_of(include_gen6, skip_fail=True):
    tree = json.loads((G / "tree/pipeline.json").read_text(encoding="utf-8"))
    sets = json.loads((OUT / "sets.json").read_text(encoding="utf-8"))
    out = [s for s in tree["specs"] if s.startswith("train/")] + list(sets["cold"])
    g6 = OUT / "gen6_specs.json"
    if include_gen6 and g6.is_file():
        out += [s for s in json.loads(g6.read_text(encoding="utf-8")) if s.startswith("gen6/")]
    fail = OUT / "rom_fail.json"                 # buildings whose fast-simulator reduced model cannot be built
    if skip_fail and fail.is_file():
        bad = set(json.loads(fail.read_text(encoding="utf-8")))
        out = [s for s in out if s.split("/")[1] not in bad]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--plan", action="store_true")
    p.add_argument("--no-gen6", action="store_true")
    a = p.parse_args()
    wub, rule = json.loads(WUB.read_text(encoding="utf-8")), json.loads(RULE.read_text(encoding="utf-8"))
    base = copy.deepcopy(wub)
    base["room_policy"] = copy.deepcopy(rule["room_policy"])
    for _, cfg in WHOLE:
        clash = set(cfg) & set(base["rule_config"])
        if clash:
            raise SystemExit(f"{clash} already in gt_wub's rule_config")
    rows = [list(bits) for bits in itertools.product((0, 1), repeat=len(WHOLE))]
    specs = specs_of(not a.no_gen6)
    print(f"whole plot {[w for w, _ in WHOLE]}: {len(rows)} rows x {REPS} room randomisations; room options "
          f"{[d['name'] for d in DELTAS]}; {len(specs)} scenarios = {len(specs) * len(rows) * REPS} episodes")
    if a.plan:
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = L.Ctx(OUT, tag="r3")
    ctx.state.update(base="gt_wub + gt_r5pa2 room map", whole=[w for w, _ in WHOLE], rows=rows, deltas=DELTAS,
                     reps=REPS, specs=specs, started=ctx.state.get("started") or L.now())
    ctx.state.pop("done", None)
    ctx.save()
    import geo5_tree2_collect as T2
    fpath = G / "tree2/unit_features.json"
    feats = L.read_json(fpath) or {}
    first = {}
    for s in specs:
        if s.split("/")[1] not in feats:
            first.setdefault(s.split("/")[1], s)
    if first:
        ctx.log("EVENT", f"r3: room features of {len(first)} new buildings")
        with mp.get_context("spawn").Pool(min(8, len(first)), maxtasksperchild=1) as pool:
            for s, f, err in pool.imap_unordered(T2.feature_job, list(first.values())):
                if err:
                    ctx.write_wake(L.Wake("crash", f"room features {s}: {err}"), "r3.collect", 1, False)
                    return 3
                feats[s.split("/")[1]] = f
        L.atomic_json(fpath, feats)
    models, adir = OUT / "models", OUT / "assign"
    adir.mkdir(parents=True, exist_ok=True)
    whole_models = []
    for r, bits in enumerate(rows):
        m = copy.deepcopy(base)
        for (w, cfg), b in zip(WHOLE, bits):
            if b:
                m["rule_config"].update(cfg)
        whole_models.append(m)
    jobs = []
    for i, s in enumerate(specs):
        feat = feats[s.split("/")[1]]
        for r, bits in enumerate(rows):
            for k in range(REPS):
                name = f"r3r{r}k{k}s{i:03d}"
                path, apath = models / f"model-{name}.json", adir / f"{name}.json"
                if not path.is_file() or not apath.is_file():
                    asg = assign(feat, seed_of(s, r, k))
                    m = T.overlay_model(whole_models[r], [{"set": d["set"]} for d in DELTAS], asg)
                    m["source"]["geo5_r3"] = {"spec": s, "row": r, "bits": bits, "rep": k}
                    O.write_model(m, path)
                    L.atomic_json(apath, {"spec": s, "row": r, "bits": bits, "rep": k, **asg})
                jobs.append((i, r, k, ctx.rel(path), s))
    n = len(rows) * REPS
    jobs.sort(key=lambda j: (j[0] // n, (j[1] * REPS + j[2] + j[0]) % n, j[0]))
    pairs = [(ctx.rel(RULE), s) for s in specs
             if not (G / "tree/records" / R.key("gt_r5pa2", s)).is_file()] + [(j[3], j[4]) for j in jobs]
    ctx.log("EVENT", f"r3 data: {len(pairs)} episodes ({len(specs)} scenarios x {len(rows)} rows x {REPS} room "
                     f"randomisations + the rule where missing), workers {a.workers}")
    try:
        R.record(ctx, "r3 data", pairs, OUT, a.workers)
    except L.Wake as w:
        ctx.write_wake(w, "r3.collect", 1, False)
        ctx.log("WAKE", f"r3 data: {w.reason}: {w.text}")
        ctx.save()
        return 3
    ctx.state["done"] = L.now()
    ctx.save()
    ctx.log("EVENT", "r3 data: done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
