"""scenarios not used so far (not in the tree pool, not in the held-out 60):
- selection set (point 2): up to 5 per station-period (20 station-periods), buildings outside the held-out 60 first,
  then any; used to compare candidates in pairs before the fixed closed-loop sets;
- cold design set (point 3): Fairbanks January / March and Denver January scenarios not in the selection set,
  buildings outside the held-out 60 first.
Writes data/runs/b2rl/geo5/r3/sets.json; with --record also records the bases (gt_wub and gt_r5pa2 on the selection
set, gt_wub and gt_r5pa2 on the cold set) under data/runs/b2rl/geo5/r3.
python3 -B pipeline/launch/launch.py pipeline/setup/geo5_r3_sets.py [--record] [--workers 11]"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo5_lib as L
import geo5_record as R

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "data/runs/b2rl/geo5"
OUT = G / "r3"
SPLIT = ROOT / "data/processed/scenarios/split-v1-r1/train"
COLD = ("fairbanks_1", "fairbanks_3", "denver_1")
MODELS = {"gt_wub": "data/runs/b2rl/geo5/wu/models/model-gt_wub.json",
          "gt_r5pa2": "data/runs/b2rl/geo5/tree/models/model-gt_r5pa2.json"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", action="store_true")
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--per", type=int, default=3)
    p.add_argument("--cold", type=int, default=99, help="per cold station-period (all that are left)")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    allsc = sorted(f"train/{b.name}/{w.name}" for b in SPLIT.iterdir() for w in b.iterdir()
                   if (w / "config.gin").is_file())
    pool = set(json.loads((G / "tree/pipeline.json").read_text(encoding="utf-8"))["specs"])
    r60 = set(json.loads((ROOT / "data/runs/b2rl/geo/train_ceiling_rest60/gt_r5pa2.json").read_text())["rows"])
    r60_b = {s.split("/")[1] for s in r60}
    unused = [s for s in allsc if s not in pool and s not in r60]
    rng = np.random.default_rng(20261109)
    by_p = {}
    for s in unused:
        by_p.setdefault(s.split("/")[2], []).append(s)
    selection, cold = [], []
    for per_name in sorted(by_p):
        cand = list(rng.permutation(by_p[per_name]))
        clean = [s for s in cand if s.split("/")[1] not in r60_b]
        rest = [s for s in cand if s.split("/")[1] in r60_b]
        order = clean + rest
        pick = order[: a.per]
        selection += pick
        if per_name in COLD:
            left = [s for s in order if s not in pick]
            cold += left[: a.cold]
    sel_b = {s.split("/")[1] for s in selection}
    out = {"made": L.now(), "selection": selection, "cold": cold,
           "checks": {"unused": len(unused), "selection": len(selection), "cold": len(cold),
                      "selection_in_held_out_buildings": sum(s.split("/")[1] in r60_b for s in selection),
                      "cold_in_held_out_buildings": sum(s.split("/")[1] in r60_b for s in cold),
                      "overlap_selection_cold": len(set(selection) & set(cold)),
                      "overlap_with_pool_or_held_out": len((set(selection) | set(cold)) & (pool | r60)),
                      "selection_buildings": len(sel_b)}}
    (OUT / "sets.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["checks"]))
    from collections import Counter
    print("cold by period:", dict(Counter(s.split("/")[2] for s in cold)))
    if not a.record:
        return 0
    ctx = L.Ctx(OUT, tag="r3")
    pairs = [(MODELS[n], s) for n in ("gt_wub", "gt_r5pa2") for s in cold + selection]
    R.record(ctx, "r3 bases (selection + cold)", pairs, OUT, a.workers)
    ctx.log("EVENT", "r3 bases: done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
