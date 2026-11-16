"""Exact compaction of a model's composed room-option tree (2026-11-10: gt_r3d1's tree had 9,504 leaves / 776 KB,
its submission notebook 1.33 MB was refused by the Kaggle kernel push with 400 Bad Request; gt_r3a1's was 0.45 MB).
The composition grafts every option's tree under every leaf of the previous one, so most splits repeat a condition
already decided by an ancestor. Simplification (decision-preserving by construction; rooms.tree_choice: x <= t -> left):
  1. a split whose outcome is implied by the ancestors' interval on its feature is replaced by the reachable child;
  2. a split whose two (simplified) children are identical is replaced by that child.
Verification: the original and the compact tree give the same option for every room of every building in
tree2/unit_features.json (run-time feature names) under a 0.25 C grid of the 24 h outside air from -40 to 45 C, and for
200,000 random points drawn around every threshold. The original model is kept as <model>.before-compact.json.
python3 -B pipeline/setup/tree_compact.py data/runs/b2rl/search/models/model-gt_r3d1.json [--dry]"""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
from smart_building.g36 import rooms


def canon(node):
    return json.dumps(node, sort_keys=True, separators=(",", ":"))


def simplify(node, bounds):
    if "leaf" in node:
        return node
    f, t = node["feature"], node["threshold"]
    lo, hi = bounds.get(f, (-math.inf, math.inf))
    if hi <= t:
        return simplify(node["left"], bounds)
    if lo >= t:
        return simplify(node["right"], bounds)
    left = simplify(node["left"], {**bounds, f: (lo, t)})
    right = simplify(node["right"], {**bounds, f: (t, hi)})
    if canon(left) == canon(right):
        return left
    return {"feature": f, "threshold": t, "left": left, "right": right}


def stats(node):
    if "leaf" in node:
        return 1, 0
    a, b = stats(node["left"]), stats(node["right"])
    return a[0] + b[0], 1 + max(a[1], b[1])


def thresholds(node, out):
    if "leaf" not in node:
        out.setdefault(node["feature"], set()).add(node["threshold"])
        thresholds(node["left"], out)
        thresholds(node["right"], out)
    return out


def verify(orig, comp):
    feats = json.loads((ROOT / "data/runs/b2rl/geo5/tree2/unit_features.json").read_text(encoding="utf-8"))
    used = thresholds(orig, {})
    n = bad = 0
    sys.path.insert(0, str(ROOT / "pipeline/setup"))
    import geo5_tk_room as RM
    for b, fb in feats.items():
        rr = RM.room_rows(fb)
        cols = {k: np.asarray(rr[k], float) for k in used if k in rr}
        if not cols:
            continue
        z = len(next(iter(cols.values())))
        for oat in np.arange(-40.0, 45.01, 0.25):
            for i in range(z):
                x = {k: float(v[i]) for k, v in cols.items()}
                x.update(oat=float(oat), block=0.0, period=0.0)
                for k in used:
                    x.setdefault(k, 0.0)
                n += 1
                bad += rooms.tree_choice(orig, x) != rooms.tree_choice(comp, x)
    rng = np.random.default_rng(0)
    names = sorted(used)
    grid = {k: np.array(sorted(used[k])) for k in names}
    for _ in range(200000):
        x = {"block": 0.0, "period": 0.0}
        for k in names:
            t = grid[k][rng.integers(len(grid[k]))]
            x[k] = float(t + rng.choice([-1.0, 0.0, 1.0]) * rng.exponential(abs(t) * 0.05 + 0.05))
        n += 1
        bad += rooms.tree_choice(orig, x) != rooms.tree_choice(comp, x)
    return n, bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    path = (ROOT / a.model) if not Path(a.model).is_absolute() else Path(a.model)
    model = json.loads(path.read_text(encoding="utf-8"))
    orig = model["room_policy"]["tree"]
    comp = simplify(orig, {})
    (l0, d0), (l1, d1) = stats(orig), stats(comp)
    print(f"original: {l0} leaves, depth {d0}, {len(canon(orig))} bytes; compact: {l1} leaves, depth {d1}, "
          f"{len(canon(comp))} bytes")
    n, bad = verify(orig, comp)
    print(f"verification: {bad} differing decisions in {n} points")
    if bad:
        return 1
    if a.dry:
        return 0
    backup = path.with_name(path.stem + ".before-compact.json")
    if not backup.exists():
        shutil.copy2(path, backup)
    model["room_policy"]["tree"] = comp
    model["room_policy"]["compacted"] = {"from_leaves": l0, "leaves": l1, "verified_points": n,
                                         "tool": "pipeline/setup/tree_compact.py"}
    path.write_text(json.dumps(model), encoding="utf-8")
    print(f"written {path.relative_to(ROOT)} ({path.stat().st_size} bytes; original kept as {backup.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
