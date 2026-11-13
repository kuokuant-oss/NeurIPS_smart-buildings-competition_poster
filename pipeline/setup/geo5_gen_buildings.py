"""Pipeline 5 stage 戊, plan §二戊 6 (2026-11-06; user: 「不是有建築生成器嗎?」「甚至之前是因為時間問題少生」): more TRAIN
buildings for the tree, made exactly like split-v1-r1's (official generator through smart_building.scenarios.factory:
the same size strata alternating shipped / default, the official zone-area check, REQ-020's one-cell-strip check on
the draw and the simulator constructor as the authority), from new seeds of the TRAIN seed range 10000-19999 that no
existing TRAIN draw used (v1-train, v1-train-r1 seeds and rejects), with their own draw label. One scenario per
building, the training stations x presets 1 / 3 / 5 / 7 balanced. Written to data/processed/scenarios/split-v1/gen5/
(spec "gen5/<building>/<weather>", spec_dir's non-train pool path), the record to data/runs/b2rl/geo5/tree/gen5_split.json,
the spec list (+ one scenario of each TRAIN r1 building still unused and outside the closed-loop sets) to
data/runs/b2rl/geo5/tree/gen5_specs.json.

python3 -B pipeline/launch/launch.py pipeline/setup/geo5_gen_buildings.py [--n 60]
"""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
OUT = ROOT / "data/runs/b2rl/geo5/tree"
UPSTREAM = ROOT / "reference/starter-kit"
POOL, LABEL = "gen5", "split-v1/gen5-tree"
STATIONS, PRESETS = ("atlanta", "denver", "fairbanks", "phoenix", "singapore"), (1, 3, 5, 7)
CLOSED = ("train_ceiling_rest60", "inner", "s4_first")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60)
    a = p.parse_args()
    from smart_building.learned.b2rl_env import setup_simulator
    setup_simulator(ROOT)                                    # the official smart_buildings package on the path
    from smart_building.scenarios import factory
    r1 = json.loads((ROOT / "configs/splits/v1-train-r1.json").read_text(encoding="utf-8"))
    old = json.loads((ROOT / "configs/splits/v1-train.json").read_text(encoding="utf-8"))
    used = set(r1["pools"]["train"]["seeds"]) | {x["seed"] for x in r1["pools"]["train"]["rejected"]} \
        | set(old["pools"]["train"]["seeds"]) | {x["seed"] for x in old["pools"]["train"].get("rejected", [])}
    split = copy.deepcopy(r1)
    split["version"] = "v1"                                  # folders split-v1/gen5/...
    strata = ["shipped", "default"] * (a.n // 2) + ["shipped"] * (a.n % 2)
    split["pools"] = {POOL: {"seed_range": [10000, 20000], "draw": {"rng_label": LABEL}, "strata": strata}}
    scratch = ROOT / ".tmp/gen5-tree"

    def check(seed, stratum, config):
        if seed in used:
            return "seed used by an existing TRAIN draw"
        return factory.floor_plan_cells_problem(config, scratch / "plans" / f"{seed}-{stratum}")
    t0 = time.time()
    seeds, rejected = factory.draw_seeds(split, POOL, strata, reporter=lambda m: print(m, flush=True),
                                         extra_check=check)
    split["pools"][POOL].update(seeds=seeds, rejected=rejected)
    combos = [f"{s}_{k}" for s in STATIONS for k in PRESETS]
    random.Random(LABEL).shuffle(combos)
    triples = [(f"building_{seed:04d}", combos[i % len(combos)], None) for i, seed in enumerate(seeds)]
    print(f"generating {len(triples)} scenarios ...", flush=True)
    files = factory.materialize_pool(split, POOL, UPSTREAM, triples, "train")
    specs, bad = [], []
    for building, weather, _ in triples:
        folder = factory.generated_scenario_dir(UPSTREAM, "v1", POOL, building, weather)
        problem = factory.constructor_problem(folder, UPSTREAM)
        print(f"constructor {building}/{weather}: {problem or 'builds'}", flush=True)
        (bad if problem else specs).append(f"{POOL}/{building}/{weather}")
    sets = json.loads((ROOT / "configs/b2rl/eval-sets.json").read_text(encoding="utf-8"))
    closed_b = {s.split("/")[1] for k in CLOSED for s in sets[k]}
    tree = json.loads((OUT / "pipeline.json").read_text(encoding="utf-8"))
    used_b = {s.split("/")[1] for s in tree["specs"]}
    spare = {}
    for r in r1["scenarios"]["train"]:
        b = r["building"]
        if b not in closed_b and b not in used_b and b not in spare and r["weather"].split("_")[-1] in "1357":
            spare[b] = f"train/{b}/{r['weather']}"
    record = {"made": time.strftime("%Y-%m-%d %H:%M:%S"), "script": "pipeline/setup/geo5_gen_buildings.py",
              "split_base": "configs/splits/v1-train-r1.json", "pool": POOL, "draw_label": LABEL, "seeds": seeds,
              "rejected": rejected, "scenarios": [t[:2] for t in triples], "constructor_failed": bad, "files": files,
              "seconds": round(time.time() - t0)}
    (OUT / "gen5_split.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    (OUT / "gen5_specs.json").write_text(json.dumps(specs + sorted(spare.values()), indent=1) + "\n",
                                         encoding="utf-8")
    print(f"{len(specs)} generated scenarios build ({len(bad)} failed the constructor) + {len(spare)} unused TRAIN r1 "
          f"buildings, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
