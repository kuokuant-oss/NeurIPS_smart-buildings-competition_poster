"""戊之三 (2026-11-09, user points 3 and 6): new generated buildings made exactly like geo5_gen_buildings.py
(official generator, same checks), new seeds outside TRAIN, TRAIN r1 and gen5, cold weathers only (Fairbanks
January / March, Denver January); pool gen6 (spec "gen6/<building>/<weather>"), record
data/runs/b2rl/geo5/r3/gen6_split.json, specs gen6_specs.json.
python3 -B pipeline/launch/launch.py pipeline/setup/geo5_gen6.py --n 30
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
OUT = ROOT / "data/runs/b2rl/geo5/r3"
GEN5 = ROOT / "data/runs/b2rl/geo5/tree/gen5_split.json"
UPSTREAM = ROOT / "reference/starter-kit"
POOL, LABEL = "gen6", "split-v1/gen6-r3"
COLD_COMBOS = ["fairbanks_1", "fairbanks_3", "denver_1"]
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
    g5 = json.loads(GEN5.read_text(encoding="utf-8"))
    used |= set(g5["seeds"]) | {x["seed"] for x in g5["rejected"]}   # 戊之三: not gen5 either
    split = copy.deepcopy(r1)
    split["version"] = "v1"                                  # folders split-v1/gen5/...
    strata = ["shipped", "default"] * (a.n // 2) + ["shipped"] * (a.n % 2)
    split["pools"] = {POOL: {"seed_range": [10000, 20000], "draw": {"rng_label": LABEL}, "strata": strata}}
    scratch = ROOT / ".tmp/gen6-r3"

    def check(seed, stratum, config):
        if seed in used:
            return "seed used by an existing TRAIN draw"
        return factory.floor_plan_cells_problem(config, scratch / "plans" / f"{seed}-{stratum}")
    t0 = time.time()
    seeds, rejected = factory.draw_seeds(split, POOL, strata, reporter=lambda m: print(m, flush=True),
                                         extra_check=check)
    split["pools"][POOL].update(seeds=seeds, rejected=rejected)
    combos = list(COLD_COMBOS)                                # 戊之三: cold weathers only (point 3)
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
    tree = json.loads((ROOT / "data/runs/b2rl/geo5/tree/pipeline.json").read_text(encoding="utf-8"))
    used_b = {s.split("/")[1] for s in tree["specs"]}
    spare = {}
    for r in r1["scenarios"]["train"]:
        b = r["building"]
        if b not in closed_b and b not in used_b and b not in spare and r["weather"].split("_")[-1] in "1357":
            spare[b] = f"train/{b}/{r['weather']}"
    record = {"made": time.strftime("%Y-%m-%d %H:%M:%S"), "script": "pipeline/setup/geo5_gen6.py",
              "split_base": "configs/splits/v1-train-r1.json", "pool": POOL, "draw_label": LABEL, "seeds": seeds,
              "rejected": rejected, "scenarios": [t[:2] for t in triples], "constructor_failed": bad, "files": files,
              "seconds": round(time.time() - t0)}
    (OUT / "gen6_split.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    (OUT / "gen6_specs.json").write_text(json.dumps(specs + sorted(spare.values()), indent=1) + "\n",
                                         encoding="utf-8")
    print(f"{len(specs)} generated scenarios build ({len(bad)} failed the constructor) + {len(spare)} unused TRAIN r1 "
          f"buildings, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
