"""New validation buildings from the organisers' generator (same procedure and checks as pipeline/setup/geo5_gen6.py), seeds
not used by any existing pool. Everything is written under <scratch>/val2; the workspace is only read.
Per building: 5 public stations (one weather preset each) and 5 unseen stations x {January, July}
(as pipeline/setup/new_stations_build.py: copy of the building's config with weather, start and convection changed).
python3 -B pipeline/launch/launch.py gen_val2.py <scratch> <n>"""
import copy, json, re, shutil, sys, time
from pathlib import Path

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
SCR = Path(sys.argv[1]) / "val2"; N = int(sys.argv[2])
UPSTREAM = ROOT / "reference/starter-kit"
POOL, LABEL = "val2", "split-v1/val2-paper-20261113"
PUBLIC, PRESETS = ("atlanta", "denver", "fairbanks", "phoenix", "singapore"), (1, 3, 5, 7)
NEW, MONTHS, CONVECTION = ("riyadh", "dubai", "seattle", "london", "beijing"), {"jan": "2023-01-05 06:00", "jul": "2023-07-05 06:00"}, 19.1
WNEW = ROOT / "data/processed/weather_new/csv"


def main():
    from smart_building.learned.b2rl_env import setup_simulator
    setup_simulator(ROOT)
    from smart_building.scenarios import factory
    from smart_buildings.smart_control.utils.scenario_generator import generate_scenario_from_config, load_scenario_from_parts
    used = set()
    for d in list((ROOT / "data/processed/scenarios").glob("*/*/building_*")) + list((ROOT / "data/processed/buildings").rglob("building_*.yaml")):
        m = re.match(r"building_(\d+)", d.name)
        if m: used.add(int(m.group(1)))
    for f in list((ROOT / "configs/splits").glob("*.json")) + list((ROOT / "data/runs/b2rl").glob("geo5/**/gen*_split.json")):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for pool in (j.get("pools") or {"_": j}).values():
            if isinstance(pool, dict):
                used |= set(pool.get("seeds") or []) | {x["seed"] for x in pool.get("rejected") or [] if isinstance(x, dict)}
    print("seeds already used:", len(used), flush=True)
    r1 = json.loads((ROOT / "configs/splits/v1-train-r1.json").read_text(encoding="utf-8"))
    split = copy.deepcopy(r1); split["version"] = "v1"
    strata = ["shipped", "default"] * (N // 2) + ["shipped"] * (N % 2)
    split["pools"] = {POOL: {"seed_range": [10000, 20000], "draw": {"rng_label": LABEL}, "strata": strata}}

    def check(seed, stratum, config):
        if seed in used:
            return "seed used by an existing pool"
        return factory.floor_plan_cells_problem(config, SCR / "plans" / f"{seed}-{stratum}")
    t0 = time.time()
    if len(sys.argv) > 3:  # release: the recorded draw (seeds, strata, rejected) of the paper's validation set
        rec = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
        assert rec["strata"] == strata, "recorded strata differ"
        seeds, rejected = rec["seeds"], rec["rejected"]
    else:
        seeds, rejected = factory.draw_seeds(split, POOL, strata, reporter=lambda m: print(m, flush=True), extra_check=check)
    assert not (set(seeds) & used)
    mats = factory.pool_materials(seeds)
    rows, bad = [], []
    for i, (seed, stratum, mat) in enumerate(zip(seeds, strata, mats)):
        config = factory.building_config(split, seed, stratum, mat)
        yaml = SCR / "buildings" / f"{config['name']}.yaml"; yaml.parent.mkdir(parents=True, exist_ok=True)
        factory.write_building_yaml(config, yaml)
        b = config["name"]; first = None
        for j, st in enumerate(PUBLIC):
            w = f"{st}_{PRESETS[(i + j) % 4]}"
            with factory.working_directory(UPSTREAM):
                cfg = load_scenario_from_parts(str(yaml), str(UPSTREAM / "smart_buildings/weather" / f"{w}.yaml"), output_base_dir=str(SCR / "seen"))
                folder = Path(generate_scenario_from_config(cfg)["output_dir"])
            if not folder.is_absolute(): folder = (UPSTREAM / folder).resolve()
            problem = factory.constructor_problem(folder, UPSTREAM)
            print(f"{b}/{w}: {problem or 'builds'} -> {folder}", flush=True)
            (bad if problem else rows).append({"tag": "seen", "building": b, "weather": w, "dir": folder.as_posix()})
            first = first or (folder, w)
        folder, w = first; text = (folder / "config.gin").read_text(encoding="utf-8")
        for st in NEW:
            for mon, start in MONTHS.items():
                d = SCR / "unseen" / b / f"{st}_{mon}"; d.mkdir(parents=True, exist_ok=True)
                g = text.replace(folder.as_posix(), d.as_posix()).replace(f"{b}/{w}", f"{b}/{st}_{mon}")
                for pat, rep in ((r"^convection_coefficient = .*$", f"convection_coefficient = {CONVECTION}"),
                                 (r"^weather_csv_path = .*$", f"weather_csv_path = '{(WNEW / (st + '.csv')).as_posix()}'"),
                                 (r"^start_timestamp = .*$", f"start_timestamp = '{start}'"),
                                 (r"^metrics_path = .*$", f'metrics_path = "{(d / "metrics").as_posix()}"')):
                    g, n = re.subn(pat, rep, g, flags=re.M); assert n == 1, (pat, n)
                (d / "config.gin").write_text(g, encoding="utf-8", newline="\n")
                for f in ("floor_plan.npy", "zone_map.npy"): shutil.copyfile(folder / f, d / f)
                assert d.as_posix() + "/floor_plan.npy" in g and d.as_posix() + "/zone_map.npy" in g, "npy paths not redirected"
                problem = factory.constructor_problem(d, UPSTREAM)
                (bad if problem else rows).append({"tag": "unseen", "building": b, "weather": f"{st}_{mon}", "dir": d.as_posix()})
                if problem: print(f"{b}/{st}_{mon}: {problem}", flush=True)
    (SCR / "manifest.json").write_text(json.dumps({"made": time.strftime("%Y-%m-%d %H:%M:%S"), "label": LABEL, "seeds": seeds, "strata": strata,
                                                   "rejected": rejected, "scenarios": rows, "failed": bad}, indent=1), encoding="utf-8")
    print(f"DONE {len(seeds)} buildings, {len(rows)} scenarios build, {len(bad)} failed, {time.time() - t0:.0f} s", flush=True)
    g = (Path(rows[0]["dir"]) / "config.gin").read_text().splitlines()
    print("\n".join(l[:200] for l in g if re.search(r"npy|weather_csv|start_timestamp|convection|metrics_path|label", l)))


if __name__ == "__main__":
    main()
