"""Scenario factory (build plan WP1 1.2; D-E03-D-E06, D-E12).

Buildings and scenarios are made with the official generator only
(smart_buildings.run_floor_generator / smart_control.utils.config_generator,
materials, scenario_generator), into data/processed/, never into reference/ or
data/raw/. The working directory during generation is the starter kit root
locally, or the remote overlay root (smart_building.remote.scenarios), which
mirror each other; so the paths written into config.gin ("../../data/processed/
..." for the floor plan, "smart_buildings/weather/csv/..." for the weather) and
the files' SHA-256 are the same in both places.

Purpose guards (validation-v1 §4 item 2, notes/07 P1.5, D-E06, D-E07):
"train" and "tune" refuse VAL and LOCK buildings, minneapolis and the held-out
station of their fold; "validate" refuses LOCK and minneapolis; "lockbox" is only
for reviewer-built jobs.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random

import yaml

PROCESSED_FROM_ROOT = Path("../../data/processed")  # relative to the starter kit / overlay root
PURPOSES = ("train", "tune", "validate", "lockbox")


def station_of(weather):
    return weather.rsplit("_", 1)[0]


def split_path(code_root, version):
    return Path(code_root) / "configs/splits" / f"{version}.json"


def load_split(code_root, version):
    return json.loads(split_path(code_root, version).read_text(encoding="utf-8"))


def pool_relative_dirs(version, pool):
    """(building YAML folder, scenario folder), relative to data/processed."""
    return Path("buildings") / f"split-{version}" / pool, Path("scenarios") / f"split-{version}" / pool


def local_scenarios_dir(version, pool):
    """--scenarios-dir for scripts/batch_evaluate.py (relative to reference/starter-kit)."""
    return (PROCESSED_FROM_ROOT / pool_relative_dirs(version, pool)[1]).as_posix()


def generated_scenario_dir(root, version, pool, building, weather):
    return Path(root) / PROCESSED_FROM_ROOT / pool_relative_dirs(version, pool)[1] / building / weather


# ------------------------------------------------------------------ guards

def fold_entry(split, fold):
    for entry in split["folds"]:
        if entry["fold"] == fold:
            return entry
    raise KeyError(f"no fold {fold}")


def check_request(split, purpose, pool, weather, fold=None, building=None):
    """Raise if this purpose may not use this pool, station or (fold, building)."""
    if purpose not in PURPOSES:
        raise ValueError(f"unknown purpose {purpose!r}")
    station = station_of(weather)
    lockbox_stations = set(split["stations"]["lockbox"])
    if purpose != "lockbox":
        if pool == "lock":
            raise PermissionError("LOCK buildings are only for reviewer-built lockbox jobs (D-E07)")
        if station in lockbox_stations:
            raise PermissionError(f"{station} is the lockbox station (D-E07, notes/07 P1.4)")
    if purpose in ("train", "tune"):
        if pool == "val":
            raise PermissionError("VAL buildings may not be used for training or tuning (D-E03, notes/07 P1.5)")
        if fold is not None and station == fold_entry(split, fold)["held_out_station"]:
            raise PermissionError(f"fold {fold} may not train or tune on its held-out station {station} (D-E06)")
    if purpose == "validate":
        if pool != "val":
            raise PermissionError("validation uses the VAL pool")
        if fold is not None:
            entry = fold_entry(split, fold)
            if weather not in entry["validation_windows"]:
                raise PermissionError(f"{weather} is not a validation window of fold {fold} (D-E06)")
            if building is not None and building not in group_buildings(split, entry["val_group"]):
                raise PermissionError(f"{building} is not in VAL group {entry['val_group']} (D-E06)")
    if purpose == "lockbox" and pool != "lock":
        raise PermissionError("lockbox jobs use the LOCK pool")


def group_buildings(split, group):
    pool = split["pools"]["val"]
    size = pool["group_size"]
    return [f"building_{seed:04d}" for seed in pool["seeds"][group * size:(group + 1) * size]]


# ------------------------------------------------------------------ official generator

def stratum_ranges(split, stratum):
    """DEFAULT_RANGES with the stratum's overrides (D-E04, D-E05)."""
    from smart_buildings.run_floor_generator import DEFAULT_RANGES
    ranges = dict(DEFAULT_RANGES)
    for key, (low, high) in split["size_strata"][stratum]["overrides"].items():
        ranges[key] = (low, high)  # tuples: config_generator._sample draws ints with randint
    return ranges


def building_config(split, seed, stratum, materials):
    """run_floor_generator.build_config with the stratum's ranges."""
    from smart_buildings.smart_control.utils.config_generator import generate_config
    config = generate_config(name=split["generator"]["name_format"].format(seed=seed), seed=seed,
                             building_shape=split["generator"]["building_shape"],
                             **stratum_ranges(split, stratum))
    if materials is not None:
        config["materials"] = materials
    return config


def pool_materials(seeds):
    """Materials stratified across the whole pool: the official run_floor_generator.py:229-241 formula (D-E03)."""
    from smart_buildings.smart_control.utils.materials import draw_materials
    digest = hashlib.sha256(repr(list(seeds)).encode()).digest()
    return draw_materials(random.Random(int.from_bytes(digest[:8], "big")), len(seeds))


def floor_plan_problem(config):
    """None if the official zone-area check passes for this building, else its message.

    Same steps as scenario_generator.generate_scenario_from_config up to the
    check (notes/01 §E: some seeds produce sliver zones and must be replaced).
    """
    import numpy as np
    from smart_buildings.smart_control.utils import scenario_generator as sg
    scenario = sg.scenario_config_from_dict(config)
    with sg._floor_plan_generation_lock:
        random.seed(scenario.seed)
        np.random.seed(scenario.seed)
        generator = sg._create_floor_plan_generator(scenario.floor_plan)
        generator.generate()
        try:
            sg._check_generated_zone_areas(generator, sg.CONTROL_VOLUME_CM, scenario.seed)
        except ValueError as error:
            return str(error)
    return None


def strip_cells(plan):
    """REQ-020 cross-check: the control volumes the simulator's classify_cv rejects
    (finite_difference_solver.py:175-260): a non-exterior cell (floor-plan value != 2,
    EXTERIOR_SPACE_VALUE_IN_FILE_INPUT) whose only two non-exterior 4-neighbours are on opposite
    sides, i.e. a one-cell-thick strip. Returns the (i, j) array of such cells."""
    import numpy as np
    inside = np.asarray(plan) != 2
    pad = np.pad(inside, 1, constant_values=False)
    up, down, left, right = pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]
    strip = (up & down & ~left & ~right) | (left & right & ~up & ~down)
    return np.argwhere(inside & strip)


def floor_plan_cells_problem(config, scratch):
    """REQ-020 cross-check for a draw candidate: the generator's floor plan as the simulator reads it
    (save_for_simulator into `scratch`, the steps of floor_plan_problem), scanned with strip_cells."""
    import numpy as np
    from smart_buildings.smart_control.utils import scenario_generator as sg
    scenario = sg.scenario_config_from_dict(config)
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    with sg._floor_plan_generation_lock:
        random.seed(scenario.seed)
        np.random.seed(scenario.seed)
        generator = sg._create_floor_plan_generator(scenario.floor_plan)
        generator.generate()
        generator.save_for_simulator(str(scratch / "floor_plan.npy"), str(scratch / "zone_map.npy"))
    cells = strip_cells(np.load(scratch / "floor_plan.npy"))
    if len(cells):
        return (f"REQ-020: {len(cells)} control volumes in one-cell-thick strips (first {tuple(int(x) for x in cells[0])}); "
                "the simulator constructor raises in classify_cv")
    return None


def constructor_problem(scenario_dir, root):
    """REQ-020 authority: None if the official simulator constructor builds the scenario's building
    (smart_buildings.evaluate.load_hybrid_environment on its config.gin, working directory = root, as the
    gateway and learned/rollouts.py load it), else the error it raised."""
    import gc
    from smart_buildings import evaluate
    with working_directory(root):
        try:
            env = evaluate.load_hybrid_environment(Path(scenario_dir) / "config.gin")
            del env
        except Exception as error:  # noqa: BLE001 - recorded, the draw decides
            message = " | ".join(str(a) for a in error.args) or str(error)
            return f"{type(error).__name__}: {message.splitlines()[0] if message else ''}"
        finally:
            gc.collect()
    return None


def draw_rng(label):
    return random.Random(int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big"))


def draw_seeds(split, pool_name, strata, reporter=None, extra_check=None):
    """D-E03: candidates in [low, high) shuffled by a fixed RNG; skip seeds failing the area check
    (and, when given, extra_check(seed, stratum, config) -> problem or None: REQ-020 for TRAIN r1)."""
    pool = split["pools"][pool_name]
    low, high = pool["seed_range"]
    candidates = list(range(low, high))
    draw_rng(pool["draw"]["rng_label"]).shuffle(candidates)
    chosen, rejected = [], []
    stream = iter(candidates)
    for stratum in strata:
        for seed in stream:
            config = building_config(split, seed, stratum, None)
            problem = floor_plan_problem(config)
            if problem is None and extra_check is not None:
                problem = extra_check(seed, stratum, config)
            if problem is None:
                chosen.append(seed)
                if reporter:
                    reporter(f"{pool_name}: seed {seed} ({stratum}) accepted")
                break
            rejected.append({"seed": seed, "stratum": stratum, "reason": problem.split(":")[0]})
            if reporter:
                reporter(f"{pool_name}: seed {seed} ({stratum}) rejected")
    return chosen, rejected


@contextlib.contextmanager
def working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def write_building_yaml(config, path):
    """run_floor_generator.save_config's yaml.dump call, written with LF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        yaml.dump(config, stream, default_flow_style=False, sort_keys=False)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def materialize_pool(split, pool_name, root, scenarios, purpose):
    """Write the pool's building YAMLs and the requested scenarios below root/../../data/processed.

    root: starter kit root or overlay root (it is the working directory while
    generating). scenarios: (building, weather, fold) triples. Returns
    {path relative to data/processed: sha256} for everything written.
    """
    from smart_buildings.smart_control.utils.scenario_generator import (
        generate_scenario_from_config, load_scenario_from_parts)
    root = Path(root)
    processed = (root / PROCESSED_FROM_ROOT).resolve()
    if processed.name != "processed" or processed.parent.name != "data":
        raise PermissionError(f"generated files go to data/processed only, not {processed}")
    pool = split["pools"][pool_name]
    for building, weather, fold in scenarios:
        check_request(split, purpose, pool_name, weather, fold=fold, building=building)
    seeds, strata = pool["seeds"], pool["strata"]
    materials = pool_materials(seeds)
    yaml_dir, scenario_dir = pool_relative_dirs(split["version"], pool_name)
    written = {}
    yaml_paths = {}
    for seed, stratum, mats in zip(seeds, strata, materials):
        config = building_config(split, seed, stratum, mats)
        path = processed / yaml_dir / f"{config['name']}.yaml"
        write_building_yaml(config, path)
        yaml_paths[config["name"]] = path
        written[(yaml_dir / path.name).as_posix()] = sha256_file(path)
    weather_dir = root / "smart_buildings/weather"
    output_base = (PROCESSED_FROM_ROOT / scenario_dir).as_posix()
    with working_directory(root):
        for building, weather, _ in scenarios:
            config = load_scenario_from_parts(str(yaml_paths[building]), str(weather_dir / f"{weather}.yaml"),
                                              output_base_dir=output_base)
            result = generate_scenario_from_config(config)
            folder = Path(result["output_dir"])
            for name in ("config.gin", "config.yaml", "floor_plan.npy", "zone_map.npy"):
                written[(scenario_dir / building / weather / name).as_posix()] = sha256_file(folder / name)
    return dict(sorted(written.items()))


def materialize_for_tasks(code_root, root, tasks):
    """Remote step 3 (remote plan §3.5): regenerate what the tasks need and check registered SHA-256."""
    report = []
    groups = {}
    for task in tasks:
        spec = task["scenario"]
        key = (spec["split"], spec["pool"], task.get("purpose", "validate"))
        groups.setdefault(key, set()).add((spec["building"], spec["weather"], spec.get("fold")))
    for (version, pool_name, purpose), needed in sorted(groups.items()):
        split = load_split(code_root, version)
        written = materialize_pool(split, pool_name, root, sorted(needed, key=lambda x: (x[0], x[1])), purpose)
        registered = split["files"][pool_name]
        mismatched = sorted(name for name, digest in written.items() if registered.get(name) != digest)
        report.append({"split": version, "pool": pool_name, "purpose": purpose, "files": len(written),
                       "mismatched": mismatched})
        if mismatched:
            raise RuntimeError(f"regenerated {pool_name} differs from the registered SHA-256 in "
                               f"{len(mismatched)} files, e.g. {mismatched[:3]} (remote plan §3.5: use the "
                               "private Dataset fallback)")
    return report
