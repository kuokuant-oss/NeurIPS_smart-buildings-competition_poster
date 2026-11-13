"""Pipeline 5 stage 戊 hooks (2026-11-04, docs/plan/pipeline5-plan-20261104.md §二戊): scaffolding only, no training.

1. Data collection with two-stage randomisation: per AHU draw the share of its rooms that get a room option (0 / 50
   / 100 %), then which rooms (two_stage_assign). The rule stays the stage-丁 version: the room option is laid over
   the base room-policy tree ("geo5_overlay": the options are the base options and copies with the room deltas, a
   room's choice = the base tree's choice + K x its code), so the base tree keeps re-deciding hourly. The overlay
   is read by the recording worker's wrapper of rooms.option_index (geo5_record._patch_assign); a worker refuses
   an overlay episode in which the wrapper never ran. Local data only, never submitted.
2. Per-unit records (unit_rows): one row per (scenario, room) with the room, AHU, neighbour-exposure and building
   features of the plan (§二戊 2) and the outcomes (the room's comfort and energy share of the episode, the building
   total). Features the floor plan alone does not give (capacity margin, code minimum airflow density, the AHU's
   distance from the 20 m3/s limit) are listed in TODO_FEATURES and come as None until their source is wired.
3. Leave-building-out folds (lobo_folds).
4. Tree structures 甲 / 乙 / 丙 (TreeA, TreeB, TreeC): the interface (fit / decide / to_room_policy), no fitting yet.
"""
import copy
import os

import numpy as np

from geo5_lib import ROOT

FRACTIONS = (0.0, 0.5, 1.0)
CV_M = 0.2
TODO_FEATURES = {"capacity_margin": "VAV maximum airflow vs the room's design load (config.gin vav max_air_flow)",
                 "code_min_density": "ASHRAE 62.1 minimum airflow per m2 from G36Config (zone_min_* switches)",
                 "ahu_flow_margin": "20 m3/s AHU limit minus the AHU's summed VAV maxima"}


# ---------------------------------------------------------------- features
def unit_features(spec):
    """Per-room static features of a scenario (worker function: loads the environment once)."""
    os.environ["SB_FASTSIM"] = "rom"
    from smart_building.learned.b2rl_env import setup_simulator, spec_dir
    setup_simulator(ROOT)
    from smart_buildings import evaluate
    from smart_buildings.smart_control.simulator import building_utils, constants
    from smart_building.g36.building import BuildingIndex
    from smart_building.g36 import rooms as rooms_mod
    env = evaluate.load_hybrid_environment(spec_dir(spec) / "config.gin")
    env.metrics_path = None
    meta = evaluate.create_building_meta_data(env, spec_dir(spec))
    index = BuildingIndex(meta, env.action_spec())
    base = rooms_mod.room_features(meta, index)
    rooms, exterior_walls, _, _, _ = building_utils.construct_building_data_types(
        floor_plan=np.asarray(meta["floor_plan"]), zone_map=np.asarray(meta["zone_map"]))
    shell = exterior_walls == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
    n = index.n_zones
    room_id = np.full(shell.shape, -1, dtype=np.int64)
    for z, vav in enumerate(index.vav_ids):
        cells = np.asarray(rooms[vav.replace("VAV_", "room_")])
        room_id[cells[:, 0], cells[:, 1]] = z
    walls = {}
    for name, (dr, dc) in (("n", (-1, 0)), ("s", (1, 0)), ("w", (0, -1)), ("e", (0, 1))):
        nb = np.roll(shell, (-dr, -dc), axis=(0, 1))          # the cell next to a room cell in that direction
        walls[name] = np.bincount(room_id[(room_id >= 0) & nb], minlength=n).astype(float) * CV_M
    # shared walls: interior (non-room, non-shell) cells touching two rooms count CV_M for that pair
    shared = {}
    other = (room_id < 0) & ~shell
    H, W = shell.shape
    for r, c in zip(*np.nonzero(other)):
        near = set()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and room_id[rr, cc] >= 0:
                near.add(int(room_id[rr, cc]))
        for a in near:
            for b in near:
                if a < b:
                    shared[(a, b)] = shared.get((a, b), 0.0) + CV_M
    env.close()
    numbers = [int(v.split("_")[-1]) for v in index.vav_ids]
    group = np.asarray(index.zone_group).astype(int)
    area, wall = base["area"], base["wall"]
    out = {"spec": spec, "rooms": numbers, "ahu": group.tolist(), "area": area.tolist(), "wall": wall.tolist(),
           "depth": [float(a / w) if w > 0 else None for a, w in zip(area, wall)],
           **{f"wall_{k}": v.tolist() for k, v in walls.items()},
           "sides": base["sides"].tolist(), "core": base["core"].tolist(),
           "shared": [[numbers[a], numbers[b], round(v, 3)] for (a, b), v in sorted(shared.items())],
           "ahu_rooms": {int(g): int(np.sum(group == g)) for g in np.unique(group)},
           "ahu_exterior_share": {int(g): float(np.mean(wall[group == g] > 0)) for g in np.unique(group)},
           **{k: None for k in TODO_FEATURES}}
    return out


# ---------------------------------------------------------------- two-stage randomisation and the overlay
def two_stage_assign(feat, n_options, seed, fractions=FRACTIONS):
    """Per AHU and room option: a share of the AHU's rooms (drawn from `fractions`), then which rooms. Returns
    {"codes": {room: bit code over the room options}, "share": {option: {ahu: share}}}."""
    rng = np.random.default_rng(seed)
    rooms, ahu = np.asarray(feat["rooms"]), np.asarray(feat["ahu"])
    codes = np.zeros(len(rooms), dtype=int)
    share = {}
    for j in range(n_options):
        share[j] = {}
        for g in np.unique(ahu):
            idx = np.nonzero(ahu == g)[0]
            f = float(rng.choice(fractions))
            k = int(round(f * len(idx)))
            on = rng.choice(idx, size=k, replace=False) if k else np.array([], dtype=int)
            codes[on] |= 1 << j
            share[j][int(g)] = f
    return {"codes": {str(int(r)): int(c) for r, c in zip(rooms, codes) if c}, "share": share}


def overlay_model(model, room_deltas, assignment):
    """The model with its room policy widened by the room options: options = base options x every code (the deltas of
    the code's options added / set), "geo5_overlay" = {K, codes}. room_deltas: list of {"add": {...}, "set": {...}}."""
    m = copy.deepcopy(model)
    rp = m["room_policy"]
    base = rp["options"]
    K = len(base)
    options = []
    for code in range(2 ** len(room_deltas)):
        for o in base:
            o2 = dict(o)
            for j, d in enumerate(room_deltas):
                if code >> j & 1:
                    for key, add in d.get("add", {}).items():
                        o2[key] = round(float(o2.get(key, 0.0)) + float(add), 6)
                    o2.update(d.get("set", {}))
            options.append(o2)
    rp["options"] = options
    rp["geo5_overlay"] = {"K": K, "codes": assignment["codes"]}
    return m


def unit_rows(record, feat, assignment, n_options):
    """Per-room rows of one episode: features, own code, AHU shares, neighbour exposure, outcomes."""
    zones = record["zones"]
    hourly = record["hourly"]
    comfort = np.asarray(hourly.get("zone_comfort_1e8", []), dtype=float)
    energy = np.asarray(hourly.get("zone_energy_1e8", []), dtype=float)
    # dashboard records keep the per-zone hourly arrays as zones x hours; make them hours x zones
    if comfort.ndim == 2 and comfort.shape[0] == len(zones):
        comfort = comfort.T
    if energy.ndim == 2 and energy.shape[0] == len(zones):
        energy = energy.T
    total = record["summary"]["cost"]
    num_of = {z: int(str(z).split("_")[-1]) for z in zones}
    pos = {r: i for i, r in enumerate(feat["rooms"])}
    codes = {int(k): v for k, v in assignment["codes"].items()}
    nb = {}
    for a, b, length in feat["shared"]:
        nb.setdefault(a, []).append((b, length))
        nb.setdefault(b, []).append((a, length))
    rows = []
    for zi, z in enumerate(zones):
        r = num_of[z]
        i = pos.get(r)
        if i is None:
            continue
        g = feat["ahu"][i]
        same = [x for x, gg in zip(feat["rooms"], feat["ahu"]) if gg == g and x != r]
        row = {"spec": record["spec"], "building": record["spec"].split("/")[1], "room": r, "ahu": g,
               "code": codes.get(r, 0),
               **{k: feat[k][i] for k in ("area", "wall", "depth", "wall_n", "wall_s", "wall_e", "wall_w", "sides",
                                          "core")},
               "ahu_rooms": feat["ahu_rooms"][g] if isinstance(next(iter(feat["ahu_rooms"])), int)
               else feat["ahu_rooms"][str(g)],
               "building_cost": total,
               "comfort_1e8": float(comfort[:, zi].sum()) if comfort.ndim == 2 else None,
               "energy_1e8": float(energy[:, zi].sum()) if energy.ndim == 2 else None}
        for j in range(n_options):
            on_same = [x for x in same if codes.get(x, 0) >> j & 1]
            row[f"ahu_share_{j}"] = assignment["share"][j][g] if g in assignment["share"][j] else \
                assignment["share"][j].get(str(g))
            row[f"nb_ahu_on_{j}"] = len(on_same) / len(same) if same else 0.0
            row[f"nb_wall_on_{j}"] = float(sum(length for x, length in nb.get(r, []) if codes.get(x, 0) >> j & 1))
        rows.append(row)
    return rows


def lobo_folds(specs, k=5, seed=0):
    """Leave-building-out folds: building -> fold, every scenario of a building in the same fold."""
    buildings = sorted({s.split("/")[1] for s in specs})
    rng = np.random.default_rng(seed)
    fold = {b: i % k for i, b in enumerate(rng.permutation(buildings))}
    return {s: fold[s.split("/")[1]] for s in specs}


# ---------------------------------------------------------------- tree structures (interfaces only)
class TreeBase:
    """fit(units, folds) learns from unit rows; decide(rows) gives each room's combined action code; to_room_policy
    turns the fit into a room policy (a "tree" mode policy over the base options x codes). Depth by leave-building-out,
    leaf effects shrunk to the mean, building identity as a random effect, cost-weighted target (plan §二戊 4)."""
    name = "base"

    def fit(self, units, folds):
        raise NotImplementedError(f"tree {self.name}: training is not part of the scaffolding")

    def decide(self, rows):
        raise NotImplementedError

    def to_room_policy(self, base_policy):
        raise NotImplementedError


class TreeA(TreeBase):
    """甲 (main): one tree on room + AHU features, leaves give the combined action (code over the room options)."""
    name = "A"


class TreeB(TreeBase):
    """乙: an AHU tree, then a room tree given the AHU decision (measures the loss of splitting the decision)."""
    name = "B"


class TreeC(TreeBase):
    """丙: a smooth teacher (room term + AHU term + a few physical pairwise terms) distilled into a shallow tree."""
    name = "C"


TREES = {"A": TreeA, "B": TreeB, "C": TreeC}
