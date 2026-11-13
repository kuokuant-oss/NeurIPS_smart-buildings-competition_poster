"""Per-room geometry for the rule (2026-10-31, docs/plan/geo-room-stages.md): each zone's floor area and exterior wall
length, from the metadata floor plan and zone map, built with the official building_utils exactly as the simulator
builds its rooms (smart_control/simulator/building.py) and as learned/tokens.static_features does (its area and
exterior-exposure columns; kept here so that the rule and its submission notebook need no learned/ module).

Exterior wall length = the number of room cells 4-adjacent to the exterior wall shell x the control volume size
(0.2 m, config.gin control_volume_cm = 20)."""
import numpy as np

CV_M = 0.2


def room_quantities(metadata, index):
    """(area m2, exterior wall length m) per zone, in the order of index.vav_ids."""
    from smart_buildings.smart_control.simulator import building_utils, constants
    rooms, exterior_walls, _, _, _ = building_utils.construct_building_data_types(
        floor_plan=np.asarray(metadata["floor_plan"]), zone_map=np.asarray(metadata["zone_map"]))
    room_id = np.full(np.shape(exterior_walls), -1, dtype=np.int64)
    for z, vav in enumerate(index.vav_ids):
        cells = np.asarray(rooms[vav.replace("VAV_", "room_")])
        room_id[cells[:, 0], cells[:, 1]] = z
    n = index.n_zones
    area = np.bincount(room_id[room_id >= 0], minlength=n) * CV_M * CV_M
    shell = exterior_walls == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
    exposed = np.zeros(room_id.shape, dtype=bool)
    for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        exposed |= np.roll(shell, shift, axis=(0, 1))
    wall = np.bincount(room_id[exposed & (room_id >= 0)], minlength=n).astype(float) * CV_M
    return area.astype(float), wall


# -- per-room options (stage M, 2026-10-31): a room policy assigns each zone one of several option dicts of the
#    per-room switches (G36Config zone_min_ra_lps_m2, zone_min_rp_lps_m2, zone_heat_wall_ref_m, zone_heat_wall_power);
#    the controller then uses per-zone arrays instead of the scalar switches (G36Controller.room_settings).
FEATURES = ("area", "wall", "wall_per_area", "sides", "compact", "core", "ahu_size", "oat", "block")


def room_features(metadata, index):
    """Per-zone physical quantities in index.vav_ids order: area m2, exterior wall m, wall per area, exterior sides (of
    N/E/S/W with >= 2 m of exterior wall), compactness (cells / bounding-box cells), core (no exterior wall), zones on
    the same AHU."""
    from smart_buildings.smart_control.simulator import building_utils, constants
    rooms, exterior_walls, _, _, _ = building_utils.construct_building_data_types(
        floor_plan=np.asarray(metadata["floor_plan"]), zone_map=np.asarray(metadata["zone_map"]))
    shell = exterior_walls == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
    area, wall = room_quantities(metadata, index)
    n = index.n_zones
    sides, compact = np.zeros(n), np.zeros(n)
    for z, vav in enumerate(index.vav_ids):
        cells = np.asarray(rooms[vav.replace("VAV_", "room_")])
        r, c = cells[:, 0], cells[:, 1]
        compact[z] = len(cells) / float((r.max() - r.min() + 1) * (c.max() - c.min() + 1))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = np.clip(r + dr, 0, shell.shape[0] - 1), np.clip(c + dc, 0, shell.shape[1] - 1)
            sides[z] += float(shell[rr, cc].sum() >= 10)
    # stage 戊 two levels (2026-11-06, docs/plan/tree-two-level-20261106.md): exterior wall m per orientation (row - 1
    # = north, as geo5_tree), perimeter depth, the AHU's share of rooms with exterior wall, and the interior wall m a
    # room shares with other rooms (a wall cell touching it and another room); trees cannot rotate axes themselves
    room_id = np.full(shell.shape, -1, dtype=np.int64)
    for z, vav in enumerate(index.vav_ids):
        cells = np.asarray(rooms[vav.replace("VAV_", "room_")])
        room_id[cells[:, 0], cells[:, 1]] = z
    shifts = {"wall_n": (-1, 0), "wall_s": (1, 0), "wall_w": (0, -1), "wall_e": (0, 1)}
    out = {}
    for name, (dr, dc) in shifts.items():
        nb = np.roll(shell, (-dr, -dc), axis=(0, 1))
        out[name] = np.bincount(room_id[(room_id >= 0) & nb], minlength=n).astype(float) * CV_M
    other = (room_id < 0) & ~shell
    near = np.stack([np.roll(room_id, (-dr, -dc), axis=(0, 1)) for dr, dc in shifts.values()])
    shared = np.zeros(n)
    for d in range(len(near)):
        own = near[d]
        mixed = other & (own >= 0) & np.any((near >= 0) & (near != own[None]), axis=0)
        shared += np.bincount(own[mixed], minlength=n)
    group = np.asarray(index.zone_group)
    ext = np.array([np.mean(wall[group == g] > 0) for g in range(int(group.max()) + 1)])
    return {"area": area, "wall": wall, "wall_per_area": wall / np.maximum(area, 1e-9), "sides": sides,
            "compact": compact, "core": (wall <= 0).astype(float),
            "ahu_size": np.asarray(index.group_sizes, dtype=float)[index.zone_group],
            **out, "depth": np.minimum(area / np.maximum(wall, CV_M), 1000.0), "ahu_ext_share": ext[group],
            "shared_wall": shared * CV_M, **neighbour_features(room_id, shell, area, wall)}


def neighbour_features(room_id, shell, area, wall):
    """戊之三 (2026-11-10): the room's neighbours through shared interior walls (the graph representation): nb_n
    (number of neighbours), nb_ext (their share with an exterior wall) and nb_wall_pa (their exterior wall per area),
    weighted by the shared length. Pairs as the training features (.tmp/setup/geo5_tree.unit_features): an interior
    non-room, non-shell cell touching two rooms (4-neighbours inside the plan) counts CV_M for that pair, in row-major
    cell order; each pair's length rounded to 0.001 m."""
    n = len(area)
    other = (room_id < 0) & ~shell
    pad = np.pad(room_id, 1, constant_values=-1)
    nbs = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
    srt = np.sort(nbs, axis=0)
    distinct = (srt[0] >= 0).astype(int) + sum(((srt[k] >= 0) & (srt[k] != srt[k - 1])).astype(int)
                                               for k in range(1, 4))
    pairs = {}
    for r, c in zip(*np.nonzero(other & (distinct >= 2))):
        near = sorted({int(v) for v in nbs[:, r, c] if v >= 0})
        for i, a in enumerate(near):
            for b in near[i + 1:]:
                pairs[(a, b)] = pairs.get((a, b), 0.0) + CV_M
    has_ext = (wall > 0).astype(float)
    wpa = wall / np.maximum(area, 1e-9)
    cnt, length, e_sum, w_sum = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for (a, b), v in sorted(pairs.items()):
        v = round(v, 3)
        for p, q in ((a, b), (b, a)):
            cnt[p] += 1
            length[p] += v
            e_sum[p] += v * has_ext[q]
            w_sum[p] += v * wpa[q]
    return {"nb_n": cnt, "nb_ext": e_sum / np.maximum(length, 1e-9), "nb_wall_pa": w_sum / np.maximum(length, 1e-9)}


BUILDING_FEATURES = ("zones", "area_m2", "zone_area_m2", "wall_per_area", "core_share", "ahus", "zones_per_ahu")


def building_features(metadata, index):
    """Static whole-building features of the rule gates (pipeline 5 stage 戊, 2026-11-05): zone count, floor area m2,
    mean zone area m2, exterior wall m per m2 of floor, share of zones without exterior wall, AHU count, zones per
    AHU. The gate trees are trained on exactly these values."""
    area, wall = room_quantities(metadata, index)
    n, k = int(index.n_zones), max(int(index.n_ahu), 1)
    return {"zones": float(n), "area_m2": float(area.sum()), "zone_area_m2": float(area.mean()),
            "wall_per_area": float(wall.sum() / max(float(area.sum()), 1e-9)), "core_share": float(np.mean(wall <= 0)),
            "ahus": float(k), "zones_per_ahu": float(n) / k}


def _hash(number, seed):
    return (int(number) * 2654435761 + int(seed) * 97 + 12345) % (2 ** 32)


def tree_choice(tree, x):
    """Leaf option of a dict tree {"feature", "threshold", "left", "right"} / {"leaf": option} for one zone's x."""
    node = tree
    while "leaf" not in node:
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return int(node["leaf"])


def block_of(policy, hour):
    """Time block of the hour of day under the policy's "blocks" (start hours, e.g. [0, 6, 10, 16]); 0 without."""
    starts = policy.get("blocks")
    if not starts or hour is None:
        return 0
    return max(i for i, start in enumerate(starts) if int(hour) >= start)


def period_of(periods, minute_of_day, weekday, start_hour, end_hour):
    """Period of the day under a room policy's "periods" (2026-11-02, docs/plan/geo-room-stages-4.md), from the
    policy's own clock and the occupancy schedule: 0 outside the schedule (weekend, before the start, after the end),
    1 warm-up (the first warm_h hours of the schedule, default 1), 3 late (the last late_h hours, default 2), 2 in
    between. Hourly decisions, so the edges are hour edges."""
    minute = int(minute_of_day)
    if int(weekday) >= 5 or minute < int(start_hour) * 60 or minute >= int(end_hour) * 60:
        return 0
    if minute < int(start_hour) * 60 + round(60 * float(periods.get("warm_h", 1.0))):
        return 1
    if minute >= int(end_hour) * 60 - round(60 * float(periods.get("late_h", 2.0))):
        return 3
    return 2


def option_index(policy, features, room_numbers, oat=None, hour=None, period=None):
    """Each zone's option index under a room policy: "fixed" (one option), "latin" (option (hash(room) + shift +
    3 x block) % K, so that K shifts give every room, in every time block, every option once while its neighbours
    differ), "tree" (tree_choice on the zone's features, the 24 h outside air, deg C, and the time block), "mix"
    (tree, except the rooms whose hash + shift is even, which take the latin option). A tree may also split on "period"
    (period_of; 0 when none is given)."""
    k, mode = len(policy["options"]), policy["mode"]
    if mode == "fixed":
        return np.full(len(room_numbers), int(policy["option"]))
    block = block_of(policy, hour)
    hashed = np.array([_hash(n, policy.get("seed", 0)) for n in room_numbers])
    latin = (hashed + int(policy.get("shift", 0)) + 3 * block) % k
    if mode == "latin":
        return latin
    rows = [{**{name: float(v[z]) for name, v in features.items()}, "oat": float(oat if oat is not None else 10.0),
             "block": float(block), "period": float(period or 0)} for z in range(len(room_numbers))]
    tree = np.array([tree_choice(policy["tree"], x) for x in rows])
    coded = policy.get("coded")
    if coded:          # stage 戊 (2026-11-06): options = base options x codes; a second tree gives each room's code
        tree = tree + int(coded["K"]) * np.array([tree_choice(coded["tree"], x) for x in rows])
    if mode == "tree":
        return tree
    if mode == "mix":
        forced = ((hashed + int(policy.get("shift", 0))) % 2) == 0
        return np.where(forced, latin, tree)
    raise ValueError(f"unknown room policy mode {mode!r}")


def settings(options, chosen):
    """Per-zone arrays for G36Controller.room_settings from the chosen option of each zone."""
    pick = [options[i] for i in chosen]
    return {"min_ra": np.array([o.get("zone_min_ra_lps_m2", 0.0) for o in pick], dtype=float),
            "min_rp": np.array([o.get("zone_min_rp_lps_m2", 0.0) for o in pick], dtype=float),
            "wall_ref": np.array([o.get("zone_heat_wall_ref_m", 0.0) for o in pick], dtype=float),
            "wall_power": np.array([o.get("zone_heat_wall_power", 0.5) for o in pick], dtype=float),
            "oa_divide": np.array([bool(o.get("zone_min_oa_divide", False)) for o in pick], dtype=bool),
            # occupied targets moved inside (+) or out to the band edge (-) of the comfort window, K (stage N3)
            "heat_margin": np.array([o.get("room_heat_margin_k", 0.0) for o in pick], dtype=float),
            "cool_margin": np.array([o.get("room_cool_margin_k", 0.0) for o in pick], dtype=float),
            # option audit O527 (2026-11-05): how far an option moved the cooling target toward the edge, K; the A12
            # comfort guard keeps the un-relaxed target (cooling target - this) for its near test (0 = none)
            "cool_relax": np.array([o.get("room_cool_relax_k", 0.0) for o in pick], dtype=float),
            # heating airflow maximum, fraction of the cooling maximum (0 = the rule's own), occupied rooms (2026-11-02)
            "heat_max_ratio": np.array([o.get("room_heat_max_ratio", 0.0) for o in pick], dtype=float),
            # heating discharge offset above the heating target, K (0 = the rule's own), occupied rooms (2026-11-02)
            "discharge_offset": np.array([o.get("room_discharge_offset_k", 0.0) for o in pick], dtype=float),
            # 2026-11-09 戊之三: occupied tracking-error reserve gain (0 = off) and own-start warm-up (O12) per room
            "margin_reserve": np.array([o.get("room_margin_reserve", 0.0) for o in pick], dtype=float),
            "stagger": np.array([o.get("room_stagger", 0.0) for o in pick], dtype=float)}
