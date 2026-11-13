"""Dashboard v1 recorder (2026-10-30): one whole 30-day episode per milestone policy and scenario on the fast simulator
(rom; SB_FASTSIM=rom, the official equipment and reward code), recorded hour by hour for the floor-plan viewer in
data/runs/dashboard/ (index.html).

The episode itself is smart_building.evaluation.gateway_episode.run_gateway_episode, unchanged. Nothing in the policy
is touched; three hooks only read:
- evaluate.load_hybrid_environment / create_building_meta_data are wrapped to keep the env and the policy metadata;
- env.step (instance attribute) is wrapped to keep the observation after each step (de-normalised exactly as
  smart_building/g36/building.py BuildingIndex._field does: exact measurement keys only);
- SetpointEnergyCarbonRegretFunction.compute_reward is wrapped as in pipeline/setup/energy_diag.py: per step every
  device's rate is priced and weighted exactly as the reward does, and every zone's share of the productivity regret
  is taken from the reward's own _get_zone_productivity_reward, so zone shares add up to the comfort part.

Zone names (RewardInfo zone ids = keys of the live simulator's building.room_dict, e.g. room_12) are mapped to grid
cells with room_dict itself (padded simulator frame); the mapping is checked against an independent 4-connected
labelling of zone_map.npy == 0 (interior space) and against the observation's VAV zone temperatures (geometry()).

Writes data/runs/dashboard/records/<policy>__<pool>__<building>__<weather>.json and
data/runs/dashboard/buildings/<building>.json, then rebuilds data/runs/dashboard/index.json (dash_index.build).

python3 -B pipeline/launch/launch.py pipeline/setup/dash_record.py <workers> <policy>,<policy>... <spec> [<spec> ...]
policy: rule (G36PolicyB2R2), rule_v4 (G36PolicyC3LaaOccrowDampfloor) or a model JSON path (b2rl_policy_class).
DASH_MAX_STEPS=<n> for smoke runs only (written with a "smoke" flag; the index ignores them).
"""
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = ROOT / "data/runs/dashboard"
PARTS = ("blower", "ac_cool", "ac_heat", "pump", "gas", "hp_heat")


def policy_name(policy):
    if policy in ("rule", "rule_v4"):
        return policy
    if policy.startswith("cond="):                      # cond=<model.json>=<rule.json>: the condition-based rule
        return "cond_" + Path(policy.split("=")[2]).stem
    return Path(policy).stem.replace("model-", "")


def policy_class(policy, spec=None):
    from smart_building.policies.b2rl import b2rl_policy_class
    from smart_building.policies.g36 import G36PolicyB2R2, G36PolicyC3LaaOccrowDampfloor
    if policy == "rule":
        return G36PolicyB2R2
    if policy == "rule_v4":
        return G36PolicyC3LaaOccrowDampfloor
    if policy.startswith("cond="):                      # 2026-10-31: the strongest candidate (pipeline/setup/cond_policy.py)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from cond_policy import cond_policy_class
        _, model, rule = policy.split("=")
        return cond_policy_class(ROOT / model, ROOT / rule, spec)
    return b2rl_policy_class(ROOT / policy)


def numeric_suffix(name):
    tail = name.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 10 ** 9


def rle(flat):
    """Run-length pairs [value, count, value, count, ...] of a 1-D integer array."""
    flat = np.asarray(flat).ravel()
    if flat.size == 0:
        return []
    change = np.flatnonzero(np.diff(flat)) + 1
    starts = np.concatenate(([0], change))
    counts = np.diff(np.concatenate((starts, [flat.size])))
    out = np.empty(2 * starts.size, dtype=np.int64)
    out[0::2], out[1::2] = flat[starts], counts
    return out.tolist()


def runs(mask, min_gap):
    """[start, end] index runs of True in a 1-D mask, joining runs separated by fewer than min_gap False entries."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out = [[int(idx[0]), int(idx[0])]]
    for i in idx[1:]:
        if i - out[-1][1] - 1 < min_gap:
            out[-1][1] = int(i)
        else:
            out.append([int(i), int(i)])
    return out


def geometry(env, scenario_dir, metadata):
    """Static geometry of one building in the simulator's own frame, plus the zone-name to grid-cell mapping check."""
    from scipy import ndimage
    from smart_buildings.smart_control.simulator import building_utils, constants

    sim = env.building.simulator
    b = sim.building
    hvac = sim._hvac
    fp = np.load(scenario_dir / "floor_plan.npy")
    zm = np.load(scenario_dir / "zone_map.npy")
    pfp, offset = building_utils.guarantee_air_padding_in_frame(fp)
    pzm, offset_zm = building_utils.guarantee_air_padding_in_frame(zm)
    ext_walls = np.asarray(b._exterior_walls)
    shape = ext_walls.shape
    rooms = {k: v for k, v in b.room_dict.items() if not constants.is_non_physical_space(k)}
    names = sorted(rooms, key=numeric_suffix)
    label = np.full(shape, -1, dtype=np.int32)
    for i, name in enumerate(names):
        rr, cc = np.asarray(rooms[name]).T
        label[rr, cc] = i

    # --- mapping check: room_dict cells == 4-connected components of zone_map interior space (value 0)
    check = {"frame_shape": list(shape), "file_shape": list(fp.shape), "offset": list(offset),
             "offset_zone_map": list(offset_zm), "shape_ok": bool(pzm.shape == shape and pfp.shape == shape)}
    interior = pzm == constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT
    comp, n_comp = ndimage.label(interior, structure=ndimage.generate_binary_structure(2, 1))
    room_cells = label >= 0
    check["rooms"] = len(names)
    check["components"] = int(n_comp)
    check["cells_rooms"] = int(room_cells.sum())
    check["cells_interior"] = int(interior.sum())
    check["room_cells_equal_interior"] = bool(np.array_equal(room_cells, interior))
    bijective = n_comp == len(names)
    seen = set()
    for i, name in enumerate(names):
        ids = np.unique(comp[label == i])
        if ids.size != 1 or ids[0] == 0 or int(ids[0]) in seen or int((comp == ids[0]).sum()) != int((label == i).sum()):
            bijective = False
            break
        seen.add(int(ids[0]))
    check["rooms_equal_components"] = bool(bijective)

    # --- floors: building rows separated by >= 10 rows of exterior space (floor_generator.MultiFloorPlan stacks floors
    # vertically with a 10-row exterior separator)
    building_mask = pfp != constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT
    row_runs = runs(building_mask.any(axis=1), 10)
    check["floors_found"] = len(row_runs)
    check["floors_metadata"] = int(metadata["num_floors"])
    cv_m = float(b.cv_size_cm) / 100.0 if hasattr(b, "cv_size_cm") else 0.2
    zone_floor = {}
    floors = []
    margin = 2
    for f, (r0, r1) in enumerate(row_runs):
        cols = np.flatnonzero(building_mask[r0:r1 + 1].any(axis=0))
        c0, c1 = int(cols[0]), int(cols[-1])
        R0, R1 = max(0, r0 - margin), min(shape[0] - 1, r1 + margin)
        C0, C1 = max(0, c0 - margin), min(shape[1] - 1, c1 + margin)
        sub = label[R0:R1 + 1, C0:C1 + 1].copy()
        walls = (pfp[R0:R1 + 1, C0:C1 + 1] == constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT)
        doors = (pzm[R0:R1 + 1, C0:C1 + 1] == constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT) & ~walls
        ext = ext_walls[R0:R1 + 1, C0:C1 + 1] == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
        sub[walls] = -2
        sub[ext] = -3
        sub[doors & (sub < 0)] = -4
        diff = np.argwhere(np.asarray(b.diffusers)[R0:R1 + 1, C0:C1 + 1] > 0)
        floors.append({"index": f, "origin": [R0, C0], "rows": int(R1 - R0 + 1), "cols": int(C1 - C0 + 1),
                       "raster_rle": rle(sub), "diffusers": diff.astype(int).ravel().tolist()})
        for i in np.unique(sub[sub >= 0]):
            zone_floor.setdefault(int(i), []).append(f)
    check["each_zone_on_one_floor"] = all(len(v) == 1 for v in zone_floor.values()) and len(zone_floor) == len(names)

    # --- per zone: area, exterior wall length (faces shared with exterior-wall cells), AHU, VAVs
    is_ext = ext_walls == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
    faces = np.zeros(shape, dtype=np.int32)
    faces[1:, :] += is_ext[:-1, :]
    faces[:-1, :] += is_ext[1:, :]
    faces[:, 1:] += is_ext[:, :-1]
    faces[:, :-1] += is_ext[:, 1:]
    ahu_of = {}
    for ahu, zs in metadata["ahu_zone_assignments"].items():
        for z in zs:
            ahu_of[z] = ahu
    zones = []
    for i, name in enumerate(names):
        cells = np.asarray(rooms[name])
        f = zone_floor[i][0]
        R0, C0 = floors[f]["origin"]
        area = len(cells) * cv_m * cv_m
        wall = float(faces[cells[:, 0], cells[:, 1]].sum()) * cv_m
        zones.append({"id": name, "floor": f, "ahu": ahu_of.get(name), "vavs": list(hvac.get_vav_ids_for_zone(name)),
                      "area_m2": round(area, 2), "ext_wall_m": round(wall, 2),
                      "exposure": round(wall / area, 4) if area else 0.0,
                      "centroid": [round(float(cells[:, 0].mean()) - R0, 1), round(float(cells[:, 1].mean()) - C0, 1)],
                      "cells": int(len(cells))})
    check["all_zones_have_ahu"] = all(z["ahu"] for z in zones)
    check["all_zones_have_vav"] = all(z["vavs"] for z in zones)
    return {"cell_m": cv_m, "zones": zones, "floors": floors,
            "ahus": sorted(metadata["ahu_zone_assignments"], key=numeric_suffix),
            "num_floors": int(metadata["num_floors"]), "comfort_window_c": [round(v - 273.15, 2) for v in metadata["comfort_temp_window"]],
            "eco_window_c": [round(v - 273.15, 2) for v in metadata["eco_temp_window"]],
            "morning_start_hour": int(metadata["morning_start_hour"]), "evening_start_hour": int(metadata["evening_start_hour"]),
            "mapping_check": check}


def obs_reader(metadata):
    """name -> (index, scale, offset) with BuildingIndex._field's rule: de-normalise only on an exact measurement key."""
    norm = metadata["observation_normalization_metadata"]
    out = {}
    for i, name in enumerate(metadata["observation_field_names"]):
        if name.startswith("VAV_"):
            parts = name.split("_", 2)
            measurement = parts[2] if len(parts) == 3 else name
        elif name.startswith(("ahs_", "hws_")):
            measurement = name[4:]
        else:
            measurement = None
        c = norm.get(measurement) if measurement else None
        out[name] = (i, float(c["sample_std"]), float(c["sample_mean"])) if c else (i, 1.0, 0.0)
    return out


def episode(job):
    policy, spec = job
    os.environ["SB_FASTSIM"] = "rom"
    from smart_building.learned.b2rl_env import setup_simulator, spec_dir
    setup_simulator(ROOT)
    from smart_buildings import evaluate
    from smart_buildings.smart_control.reward import setpoint_energy_carbon_regret as regret
    from smart_buildings.smart_control.utils import conversion_utils
    from smart_building.evaluation.gateway_episode import run_gateway_episode

    name = policy_name(policy)
    scenario_dir = spec_dir(spec)
    hold = {}
    steps = []          # one dict per step (reward side), filled with the observation after the step
    seen = set()

    original_load = evaluate.load_hybrid_environment
    original_meta = evaluate.create_building_meta_data
    # Per-room energy (2026-10-31): every step the simulator asks each VAV for its coil's reheat once
    # (execute_step_sim, official and fast alike) and bills the boiler for their sum; keep those per-VAV watts.
    from smart_buildings.smart_control.simulator import vav as vav_module
    original_reheat = vav_module.Vav.compute_reheat_energy_rate
    reheat_now, reheat_steps, in_step = {}, [], [False]

    def reheat_hook(self, supply_air_temp, input_water_temp):
        w = original_reheat(self, supply_air_temp, input_water_temp)
        if in_step[0]:
            reheat_now[self.device_id()] = reheat_now.get(self.device_id(), 0.0) + float(w)
        return w

    vav_module.Vav.compute_reheat_energy_rate = reheat_hook
    # The reward reports the AHU system as one device (AirHandlerSystem sums its AHUs); keep each AHU's own fan and
    # cooling rates from the calls the system makes for the reward (the last value per AHU = the state rewarded).
    from smart_buildings.smart_control.simulator import air_handler as ahu_module
    ahu_now = {}
    ahu_originals = {m: getattr(ahu_module.AirHandler, m) for m in
                     ("compute_intake_fan_energy_rate", "compute_exhaust_fan_energy_rate",
                      "compute_cooling_electrical_energy_rate")}

    def ahu_hook(method):
        original_method = ahu_originals[method]

        def hooked(self, *args, **kwargs):
            value = original_method(self, *args, **kwargs)
            ahu_now.setdefault(self.device_id(), {})[method] = float(value)
            return value
        return hooked

    for m in ahu_originals:
        setattr(ahu_module.AirHandler, m, ahu_hook(m))

    def load(path):
        env = original_load(path)
        hold["env"] = env
        sim = env.building.simulator
        sim_step = sim.execute_step_sim

        def wrapped_sim_step(*args, **kwargs):
            reheat_now.clear()
            in_step[0] = True
            try:
                return sim_step(*args, **kwargs)
            finally:
                in_step[0] = False
                reheat_steps.append(dict(reheat_now))

        sim.execute_step_sim = wrapped_sim_step
        step = env.step

        def wrapped_step(action):
            result = step(action)
            if steps and "obs" not in steps[-1]:
                steps[-1]["obs"] = np.asarray(result[0], dtype=np.float64).copy()
                steps[-1]["reward"] = float(result[1])
            return result

        env.step = wrapped_step
        return env

    def meta(env, output_dir):
        m = original_meta(env, output_dir)
        hold["meta"] = m
        return m

    evaluate.load_hybrid_environment = load
    evaluate.create_building_meta_data = meta
    original = regret.SetpointEnergyCarbonRegretFunction.compute_reward

    def wrapped(self, info):
        response = original(self, info)
        stamp = (info.start_timestamp.seconds, info.start_timestamp.nanos)
        if stamp in seen:
            return response
        seen.add(stamp)
        start = conversion_utils.proto_to_pandas_timestamp(info.start_timestamp)
        end = conversion_utils.proto_to_pandas_timestamp(info.end_timestamp)
        dt = (end - start).total_seconds()
        rates = dict.fromkeys(PARTS, 0.0)
        for a in info.air_handler_reward_infos.values():
            rates["blower"] += a.blower_electrical_energy_rate
            ac = a.air_conditioning_electrical_energy_rate
            rates["ac_cool" if ac < 0 else "ac_heat"] += abs(ac)
        for bo in info.boiler_reward_infos.values():
            rates["pump"] += bo.pump_electrical_energy_rate
            rates["gas"] += bo.natural_gas_heating_energy_rate
        for h in info.heat_pump_reward_infos.values():
            rates["pump"] += abs(h.pump_electrical_energy_rate)
            rates["hp_heat"] += abs(h.electricity_heating_energy_rate)
        elec, gas = self._electricity_energy_cost, self._natural_gas_energy_cost
        max_cost = elec.cost(start, end, self._max_electricity_rate) + gas.cost(start, end, self._max_natural_gas_rate)
        max_carbon = (elec.carbon(start, end, self._max_electricity_rate)
                      + gas.carbon(start, end, self._max_natural_gas_rate))
        scale = self._productivity_weight + self._energy_cost_weight + self._carbon_emission_weight
        energy, carbon = {}, {}

        def priced(model, rate):    # one rate's energy + carbon cost, weighted exactly as the reward does
            e = self._energy_cost_weight * model.cost(start, end, rate) / max_cost / scale if max_cost > 0 else 0.0
            c = (self._carbon_emission_weight * model.carbon(start, end, rate) / max_carbon / scale
                 if max_carbon > 0 else 0.0)
            return e, c

        for p, rate in rates.items():
            energy[p], carbon[p] = priced(gas if p == "gas" else elec, rate)
        # per AHU (fan, coil) cost: the system's priced fan and coil cost shared by each AHU's own rate
        fan_rate = sum(a.blower_electrical_energy_rate for a in info.air_handler_reward_infos.values())
        ac_rate = sum(a.air_conditioning_electrical_energy_rate for a in info.air_handler_reward_infos.values())
        fan_cost, coil_cost = sum(priced(elec, fan_rate)), sum(priced(elec, abs(ac_rate)))
        own = {aid: (m.get("compute_intake_fan_energy_rate", 0.0) + m.get("compute_exhaust_fan_energy_rate", 0.0),
                     m.get("compute_cooling_electrical_energy_rate", 0.0)) for aid, m in ahu_now.items()}
        f_sum, a_sum = sum(f for f, _ in own.values()), sum(a for _, a in own.values())
        if own:
            ahu_cost = {aid: (fan_cost * f / f_sum if f_sum else 0.0, coil_cost * a / a_sum if a_sum else 0.0)
                        for aid, (f, a) in own.items()}
        else:
            ahu_cost = {"system": (fan_cost, coil_cost)}
        ahu_check = (abs(f_sum - fan_rate), abs(a_sum - ac_rate))
        zones = {}
        total_occ = 0.0
        for zid, z in info.zone_reward_infos.items():
            total_occ += z.average_occupancy
        denom = ((self._max_productivity_personhour_usd - self._min_productivity_personhour_usd)
                 * total_occ * dt / 3600.0)
        for zid, z in info.zone_reward_infos.items():
            p = self._get_zone_productivity_reward(z.heating_setpoint_temperature, z.cooling_setpoint_temperature,
                                                   z.zone_air_temperature, dt, z.average_occupancy)
            pmax = self._max_productivity_personhour_usd * z.average_occupancy * dt / 3600.0
            share = -self._productivity_weight * (p - pmax) / denom / scale if total_occ > 0 else 0.0
            zones[zid] = (z.zone_air_temperature, z.heating_setpoint_temperature, z.cooling_setpoint_temperature,
                          z.average_occupancy, z.air_flow_rate, z.air_flow_rate_setpoint, share)
        steps.append({"start": start, "dt": dt, "energy": energy, "carbon": carbon, "zones": zones,
                      "comfort": -self._productivity_weight * response.normalized_productivity_regret / scale,
                      "occupancy": response.total_occupancy, "ahu_cost": ahu_cost, "ahu_check": ahu_check,
                      "reheat": reheat_steps[-1] if reheat_steps else {}, "sim_steps": len(reheat_steps)})
        return response

    regret.SetpointEnergyCarbonRegretFunction.compute_reward = wrapped
    max_steps = int(os.environ.get("DASH_MAX_STEPS", "0")) or None
    t0 = time.perf_counter()
    try:
        row = run_gateway_episode(scenario_dir, policy_class(policy, spec), policy_key=name, label=spec,
                                  max_steps=max_steps)
    finally:
        regret.SetpointEnergyCarbonRegretFunction.compute_reward = original
        evaluate.load_hybrid_environment = original_load
        evaluate.create_building_meta_data = original_meta
        vav_module.Vav.compute_reheat_energy_rate = original_reheat
        for m, f in ahu_originals.items():
            setattr(ahu_module.AirHandler, m, f)
    elapsed = time.perf_counter() - t0
    env, metadata = hold["env"], hold["meta"]
    building = spec.split("/")[1]
    geo = geometry(env, scenario_dir, metadata)
    write_json(OUT / "buildings" / f"{building}.json", {"building": building, **geo})
    record = summarise(name, spec, row, steps, metadata, geo, elapsed, smoke=max_steps is not None)
    tag = "smoke__" if max_steps is not None else ""
    write_json(OUT / "records" / f"{tag}{name}__{spec.replace('/', '__')}.json", record)
    return name, spec, record["summary"], geo["mapping_check"]


def summarise(name, spec, row, steps, metadata, geo, elapsed, smoke):
    n = len(steps)
    if n != row["steps"]:
        raise RuntimeError(f"{n} recorded steps for {row['steps']} episode steps")
    zone_ids = [z["id"] for z in geo["zones"]]
    reward_ids = set(steps[0]["zones"])
    ids_ok = reward_ids == set(zone_ids)
    zi = {z: i for i, z in enumerate(zone_ids)}
    nz = len(zone_ids)
    reader = obs_reader(metadata)
    # hourly bins on the simulation clock (the naive timestamps the setpoint schedule judges hours by; RewardInfo
    # carries them as UTC): a step belongs to the clock hour its interval starts in
    t0 = steps[0]["start"].tz_convert(None)
    offsets = np.array([(s["start"].tz_convert(None) - t0).total_seconds() for s in steps])
    hour_of = ((t0.minute * 60 + t0.second + offsets) // 3600).astype(int)
    hours = int(hour_of[-1]) + 1
    counts = np.maximum(np.bincount(hour_of, minlength=hours).astype(float), 1.0)

    T = np.zeros((n, nz)); H = np.zeros((n, nz)); C = np.zeros((n, nz)); O = np.zeros((n, nz))
    F = np.zeros((n, nz)); FS = np.zeros((n, nz)); S = np.zeros((n, nz))
    for k, s in enumerate(steps):
        for zid, v in s["zones"].items():
            j = zi[zid]
            T[k, j], H[k, j], C[k, j], O[k, j], F[k, j], FS[k, j], S[k, j] = v
    T, H, C = T - 273.15, H - 273.15, C - 273.15
    dt_h = np.array([s["dt"] for s in steps]) / 3600.0
    comfort = np.array([s["comfort"] for s in steps])
    energy = {p: np.array([s["energy"][p] for s in steps]) for p in PARTS}
    carbon = {p: np.array([s["carbon"][p] for s in steps]) for p in PARTS}
    occ_total = np.array([s["occupancy"] for s in steps])

    # observations after each step (the last step's observation is always there: env.step returns it)
    obs = np.stack([s["obs"] for s in steps])

    def field(fname):
        i, scale, offset = reader[fname]
        return obs[:, i] * scale + offset

    vav_of = {z["id"]: z["vavs"] for z in geo["zones"]}
    reheat = np.zeros((n, nz)); damper = np.zeros((n, nz)); obs_t = np.zeros((n, nz))
    for z, j in zi.items():
        vs = vav_of[z]
        reheat[:, j] = np.mean([field(f"{v}_reheat_valve_percentage_command") for v in vs], axis=0)
        damper[:, j] = np.mean([field(f"{v}_supply_air_damper_percentage_command") for v in vs], axis=0)
        obs_t[:, j] = np.mean([field(f"{v}_zone_air_temperature_sensor") for v in vs], axis=0) - 273.15
    # The observation env.step(k) returns reports the zone temperatures RewardInfo gave for step k - 1 (measured: equal
    # to float32 rounding for every zone and step); this is the check that VAV_n (observation) and room_m (RewardInfo,
    # room_dict) are the same zone. Commands (damper, reheat) in that observation are the ones applied in step k.
    obs_temp_diff = float(np.max(np.abs(obs_t[1:] - T[:-1]))) if n > 1 else 0.0
    lag_diff = {"obs_k_vs_reward_k": float(np.max(np.abs(obs_t - T))),
                "obs_k_vs_reward_k-1": float(np.max(np.abs(obs_t[1:] - T[:-1]))) if n > 1 else None,
                "obs_k_vs_reward_k+1": float(np.max(np.abs(obs_t[:-1] - T[1:]))) if n > 1 else None,
                "median_abs_k": float(np.median(np.abs(obs_t - T))),
                "p99_abs_k": float(np.percentile(np.abs(obs_t - T), 99))}
    if os.environ.get("DASH_DEBUG"):
        k, j = np.unravel_index(np.argmax(np.abs(obs_t - T)), T.shape)
        print("DEBUG worst", k, zone_ids[j], "obs", obs_t[max(0, k - 2):k + 3, j], "reward", T[max(0, k - 2):k + 3, j],
              lag_diff, flush=True)
    pct = 100.0 if max(np.max(np.abs(reheat)), np.max(np.abs(damper))) <= 1.0 + 1e-6 else 1.0

    def hourly(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return np.bincount(hour_of, weights=x, minlength=hours) / counts
        return np.stack([np.bincount(hour_of, weights=x[:, j], minlength=hours) / counts for j in range(x.shape[1])])

    def rnd(a, d):
        return np.round(np.asarray(a, dtype=float), d).tolist()

    # --- zone summaries (5-minute steps; "occupied" = the zone's average occupancy > 0 in the step)
    occ = O > 0
    cold = np.maximum(H - T, 0.0)
    hot = np.maximum(T - C, 0.0)
    out = (cold > 0) | (hot > 0)
    w = dt_h[:, None]
    zone_summary = {
        "occ_hours": rnd((occ * w).sum(0), 2),
        "occ_hours_out": rnd(((occ & out) * w).sum(0), 2),
        "occ_hours_cold": rnd(((occ & (cold > 0)) * w).sum(0), 2),
        "occ_hours_hot": rnd(((occ & (hot > 0)) * w).sum(0), 2),
        "dh_cold": rnd((cold * occ * w).sum(0), 3),
        "dh_hot": rnd((hot * occ * w).sum(0), 3),
        "max_cold": rnd(np.where(occ, cold, 0).max(0), 2),
        "max_hot": rnd(np.where(occ, hot, 0).max(0), 2),
        "t_mean": rnd(T.mean(0), 2),
        "t_mean_occ": rnd([T[occ[:, j], j].mean() if occ[:, j].any() else float("nan") for j in range(nz)], 2),
        "flow_mean": rnd(F.mean(0), 4),
        "flow_mean_occ": rnd([F[occ[:, j], j].mean() if occ[:, j].any() else 0.0 for j in range(nz)], 4),
        "flow_max": rnd(FS.max(0), 4),
        "reheat_mean": rnd(reheat.mean(0) * pct, 1),
        "comfort_cost": rnd(S.sum(0) / n, 7),           # the zone's share of the episode's mean comfort cost
    }
    zone_summary["t_mean_occ"] = [None if (isinstance(v, float) and math.isnan(v)) else v
                                  for v in zone_summary["t_mean_occ"]]

    # --- per-room energy (2026-10-31): each AHU's fan and coil cost split over its zones by their airflow (one supply
    # temperature per AHU, so the coil split is exact; the fan split is a convention); gas and pump cost split by the
    # zones' reheat (the boiler is billed for the coils' summed heat; a VAV's coil heat goes to its zones by area as in
    # the simulator). Boiler cost in a step without reheat, heat-pump heating and unknown devices stay unallocated.
    area = np.array([z["area_m2"] for z in geo["zones"]])
    ahu_zones, vav_zones = {}, {}
    for j, z in enumerate(geo["zones"]):
        ahu_zones.setdefault(z["ahu"], []).append(j)
        for v in z["vavs"]:
            vav_zones.setdefault(v, []).append(j)
    fan, coil, gas_z, pump_z, reheat_w = (np.zeros((n, nz)) for _ in range(5))
    unallocated = np.zeros(n)
    seen_ahus, seen_vavs = set(), set()
    for k, s in enumerate(steps):
        for aid, (f_cost, c_cost) in s["ahu_cost"].items():
            seen_ahus.add(aid)
            js = ahu_zones.get(aid)
            if not js:
                unallocated[k] += f_cost + c_cost
                continue
            flow = F[k, js]
            share = flow / flow.sum() if flow.sum() > 0 else area[js] / area[js].sum()
            fan[k, js] += f_cost * share
            coil[k, js] += c_cost * share
        for v, watts in s["reheat"].items():
            seen_vavs.add(v)
            js = vav_zones.get(v)
            if js:
                reheat_w[k, js] += watts * area[js] / area[js].sum()
        total = sum(s["reheat"].values())
        g = s["energy"]["gas"] + s["carbon"]["gas"]
        p = s["energy"]["pump"] + s["carbon"]["pump"]
        if abs(total) > 1e-9:
            gas_z[k], pump_z[k] = g * reheat_w[k] / total, p * reheat_w[k] / total
            unallocated[k] += (g + p) * (1.0 - reheat_w[k].sum() / total)
        else:
            unallocated[k] += g + p
        unallocated[k] += s["energy"]["hp_heat"] + s["carbon"]["hp_heat"]
    zone_energy = fan + coil + gas_z + pump_z
    zone_summary.update(
        energy_cost=rnd(zone_energy.sum(0) / n, 9),     # the zone's share of the episode's mean energy + carbon cost
        fan_cost=rnd(fan.sum(0) / n, 9), coil_cost=rnd(coil.sum(0) / n, 9),
        gas_cost=rnd(gas_z.sum(0) / n, 9), pump_cost=rnd(pump_z.sum(0) / n, 9),
        total_cost=rnd((zone_energy.sum(0) + S.sum(0)) / n, 9),
        reheat_kwh=rnd((reheat_w * w).sum(0) / 1000.0, 1))

    # --- episode summary and the reward-part check
    comfort_sum = float(comfort.sum())
    energy_sum = float(sum(e.sum() for e in energy.values()))
    carbon_sum = float(sum(c.sum() for c in carbon.values()))
    check = {
        "comfort": row["comfort_part"] + comfort_sum,      # row parts are signed rewards; ours are costs
        "energy": row["energy_part"] + energy_sum,
        "carbon": row["carbon_part"] + carbon_sum,
        "zone_shares_vs_comfort": float(S.sum() - comfort_sum),
        "parts_vs_reward": float(-row["total_reward"] - (comfort_sum + energy_sum + carbon_sum)),
        "obs_vs_reward_zone_temp_max_k": obs_temp_diff,
        "obs_vs_reward_zone_temp": lag_diff,
        "zone_ids_match": bool(ids_ok),
        "zone_energy_vs_total": float(zone_energy.sum() + unallocated.sum() - energy_sum - carbon_sum),
        "ahu_ids_match": seen_ahus == set(ahu_zones),
        "vav_ids_match": seen_vavs == set(vav_zones),
        "sim_steps_last": int(steps[-1]["sim_steps"]),
        "ahu_rates_vs_system_w": max(max(s["ahu_check"]) for s in steps),
    }
    summary = {
        "cost": -row["mean_reward"], "steps": n, "occupied_steps": int((occ_total > 0).sum()),
        "comfort": comfort_sum / n, "energy": energy_sum / n, "carbon": carbon_sum / n,
        "parts": {p: {"energy": float(energy[p].sum() / n), "carbon": float(carbon[p].sum() / n)} for p in PARTS},
        "check": check, "check_max_abs": max(abs(check[k]) for k in ("comfort", "energy", "carbon",
                                                                     "zone_shares_vs_comfort", "parts_vs_reward",
                                                                     "zone_energy_vs_total")),
        "energy_unallocated": float(unallocated.sum() / n),
        "elapsed_seconds": round(elapsed, 1),
    }

    # --- hourly series
    band_lo, band_hi = hourly(H), hourly(C)
    uniform = bool(np.allclose(band_lo, band_lo[:1]) and np.allclose(band_hi, band_hi[:1]))
    ahus = {}
    for k, ahu in enumerate(geo["ahus"], start=1):
        pre = f"ahs_ahu_{k}_"
        sat = field(pre + "supply_air_temperature_sensor")
        sat_sp = field(pre + "supply_air_temperature_setpoint")
        flow = field(pre + "supply_air_flowrate_sensor")
        oa = field(pre + "outside_air_flowrate_sensor")
        recirc = field(pre + "recirculation_percentage_command")
        frac = np.where(flow > 1e-6, oa / np.maximum(flow, 1e-6), np.nan)
        rscale = 100.0 if np.nanmax(np.abs(recirc)) <= 1.0 + 1e-6 else 1.0
        ahus[ahu] = {
            "on": rnd(hourly(field(pre + "supervisor_run_command") > 0.5), 2),
            "sat": rnd(hourly(sat - 273.15 if np.nanmean(sat) > 200 else sat), 2),
            "sat_sp": rnd(hourly(sat_sp - 273.15 if np.nanmean(sat_sp) > 200 else sat_sp), 2),
            "flow": rnd(hourly(flow), 3),
            "oa_frac": [None if math.isnan(v) else round(v, 3) for v in
                        (np.bincount(hour_of, weights=np.nan_to_num(frac), minlength=hours)
                         / np.maximum(np.bincount(hour_of, weights=np.isfinite(frac).astype(float), minlength=hours), 1)
                         ).tolist()],
            "recirc": rnd(hourly(recirc * rscale), 1),
            "static_pa": rnd(hourly(field(pre + "supply_air_static_pressure_sensor")), 1),
        }
    oat = field("ahs_ahu_1_outside_air_temperature_sensor")
    oat = oat - 273.15 if np.nanmean(oat) > 200 else oat
    start_clock = t0.floor("h")
    hourly_block = {
        "outside_t": rnd(hourly(oat), 2),
        "occupancy": rnd(hourly(occ_total), 2),
        "comfort": rnd(hourly(comfort), 7), "energy": rnd(hourly(sum(energy.values())), 7),
        "carbon": rnd(hourly(sum(carbon.values())), 7),
        "parts": {p: rnd(hourly(energy[p] + carbon[p]), 7) for p in PARTS if (energy[p].any() or carbon[p].any())},
        "zone_t": rnd(hourly(T), 2),
        "zone_occ": rnd(hourly(O), 2),
        "zone_flow": rnd(hourly(F), 3),
        "zone_reheat": rnd(hourly(reheat * pct), 0),
        "zone_energy_1e8": np.rint(hourly(zone_energy) * 1e8).astype(int).tolist(),   # per-step cost x 1e8
        "zone_comfort_1e8": np.rint(hourly(S) * 1e8).astype(int).tolist(),   # per-step comfort share x 1e8 (11-01)
        "ahu": ahus,
    }
    if uniform:
        hourly_block["band_lo"], hourly_block["band_hi"] = rnd(band_lo[0], 2), rnd(band_hi[0], 2)
    else:
        hourly_block["zone_lo"], hourly_block["zone_hi"] = rnd(band_lo, 2), rnd(band_hi, 2)
    pool, building, weather = spec.split("/")
    return {"policy": name, "spec": spec, "pool": pool, "building": building, "weather": weather,
            "smoke": smoke, "simulator": "rom", "steps": n, "hours": hours, "step_seconds": 300,
            "start_clock": start_clock.strftime("%Y-%m-%dT%H:%M"), "first_step": t0.strftime("%Y-%m-%dT%H:%M"),
            "time_zone": str(metadata["time_zone"]),
            "zones": zone_ids, "summary": summary, "zone_summary": zone_summary, "hourly": hourly_block}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def mem_available_mb():
    """MemAvailable from /proc/meminfo (Linux); None elsewhere."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


def wait_for_memory():
    """Another job shares this host: start an episode only while at least DASH_MIN_FREE_MB (default 6000) plus one
    episode's footprint (about 1500 MB) stays available."""
    need = int(os.environ.get("DASH_MIN_FREE_MB", "6000")) + 1500
    time.sleep((os.getpid() % 7) * 2.0)          # stagger the workers' first look
    while True:
        free = mem_available_mb()
        if free is None or free >= need:
            return
        time.sleep(20.0)


def safe_episode(job):
    try:
        wait_for_memory()
        return episode(job)
    except Exception:  # noqa: BLE001 - report and keep the other episodes going
        return policy_name(job[0]), job[1], {"error": traceback.format_exc()}, None


def main():
    workers, policies, specs = int(sys.argv[1]), sys.argv[2].split(","), sys.argv[3:]
    if workers > 8:
        raise SystemExit("at most 8 workers on this host")
    jobs = [(p, s) for s in specs for p in policies]
    if not os.environ.get("DASH_FORCE"):        # resume: keep records already written
        jobs = [(p, s) for p, s in jobs
                if not (OUT / "records" / f"{policy_name(p)}__{s.replace('/', '__')}.json").exists()]
    if not jobs:
        print("nothing to run", flush=True)
        return
    failed = 0
    with mp.get_context("spawn").Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
        for name, spec, summary, check in pool.imap_unordered(safe_episode, jobs):
            if "error" in summary:
                failed += 1
                print(f"FAILED {name} {spec}\n{summary['error']}", flush=True)
                continue
            ok = all(v for k, v in check.items() if isinstance(v, bool))
            print(f"{name:28s} {spec:36s} cost {summary['cost']:.5f} comfort {summary['comfort']:.5f} "
                  f"energy {summary['energy']:.5f} carbon {summary['carbon']:.5f} "
                  f"check {summary['check_max_abs']:.1e} tempdiff {summary['check']['obs_vs_reward_zone_temp_max_k']:.1e} "
                  f"map {'ok' if ok else 'FAIL'} {summary['elapsed_seconds']:.0f}s", flush=True)
    import dash_index
    dash_index.build()
    print(f"done; {failed} failed", flush=True)


if __name__ == "__main__":
    main()
