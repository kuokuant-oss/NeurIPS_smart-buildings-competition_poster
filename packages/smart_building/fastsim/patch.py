"""Process-local patches that route the competition simulator's thermal step through a fast core (core.py).

Only SimulatorFlexibleGeometries (the floor-plan simulator of every competition scenario) and its
FloorPlanBasedBuilding are patched, and only their thermal parts; the HVAC devices, occupancy, weather, reward,
observations and actions stay the official code. execute_step_sim below is the official method line for line
(data/raw/kaggle/smart_buildings/smart_control/simulator/simulator_flexible_floor_plan.py) except where it
touches the temperature field, marked "fastsim". A simulator built while a mode is installed carries its core
in _fastsim; simulators built before install, or after uninstall, run the official code unchanged.
"""
import collections
import hashlib
import inspect
import logging

import numba
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_ORIGINAL = {}
_STATE = {"mode": None, "rom": {}}

# sha256 (first 16 hex) of the official sources this module mirrors or replaces, in the competition copy
# (data/raw/kaggle) and the starter kit alike (2026-10-27). A different simulator refuses to install: the copy in
# _execute_step_sim, the cores and the shuffle kernel have to be checked against it first.
MIRRORED = {
    "SimulatorFlexibleGeometries.execute_step_sim": "7b37b47a721ec9b8",
    "SimulatorFlexibleGeometries.reset": "c8a5a512b4014f85",
    "Simulator.reset": "6195adddd6590d76",
    "FiniteDifferenceSolver.solve": "29efdd18dcee6c55",
    "FiniteDifferenceSolver._add_air_side": "925b8b7bc45887fa",
    "FloorPlanBasedBuilding.get_zone_average_temps": "b6ed2f14a5d0e954",
    "FloorPlanBasedBuilding.apply_thermal_power_zone": "f3a520b78e9119c9",
    "FloorPlanBasedBuilding.reset": "2a97040f29ae4d04",
    "StochasticConvectionSimulator.apply_convection": "e40c0bdfbb4ec84b",
    "StochasticConvectionSimulator._shuffle_max_dist": "09e9dac077b8e4a2",
    "_convect_room": "dcb8e1d7978b35dc",
}


def official_sources():
    """Hashes of the official sources named in MIRRORED, as they are in the imported simulator."""
    from smart_buildings.smart_control.simulator import building, finite_difference_solver, simulator
    _, _, _, sfg, convection = _modules()
    objects = {
        "SimulatorFlexibleGeometries.execute_step_sim": sfg.SimulatorFlexibleGeometries.execute_step_sim,
        "SimulatorFlexibleGeometries.reset": sfg.SimulatorFlexibleGeometries.reset,
        "Simulator.reset": simulator.Simulator.reset,
        "FiniteDifferenceSolver.solve": finite_difference_solver.FiniteDifferenceSolver.solve,
        "FiniteDifferenceSolver._add_air_side": finite_difference_solver.FiniteDifferenceSolver._add_air_side,
        "FloorPlanBasedBuilding.get_zone_average_temps": building.FloorPlanBasedBuilding.get_zone_average_temps,
        "FloorPlanBasedBuilding.apply_thermal_power_zone": building.FloorPlanBasedBuilding.apply_thermal_power_zone,
        "FloorPlanBasedBuilding.reset": building.FloorPlanBasedBuilding.reset,
        "StochasticConvectionSimulator.apply_convection":
            convection.StochasticConvectionSimulator.apply_convection,
        "StochasticConvectionSimulator._shuffle_max_dist":
            convection.StochasticConvectionSimulator._shuffle_max_dist,
        "_convect_room": convection._convect_room.py_func,
    }
    return {name: hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()[:16] for name, obj in objects.items()}


def installed():
    return _STATE["mode"]


def _modules():
    from smart_buildings.smart_control.simulator import air_handler as air_handler_py
    from smart_buildings.smart_control.simulator import building as building_py
    from smart_buildings.smart_control.simulator import constants
    from smart_buildings.smart_control.simulator import simulator_flexible_floor_plan as sfg
    from smart_buildings.smart_control.simulator import stochastic_convection_simulator as convection
    return air_handler_py, building_py, constants, sfg, convection


def install(mode, rom_options=None):
    """Patch the simulator classes of this process; mode "exact" or "rom"."""
    if mode not in ("exact", "rom"):
        raise ValueError(f"fast simulator mode {mode!r}: expected 'exact' or 'rom'")
    _, building_py, _, sfg, _ = _modules()
    cls, building = sfg.SimulatorFlexibleGeometries, building_py.FloorPlanBasedBuilding
    if not _ORIGINAL:
        changed = sorted(name for name, digest in official_sources().items() if MIRRORED[name] != digest)
        if changed:
            raise RuntimeError(f"fast simulator: the official simulator changed ({', '.join(changed)}); "
                               "check patch.py, core.py and rom.py against it before using the fast simulator")
        _ORIGINAL.update({
            "init": cls.__init__, "execute_step_sim": cls.execute_step_sim, "reward_info": cls.reward_info,
            "setup_step_sim": cls.setup_step_sim, "reset": building.reset,
            "get_zone_average_temps": building.get_zone_average_temps,
            "get_zone_temp_stats": building.get_zone_temp_stats,
            "apply_thermal_power_zone": building.apply_thermal_power_zone,
        })
        cls.__init__ = _init
        cls.execute_step_sim = _execute_step_sim
        cls.reward_info = _reward_info
        cls.setup_step_sim = _setup_step_sim
        building.reset = _building_reset
        building.get_zone_average_temps = _get_zone_average_temps
        building.get_zone_temp_stats = _get_zone_temp_stats
        building.apply_thermal_power_zone = _apply_thermal_power_zone
    _STATE["mode"] = mode
    _STATE["rom"] = dict(rom_options or {})
    # 2026-11-10: SB_RETURN_AIR=flow -> airflow-weighted return air (starter-kit PR #8), else the shipped area weighting
    import os
    _STATE["return_air"] = os.environ.get("SB_RETURN_AIR", "area")
    if _STATE["return_air"] == "flow" and "air_handler_reward_infos" not in _ORIGINAL:
        _ORIGINAL["air_handler_reward_infos"] = cls._get_air_handler_reward_infos
        cls._get_air_handler_reward_infos = _air_handler_reward_infos_flow


def uninstall():
    """Restore the official methods; simulators already carrying a core keep using it."""
    if not _ORIGINAL:
        return
    _, building_py, _, sfg, _ = _modules()
    cls, building = sfg.SimulatorFlexibleGeometries, building_py.FloorPlanBasedBuilding
    cls.__init__ = _ORIGINAL["init"]
    cls.execute_step_sim = _ORIGINAL["execute_step_sim"]
    cls.reward_info = _ORIGINAL["reward_info"]
    cls.setup_step_sim = _ORIGINAL["setup_step_sim"]
    if "air_handler_reward_infos" in _ORIGINAL:
        cls._get_air_handler_reward_infos = _ORIGINAL["air_handler_reward_infos"]
    building.reset = _ORIGINAL["reset"]
    building.get_zone_average_temps = _ORIGINAL["get_zone_average_temps"]
    building.get_zone_temp_stats = _ORIGINAL["get_zone_temp_stats"]
    building.apply_thermal_power_zone = _ORIGINAL["apply_thermal_power_zone"]
    _ORIGINAL.clear()
    _STATE["mode"] = None


def attach(simulator, mode):
    """Give a constructed simulator its core (core.ExactCore or core.RomCore)."""
    from smart_building.fastsim import core, rom
    air_handler_py, _, _, _, _ = _modules()
    data = core.ThermalData(simulator)
    # The official building keeps a Python list of neighbour coordinates for every cell. Only the solver's
    # constructor reads it (finite_difference_solver.get_cv_mapping), and on the large floor plans it is about 40 %
    # of a worker's memory (val/building_30275: 1.74 GB -> 1.03 GB without it), so it goes before a model build.
    simulator.building.neighbors = []
    h = simulator._weather_controller.get_air_convection_coefficient(simulator.current_timestamp)
    if mode == "rom":
        ahus = (simulator._hvac.air_handler.ahus if isinstance(simulator._hvac.air_handler,
                                                               air_handler_py.AirHandlerSystem)
                else [simulator._hvac.air_handler])
        served = {vav.air_handler.device_id() for vav in simulator._hvac.vavs.values()}
        if any(ahu.device_id() not in served for ahu in ahus):
            raise ValueError("an air handler serves no zone; its recirculation reads the whole field (use exact)")
        model = rom.cached(data, h, **_STATE["rom"], log=log.info)
        thermal = core.RomCore(data, model)
    else:
        thermal = core.ExactCore(data)
        thermal.prepare(h)
    thermal.room_areas = {z: len(coords) for z, coords in simulator.building.room_dict.items()}
    thermal.version = 0
    thermal.reward_cache = None
    thermal.convection = _BatchedConvection(simulator.building) if mode == "exact" else None
    thermal.reset(simulator.building.temp)
    simulator._fastsim = thermal
    simulator.building._fastsim = thermal
    return thermal


def _init(self, *args, **kwargs):
    _ORIGINAL["init"](self, *args, **kwargs)
    if _STATE["mode"]:
        attach(self, _STATE["mode"])


def _building_reset(self):
    _ORIGINAL["reset"](self)
    thermal = getattr(self, "_fastsim", None)
    if thermal is not None:
        thermal.reset(self.temp)
        thermal.version += 1


def _get_zone_average_temps(self):
    thermal = getattr(self, "_fastsim", None)
    if thermal is None:
        return _ORIGINAL["get_zone_average_temps"](self)
    return dict(zip(thermal.data.zones, thermal.zone_means()))


def _get_zone_temp_stats(self, zone_name):
    thermal = getattr(self, "_fastsim", None)
    if thermal is None or thermal.kind == "exact":
        return _ORIGINAL["get_zone_temp_stats"](self, zone_name)
    mean = thermal.zone_means()[thermal.data.zone_index[zone_name]]
    return mean, mean, mean


def _apply_thermal_power_zone(self, zone_name, power):
    thermal = getattr(self, "_fastsim", None)
    if thermal is None:
        return _ORIGINAL["apply_thermal_power_zone"](self, zone_name, power)
    index = thermal.data.zone_index.get(zone_name)
    if index is None:
        raise ValueError(f"fast simulator: thermal power for {zone_name!r}, which is not a physical zone")
    thermal.set_zone_power(index, power)


def _setup_step_sim(self):
    _ORIGINAL["setup_step_sim"](self)
    thermal = getattr(self, "_fastsim", None)
    if thermal is not None:
        thermal.version += 1   # actions are about to change the devices the reward reads


def _recirculation_temps_flow(self, avg_temps, room_areas):
    """Each air handler's return air as the airflow-weighted mixture of its VAVs' extracted air (starter-kit PR #8,
    simulator_flexible_floor_plan._recirculation_temps): a VAV counts with unrestricted_flow_rate_demand when its damper
    is open and its maximum flow is positive; zones within a VAV with their floor area; no airflow -> area."""
    air_handler_py, _, _, _, _ = _modules()
    hvac = self._hvac
    flow_temp_sum = collections.defaultdict(float)
    flow_total = collections.defaultdict(float)
    area_temp_sum = collections.defaultdict(float)
    area_total = collections.defaultdict(float)
    for v_id, vav in hvac.vavs.items():
        zones = [z for z in hvac.get_zones_for_vav(v_id) if z in avg_temps]
        v_area = sum(room_areas[z] for z in zones)
        if v_area <= 0:
            continue
        v_temp = sum(avg_temps[z] * room_areas[z] for z in zones) / v_area
        ahu_id = vav.air_handler.device_id()
        v_flow = vav.unrestricted_flow_rate_demand if vav.damper_setting > 0 and vav.max_air_flow_rate > 0 else 0.0
        flow_temp_sum[ahu_id] += v_temp * v_flow
        flow_total[ahu_id] += v_flow
        area_temp_sum[ahu_id] += v_temp * v_area
        area_total[ahu_id] += v_area
    ahus = hvac.air_handler.ahus if isinstance(hvac.air_handler, air_handler_py.AirHandlerSystem) \
        else [hvac.air_handler]
    temps = {}
    for ahu in ahus:
        ahu_id = ahu.device_id()
        if flow_total.get(ahu_id, 0) > 0:
            temps[ahu_id] = flow_temp_sum[ahu_id] / flow_total[ahu_id]
        elif area_total.get(ahu_id, 0) > 0:
            temps[ahu_id] = area_temp_sum[ahu_id] / area_total[ahu_id]
        else:
            temps[ahu_id] = self.building.temp.mean()
    return temps


def _air_handler_reward_infos_flow(self):
    """The official _get_air_handler_reward_infos with the airflow-weighted return air (PR #8)."""
    air_handler_py, _, _, sfg, _ = _modules()
    reward_info_cls = sfg.RewardInfo if hasattr(sfg, "RewardInfo") else None
    if reward_info_cls is None:
        from smart_buildings.smart_control.proto import smart_control_reward_pb2 as reward_pb
        reward_info_cls = reward_pb.RewardInfo
    hvac = self._hvac
    blower = hvac.air_handler.compute_intake_fan_energy_rate() + hvac.air_handler.compute_exhaust_fan_energy_rate()
    ambient_temp = self._weather_controller.get_current_temp(self._current_timestamp)
    room_areas = {z: len(c) for z, c in self.building.room_dict.items()}
    temps = _recirculation_temps_flow(self, self.building.get_zone_average_temps(), room_areas)
    recirculation_input = temps if isinstance(hvac.air_handler, air_handler_py.AirHandlerSystem) \
        else temps[hvac.air_handler.device_id()]
    ac = hvac.air_handler.compute_cooling_electrical_energy_rate(recirculation_input, ambient_temp)
    info = reward_info_cls.AirHandlerRewardInfo(blower_electrical_energy_rate=blower,
                                                air_conditioning_electrical_energy_rate=ac)
    return {hvac.air_handler.device_id(): info}


def _reward_info(self, occupancy_function):
    """The official reward_info, computed once per simulator state (the gateway loop and the training loop ask
    twice per step; average_zone_occupancy caches the interval, so the second call is the same message)."""
    thermal = getattr(self, "_fastsim", None)
    if thermal is None:
        return _ORIGINAL["reward_info"](self, occupancy_function)
    key = (thermal.version, id(occupancy_function))
    if thermal.reward_cache is None or thermal.reward_cache[0] != key:
        thermal.reward_cache = (key, _ORIGINAL["reward_info"](self, occupancy_function))
    return thermal.reward_cache[1]


@numba.njit(cache=False)
def _convect_rooms(cvs, cvs_ptr, cand_ptr, cand_ptr_offset, cand_idx, cand_idx_offset, p, temp):
    """stochastic_convection_simulator._convect_room (distance-limited branch) over every room in order, in one
    compiled call: the same draws from numba's process-global generator in the same order, so the same swaps."""
    for room in range(len(cvs_ptr) - 1):
        base = cvs_ptr[room]
        n = cvs_ptr[room + 1] - base
        pbase = cand_ptr_offset[room]
        ibase = cand_idx_offset[room]
        left = np.empty(n, dtype=np.int64)
        right = np.empty(n, dtype=np.int64)
        m = 0
        for i in range(n):
            if np.random.random() > p:
                continue
            start = cand_ptr[pbase + i]
            end = cand_ptr[pbase + i + 1]
            if end <= start:
                continue
            j = cand_idx[ibase + start + np.random.randint(end - start)]
            left[m] = i
            right[m] = j
            m += 1
        for i in range(m - 1, 0, -1):
            j = np.random.randint(i + 1)
            left[i], left[j] = left[j], left[i]
            right[i], right[j] = right[j], right[i]
        for k in range(m):
            a = base + left[k]
            b = base + right[k]
            r1 = cvs[a, 0]
            c1 = cvs[a, 1]
            r2 = cvs[b, 0]
            c2 = cvs[b, 1]
            held = temp[r1, c1]
            temp[r1, c1] = temp[r2, c2]
            temp[r2, c2] = held


class _BatchedConvection:
    """StochasticConvectionSimulator.apply_convection for p < 1 with a distance limit: the official rooms, order,
    candidate lists and numba kernel, called once per step instead of once per room."""

    def __init__(self, building):
        self.simulator = building.convection_simulator
        self.arrays = None
        sim = self.simulator
        if sim is None or not hasattr(sim, "_room_arrays") or sim.p == 0 or sim.distance in (0, -1):
            return
        rooms = [(k, v) for k, v in building.room_dict.items()
                 if k not in ("exterior_space", "interior_wall") and v]
        parts = [sim._room_arrays(k, v, sim.distance, False) for k, v in rooms]
        lengths = [len(c) for c, _, _ in parts]
        self.arrays = (
            np.concatenate([c for c, _, _ in parts]).astype(np.int64),
            np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
            np.concatenate([ptr for _, ptr, _ in parts]).astype(np.int64),
            np.concatenate([[0], np.cumsum([len(ptr) for _, ptr, _ in parts])]).astype(np.int64),
            np.concatenate([idx for _, _, idx in parts]).astype(np.int64),
            np.concatenate([[0], np.cumsum([len(idx) for _, _, idx in parts])]).astype(np.int64),
        )

    def apply(self, building):
        if self.arrays is None:
            building.apply_convection()
            return
        _convect_rooms(*self.arrays, float(self.simulator.p), building.temp)


def _execute_step_sim(self, video_filename="sample.mp4"):
    """SimulatorFlexibleGeometries.execute_step_sim with the thermal parts on the core (marked fastsim)."""
    thermal = getattr(self, "_fastsim", None)
    if thermal is None:
        return _ORIGINAL["execute_step_sim"](self, video_filename)
    air_handler_py, _, constants, sfg, _ = _modules()

    current_ts = self._current_timestamp
    hvac = self._hvac
    room_dict = self.building.room_dict
    room_areas = thermal.room_areas   # fastsim: fixed at construction (the official rebuilds the same dict)
    avg_temps = self.building.get_zone_average_temps()
    ambient_temperature = self._weather_controller.get_current_temp(current_ts)

    ahu_temp_weighted_sum = collections.defaultdict(float)
    ahu_area_total = collections.defaultdict(float)
    vav_cached_data = {}
    vav_sensing_temp_map = {}

    for v_id, vav in hvac.vavs.items():
        ahu_id = vav.air_handler.device_id()
        assigned_zones = [z for z in hvac.get_zones_for_vav(v_id) if z in avg_temps]
        if not assigned_zones:
            continue

        v_area = sum(room_areas[z] for z in assigned_zones)
        v_sensing_temp = sum(avg_temps[z] * room_areas[z] for z in assigned_zones) / v_area
        vav.update_settings(v_sensing_temp, current_ts)

        vav_cached_data[v_id] = sfg._VavZones(assigned_zones=assigned_zones, area=v_area,
                                              sensing_temp=v_sensing_temp)
        vav_sensing_temp_map[v_id] = v_sensing_temp

        ahu_temp_weighted_sum[ahu_id] += v_sensing_temp * v_area
        ahu_area_total[ahu_id] += v_area

    if isinstance(hvac.air_handler, air_handler_py.AirHandlerSystem):
        ahus = hvac.air_handler.ahus
    else:
        ahus = [hvac.air_handler]

    recirculation_temps = {}
    for ahu in ahus:
        ahu_id = ahu.device_id()
        if ahu_area_total.get(ahu_id, 0) > 0:
            recirculation_temps[ahu_id] = ahu_temp_weighted_sum[ahu_id] / ahu_area_total[ahu_id]
        else:
            recirculation_temps[ahu_id] = self.building.temp.mean()   # exact only (rom refuses such buildings)
    if _STATE.get("return_air") == "flow":                             # 2026-11-10: starter-kit PR #8 mixing
        recirculation_temps = _recirculation_temps_flow(self, avg_temps, room_areas)

    if isinstance(hvac.air_handler, air_handler_py.AirHandlerSystem):
        supply_air_temp = hvac.air_handler.get_supply_air_temp(recirculation_temps, ambient_temperature)
    else:
        supply_air_temp = hvac.air_handler.get_supply_air_temp(
            recirculation_temps[hvac.air_handler.device_id()], ambient_temperature)

    convection_coefficient = self._weather_controller.get_air_convection_coefficient(current_ts)

    hvac.air_handler.reset_demand()
    hvac.hot_water_system.reset_demand()

    for vav in hvac.vavs.values():
        hvac.hot_water_system.add_demand(vav.reheat_flow_factor)

    air_flow_demand = collections.defaultdict(float)
    for vav in hvac.vavs.values():
        air_flow_demand[vav.air_handler.device_id()] += vav.unrestricted_flow_rate_demand
    for vav in hvac.vavs.values():
        demanded = air_flow_demand[vav.air_handler.device_id()]
        if demanded > 0:
            vav.supply_air_flow_ratio = min(1.0, vav.air_handler.max_air_flow_rate / demanded)
        else:
            vav.supply_air_flow_ratio = 1.0

    zone_q_aggregate = collections.defaultdict(float)
    total_reheat_w = 0.0
    coil_entering_air_temps = []
    air_side_conductance = np.zeros(len(self._coupled_vavs))
    air_side_supply_temp = np.zeros(len(self._coupled_vavs))

    for column, v_id in enumerate(self._coupled_vavs):
        if v_id not in vav_cached_data:
            continue
        vav = hvac.vavs[v_id]
        cached_data = vav_cached_data[v_id]
        assigned_zones = cached_data.assigned_zones
        total_vav_area = cached_data.area

        if isinstance(supply_air_temp, dict):
            current_supply_temp = supply_air_temp[vav.air_handler.device_id()]
        else:
            current_supply_temp = supply_air_temp
        air_side_supply_temp[column] = current_supply_temp

        reheat_w = vav.compute_reheat_energy_rate(current_supply_temp,
                                                  hvac.hot_water_system.supply_water_temperature_sensor)
        total_reheat_w += reheat_w
        coil_entering_air_temps.append(current_supply_temp)

        if vav.damper_setting > 0 and vav.max_air_flow_rate > 0:
            air_side_conductance[column] = (vav.flow_rate_demand * constants.AIR_DENSITY
                                            * constants.AIR_HEAT_CAPACITY)
            for z in assigned_zones:
                zone_q_aggregate[z] += reheat_w * room_areas[z] / total_vav_area

        if vav.flow_rate_demand > 0:
            vav.air_handler.add_demand(vav.flow_rate_demand)

    if self._internal_gain_w_per_m2 > 0.0:
        fraction = 1.0 if hvac.is_comfort_mode(current_ts) else self._internal_gain_standby_fraction
        cv_area_m2 = (self.building.cv_size_cm / 100.0) ** 2
        watts_per_cv = self._internal_gain_w_per_m2 * cv_area_m2 * fraction
        for zone_name, coords in room_dict.items():
            if zone_name in ("exterior_space", "interior_wall"):
                continue
            zone_q_aggregate[zone_name] += watts_per_cv * len(coords)

    if self._occupancy is not None and self._occupant_sensible_heat_w > 0.0:
        interval_end = current_ts + pd.Timedelta(self._time_step_sec, unit="s")
        for zone_name in room_dict:
            if zone_name in ("exterior_space", "interior_wall"):
                continue
            occupants = self._occupancy.average_zone_occupancy(zone_name, current_ts, interval_end)
            zone_q_aggregate[zone_name] += occupants * self._occupant_sensible_heat_w

    for z, q_total in zone_q_aggregate.items():
        self.building.apply_thermal_power_zone(z, q_total)

    # fastsim: the thermal step on the core (the official finite_differences_timestep)
    grid = thermal.step(ambient_temperature, convection_coefficient, air_side_conductance, air_side_supply_temp)
    if grid is not None:
        self.building.temp = grid
    thermal.version += 1

    # fastsim: the sensed means of the settled field (the official _air_side_sensed.T @ temp)
    air_side_q = air_side_conductance * (air_side_supply_temp - thermal.sensed_means())
    for column, v_id in enumerate(self._coupled_vavs):
        cached_data = vav_cached_data.get(v_id)
        if cached_data is None:
            continue
        for z in cached_data.assigned_zones:
            zone_q_aggregate[z] += air_side_q[column] * room_areas[z] / cached_data.area
    for z, q_total in zone_q_aggregate.items():
        self.building.apply_thermal_power_zone(z, q_total)

    # fastsim: the convection shuffle on the full field (exact); the reduced model has no field to shuffle and
    # leaves it out (it moves air within a room and keeps every zone mean)
    if thermal.convection is not None:
        thermal.convection.apply(self.building)
        thermal.load_field(self.building.temp)

    hvac.hot_water_system.return_water_temperature_sensor = (
        self._calculate_return_water_temperature_from_reheat(total_reheat_w, coil_entering_air_temps))

    # fastsim: the divergence guard on what the core holds (the whole field, or the zone means)
    lowest, highest = thermal.bounds(ambient_temperature)
    if lowest < 150 or highest > 500:
        log.warning("Temperature field left physical bounds at %s: min %.1f K, max %.1f K.",
                    self._current_timestamp, lowest, highest)
        if thermal.kind == "exact":
            self.building.temp = np.clip(self.building.temp, 150, 500)
            thermal.load_field(self.building.temp)

    self._current_timestamp += pd.Timedelta(self._time_step_sec, unit="s")
    if self._capture_frames:
        self._log_and_plotter.log(self.building.temp)

    if (self._video_export_day is not None
            and self.current_timestamp == self._start_timestamp + pd.Timedelta(days=self._video_export_day)):
        self.get_video(path=constants.VIDEO_PATH_ROOT + video_filename)
