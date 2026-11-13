"""B1-R: the Guideline 36 style rule-based backbone (WP2 step 2.8).

G36Policy composes the modules of smart_building.g36; every tunable lives in
G36Config with the register ID of docs/plan/baseline-v1-build.md §3 that sets
it. Fields marked "A/B" are the switches of D-R06, D-R12, D-R13, D-R16, D-R18,
D-R21 and D-R22. Their defaults are the set of the first full evaluation and the
first submission (D-R24, REQ-008, approved 2026-10-16): Ti 120 s, suppression
on, static pressure trim & respond, G36 fan modes, group average; D-R12 and
D-R18 keep their registered main cases. Gaps settled on 2026-10-16 are tagged
with their new rows: D-R26 (modes, setpoints, cool-down supply air), D-R27 (loop
discretisation and estimates), D-R28 (reheat valve by inverting D-R09), D-R20
(economizer target floor). Quantities tied to the controller's own commands use
its own commands (D-R25 case a, pending REQ-012).

D-T03  Before returning, every action is checked for finite values (NaN passes
       the simulator's range check and spoils the rest of the episode,
       notes/01 §B, §G) and clipped to the action spec.
D-T02, D-R23
       If B1-R raises, or returns a non-finite action, the official pid_water
       action is used for that step and the failure is recorded (counted, the
       first traceback of each exception type logged). If building B1-R from
       the metadata fails, every step uses pid_water. The pid_water instance
       runs in shadow mode: it is called with every observation, so its state
       equals that of the official pid_water run along the actual trajectory
       (its integrals and previous errors are updated from observations only,
       pid.py:158-159, 241-259; the water integral stays 0 by default,
       pid_plus.py:482); its action is used only on a failing step
       (reviewer instruction 2026-10-16, notes/09).

Diagnostics for local checks (DR3): with the environment variable
SMART_BUILDING_G36_DIAGNOSTICS set to a directory, the policy writes its step
count, fallback count, per-step get_action wall times and mode counters there
every 288 steps. Unset (the default, and in submissions) nothing is written.
"""
from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import time
import traceback

import numpy as np

from smart_building.g36_v1 import ahu, plant, zone
from smart_building.g36_v1 import requests as zone_requests
from smart_building.g36_v1.building import BuildingIndex
from smart_building.g36_v1.modes import (MODE_NAMES, WARM_UP, SetbackSetup, is_occupied, operating_mode,
                                      zone_group_setpoints)
from smart_building.g36_v1.optimal_start import OptimalStart
from smart_building.g36_v1.requests import SetpointChangeSuppression
from smart_building.g36_v1.trim_respond import ignore_count

LOGGER = logging.getLogger(__name__)
DIAGNOSTICS_ENV = "SMART_BUILDING_G36_DIAGNOSTICS"
DUMP_EVERY = 288


@dataclass(frozen=True)
class G36Config:
    # D-R04 zone groups and operating modes
    setback_min_zones: int = 5                    # D-R04
    mode_release_margin_k: float = 1.0            # D-R04 (ZoneStatus.mo bouLim)
    # D-R05 optimal start (OptimalStart.mo:4-36, OptimalStartCalculation.mo:54-70)
    t_opt_max_s: float = 10800.0                  # D-R05 (D-P02 tunes 10800-21600)
    n_day: int = 3                                # D-R05
    u_low_k: float = 0.0                          # D-R05
    u_high_k: float = 0.5                         # D-R05
    initial_slope_k_per_s: float = 1.0 / 3600.0   # D-R05
    thr_opt_on_s: float = 60.0                    # D-R05 (thrOptOn, notes/04 D)
    # D-R06 zone loops
    zone_gain_per_k: float = 0.1                  # D-R06
    zone_ti_s: float = 120.0                      # D-R06 A/B: 900 (ControlLoops.mo) or 120 (Reheat/Controller.mo); D-R24: 120
    # D-R07 dual maximum
    discharge_offset_k: float = 11.0              # D-R07 (heating setpoint + 11 K)
    discharge_limit_k: float = zone.MAX_DISCHARGE_TEMPERATURE  # D-R07 (308.15 K, vav.py:79)
    discharge_zone_margin_k: float = 3.0          # D-R07
    valve_control: str = "inverse"                # D-R28: "inverse" (D-R09 equations solved for the valve); "pi" = D-R07 valve PI (WP3 ablation)
    valve_gain_per_k: float = 0.5                 # D-R07 (ablation)
    valve_ti_s: float = 300.0                     # D-R07 (ablation)
    # D-R08 airflow limits, fractions of the design flow at 2000 Pa
    min_flow: float = 0.1                         # D-R08 (damper floor 0.1)
    cooling_max_flow: float = 1.0                 # D-R08 (damper 1.0)
    heating_max_ratio: float = 0.5                # D-R08 (50 % of cooling maximum)
    # D-R10 request thresholds (SystemRequests.mo)
    cooling_three_k: float = 3.0                  # D-R10
    cooling_two_k: float = 2.0                    # D-R10
    cooling_loop_on: float = 0.95                 # D-R10
    cooling_loop_off: float = 0.85                # D-R10
    damper_on: float = 0.95                       # D-R10
    damper_off: float = 0.85                      # D-R10
    flow_three_below: float = 0.5                 # D-R10
    flow_two_below: float = 0.7                   # D-R10
    hot_water_three_k: float = 17.0               # D-R10
    hot_water_two_k: float = 8.0                  # D-R10
    valve_on: float = 0.95                        # D-R10
    valve_off: float = 0.85                       # D-R10
    plant_valve_on: float = 0.95                  # D-R10
    plant_valve_off: float = 0.10                 # D-R10
    # D-R12 ignore count of the air-side trim & respond loops
    ignore_override: int | None = None            # D-R12: None = round(0.1 x zones); ablation 0, 1, 2
    # D-R13 request suppression after setpoint changes
    suppression: bool = True                      # D-R13 A/B (on / off); D-R24: on
    suppression_s_per_k: float = 540.0            # D-R13
    suppression_max_s: float = 1800.0             # D-R13
    # D-R15 supply air temperature (D-R14 converts the per-2-min amounts)
    sat_min_k: float = 285.15                     # D-R15
    sat_max_k: float = 291.15                     # D-R15 (also SP0)
    sat_oat_low_k: float = 289.15                 # D-R15
    sat_oat_high_k: float = 294.15                # D-R15
    sat_heating_mode_k: float = 305.0             # D-R15 (warm-up / setback)
    sat_trim_k: float = 0.1                       # D-R15
    sat_respond_k: float = -0.2                   # D-R15
    sat_respond_max_k: float = -0.6               # D-R15
    sat_period_s: float = 120.0                   # D-R15, D-R14
    sat_delay_s: float = 600.0                    # D-R15, D-R14
    cooldown_sat_k: float | None = 285.15         # D-R26: cool-down TSupCoo_min; None = reset value (variant)
    # D-R16 static pressure
    static_pressure_reset: str = "trim_respond"   # D-R16 A/B: "trim_respond" (a) or "closed_form" (b); D-R24: (a)
    sp_initial_pa: float = 120.0                  # D-R16
    sp_min_pa: float = zone.RUNNING_MIN_STATIC_PRESSURE  # D-R16 (200 Pa)
    sp_max_pa: float = zone.DESIGN_STATIC_PRESSURE       # D-R16 (2000 Pa)
    sp_trim_pa: float = -12.0                     # D-R16
    sp_respond_pa: float = 15.0                   # D-R16
    sp_respond_max_pa: float = 32.0               # D-R16
    sp_period_s: float = 120.0                    # D-R16, D-R14
    sp_delay_s: float = 600.0                     # D-R16, D-R14
    # D-R17 supply water temperature
    hw_initial_k: float = 350.0                   # D-R17
    hw_min_k: float = 310.0                       # D-R17 (D-P02 tunes 310-341.45)
    hw_max_k: float = 350.0                       # D-R17
    hw_trim_k: float = -1.1                       # D-R17
    hw_respond_k: float = 1.7                     # D-R17
    hw_respond_max_k: float = 3.9                 # D-R17
    hw_period_s: float = 300.0                    # D-R17
    hw_delay_s: float = 600.0                     # D-R17
    hw_ignore: int = 2                            # D-R17
    hw_max_rise_k: float = 2.5                    # D-R17
    # D-R18 pump differential pressure
    pump_pressure: str = "official"               # D-R18 main "official"; ablation "plant_reset"
    dp_min_bar: float = 0.0                       # D-R18 (action range, config.gin:328-331)
    dp_max_bar: float = 20.0                      # D-R18
    plant_reset_initial: float = 1.0              # D-R18 ablation
    plant_reset_trim: float = -0.02               # D-R18 ablation
    plant_reset_respond: float = 0.03             # D-R18 ablation
    plant_reset_respond_max: float = 0.07         # D-R18 ablation
    plant_reset_period_s: float = 300.0           # D-R18 ablation
    plant_reset_delay_s: float = 900.0            # D-R18 ablation
    plant_reset_ignore: int = 2                   # D-R18 ablation
    # D-R20 economizer
    economizer_k_per_pa: float = 0.00322          # D-R20 (pid_plus.py:54-64)
    max_recirculation: float = 0.7                # D-R20 (official cap)
    # D-R21 / D-R22 fan on/off
    fan_control: str = "g36_modes"                # D-R21 A/B: "g36_modes" (a) or "official_pid" (b); D-R24: (a)
    ahu_zone_aggregate: str = "average"           # D-R22 A/B: "worst" or "average"; D-R24: average (inert under D-R21 a)


class G36Controller:
    """The B1-R sequences of one building (no exception handling here)."""

    def __init__(self, metadata, action_spec, config):
        self.cfg = cfg = config
        self.index = ix = BuildingIndex(metadata, action_spec)                # D-R01
        n = ix.n_zones
        sizes = ix.group_sizes
        ignore = [cfg.ignore_override if cfg.ignore_override is not None else ignore_count(s)
                  for s in sizes]                                           # D-R12
        heat_occ, cool_occ = ix.comfort_window
        self.optimal_start = OptimalStart(                                  # D-R05
            ix.zone_group, ix.group_slices, heating_setpoint=heat_occ, cooling_setpoint=cool_occ,
            morning_hour=ix.morning_start_hour, evening_hour=ix.evening_start_hour,
            t_opt_max=cfg.t_opt_max_s, n_day=cfg.n_day, u_low=cfg.u_low_k, u_high=cfg.u_high_k,
            initial_slope=cfg.initial_slope_k_per_s, thr_opt_on=cfg.thr_opt_on_s)
        self.setback_setup = SetbackSetup(ix.group_slices, min_zones=cfg.setback_min_zones,
                                          release_margin=cfg.mode_release_margin_k)  # D-R04
        self.heating_loop = zone.PILoop(n, cfg.zone_gain_per_k, cfg.zone_ti_s)       # D-R06
        self.cooling_loop = zone.PILoop(n, cfg.zone_gain_per_k, cfg.zone_ti_s)       # D-R06
        if cfg.valve_control not in ("inverse", "pi"):
            raise ValueError(f"unknown valve control {cfg.valve_control!r}")
        self.valve_loop = zone.PILoop(n, cfg.valve_gain_per_k, cfg.valve_ti_s)      # D-R07 (ablation)
        self.cooling_latch = np.zeros(n, dtype=bool)                                # D-R10
        self.pressure_latch = np.zeros(n, dtype=bool)
        self.hot_water_latch = np.zeros(n, dtype=bool)
        self.plant_latch = np.zeros(n, dtype=bool)
        self.suppression = SetpointChangeSuppression(                               # D-R13
            ix.n_ahu, enabled=cfg.suppression, seconds_per_kelvin=cfg.suppression_s_per_k,
            max_seconds=cfg.suppression_max_s)
        self.supply_air = ahu.SupplyAirTemperature(                                 # D-R15
            ignore, minimum=cfg.sat_min_k, maximum=cfg.sat_max_k, oat_low=cfg.sat_oat_low_k,
            oat_high=cfg.sat_oat_high_k, heating_mode=cfg.sat_heating_mode_k,
            cooldown=cfg.cooldown_sat_k, trim=cfg.sat_trim_k, respond=cfg.sat_respond_k,
            respond_max=cfg.sat_respond_max_k, period=cfg.sat_period_s, delay=cfg.sat_delay_s)
        self.static_pressure = ahu.StaticPressure(                                  # D-R16
            cfg.static_pressure_reset, ignore, initial=cfg.sp_initial_pa, minimum=cfg.sp_min_pa,
            maximum=cfg.sp_max_pa, trim=cfg.sp_trim_pa, respond=cfg.sp_respond_pa,
            respond_max=cfg.sp_respond_max_pa, period=cfg.sp_period_s, delay=cfg.sp_delay_s)
        self.water = plant.SupplyWaterTemperature(                                  # D-R17
            initial=cfg.hw_initial_k, minimum=cfg.hw_min_k, maximum=cfg.hw_max_k, trim=cfg.hw_trim_k,
            respond=cfg.hw_respond_k, respond_max=cfg.hw_respond_max_k, period=cfg.hw_period_s,
            delay=cfg.hw_delay_s, ignore=cfg.hw_ignore, max_rise=cfg.hw_max_rise_k)
        if cfg.pump_pressure not in ("official", "plant_reset"):
            raise ValueError(f"unknown pump pressure method {cfg.pump_pressure!r}")
        self.plant_reset = None
        if cfg.pump_pressure == "plant_reset":                                       # D-R18 ablation
            self.plant_reset = plant.PlantReset(
                dp_min=cfg.dp_min_bar, dp_max=cfg.dp_max_bar, temp_low=cfg.hw_min_k,
                temp_high=cfg.hw_max_k, initial=cfg.plant_reset_initial, trim=cfg.plant_reset_trim,
                respond=cfg.plant_reset_respond, respond_max=cfg.plant_reset_respond_max,
                period=cfg.plant_reset_period_s, delay=cfg.plant_reset_delay_s,
                ignore=cfg.plant_reset_ignore)
        self.previous_valve = None
        self.previous_plant_on = None
        self.counters = {name: 0 for name in (
            "steps", "fan_on_group_steps", "plant_on_steps", "plant_switches", "valve_jumps_over_0_9")}
        self.counters.update({f"mode_{name}_group_steps": 0 for name in MODE_NAMES})

    def act(self, obs):
        cfg, ix = self.cfg, self.index
        groups = ix.zone_group
        o = ix.parse(obs)                                                    # D-R01-D-R03
        temp = o.zone_temp
        occupied = is_occupied(o.minute_of_day, o.weekday, ix.morning_start_hour,
                               ix.evening_start_hour)                        # D-R04
        warm, cool = self.optimal_start.update(temp, o.minute_of_day, o.weekday)    # D-R05
        eco_heat, eco_cool = ix.eco_window
        setback, setup = self.setback_setup.update(temp, occupied, eco_heat, eco_cool)  # D-R04
        mode = operating_mode(occupied, cool, warm, setup, setback)                # D-R26
        # D-R26: setpoints follow the mode (Setpoints.mo:1286-1289): occupied,
        # warm-up, cool-down -> comfort pair; unoccupied, setback, setup -> eco pair.
        # With an empty metadata holiday set (all shipped scenarios) the occupied
        # mode coincides with the comfort flag of D-R03.
        heat_g, cool_g = zone_group_setpoints(mode, ix.comfort_window, ix.eco_window)
        heat_sp, cool_sp = heat_g[groups], cool_g[groups]

        heating = self.heating_loop.update(heat_sp - temp)                   # D-R06
        cooling = self.cooling_loop.update(temp - cool_sp)                   # D-R06

        # D-R09: discharge air temperature from the observed actuator state.
        # D-R27: return air = unweighted mean of the group's zones (the
        # simulator weights by area; the metadata has no zone areas).
        return_temp = np.array([temp[s].mean() for s in ix.group_slices])
        supply_est = zone.estimate_supply_air(return_temp, o.oat, o.recirculation, o.sat_setpoint,
                                              o.ahu_on)
        box_pressure = o.static_pressure[groups]
        ratio = zone.supply_flow_ratio(zone.box_flow_demand(o.damper, box_pressure), groups, ix.n_ahu)
        discharge_est = zone.estimate_discharge(o.damper, o.valve, box_pressure, ratio[groups],
                                                supply_est[groups], o.water_temp, o.water_dp)

        flow, discharge_sp = zone.dual_maximum(                               # D-R07, D-R08
            heating, cooling, supply_est[groups], heat_sp, temp, discharge_est,
            min_flow=cfg.min_flow, cooling_max_flow=cfg.cooling_max_flow,
            heating_max_flow=cfg.heating_max_ratio * cfg.cooling_max_flow,
            discharge_offset=cfg.discharge_offset_k, discharge_limit=cfg.discharge_limit_k,
            zone_margin=cfg.discharge_zone_margin_k)
        if cfg.valve_control == "inverse":                                  # D-R28
            valve = zone.reheat_valve_for(discharge_sp, o.damper, box_pressure, ratio[groups],
                                          supply_est[groups], o.water_temp, o.water_dp)
        else:
            valve = self.valve_loop.update(discharge_sp - discharge_est)     # D-R07 (ablation)

        # D-R10 / D-R13 requests. REQ-011 (1), approved: suppression applies to the 3 and 2
        # cooling SAT requests, timed from changes of the group's cooling setpoint.
        suppressed = self.suppression.update(cool_g)
        cooling_requests, self.cooling_latch = zone_requests.cooling_sat_requests(
            temp - cool_sp, cooling, self.cooling_latch, suppressed[groups],
            three_above=cfg.cooling_three_k, two_above=cfg.cooling_two_k,
            loop_on=cfg.cooling_loop_on, loop_off=cfg.cooling_loop_off)
        # Static pressure requests at the pressure in force before this step's reset.
        demanded = zone.damper_command(flow, self.static_pressure.setpoint[groups])  # D-R09
        pressure_requests, self.pressure_latch = zone_requests.static_pressure_requests(
            demanded, self.pressure_latch, damper_on=cfg.damper_on, damper_off=cfg.damper_off,
            three_below=cfg.flow_three_below, two_below=cfg.flow_two_below)
        hot_water_requests, self.hot_water_latch = zone_requests.hot_water_reset_requests(
            discharge_sp - discharge_est, valve, self.hot_water_latch,
            three_above=cfg.hot_water_three_k, two_above=cfg.hot_water_two_k,
            valve_on=cfg.valve_on, valve_off=cfg.valve_off)
        plant_requests, self.plant_latch = zone_requests.heating_plant_requests(
            valve, self.plant_latch, valve_on=cfg.plant_valve_on, valve_off=cfg.plant_valve_off)

        def per_group(values):
            return np.bincount(groups, weights=values, minlength=ix.n_ahu)

        error = ahu.group_errors(temp, ix.group_slices, heat_g, cool_g, cfg.ahu_zone_aggregate)  # D-R22
        fan_on = ahu.fan_command(cfg.fan_control, mode=mode, error=error)  # D-R21
        sat = self.supply_air.update(per_group(cooling_requests), fan_on, o.oat, mode=mode,
                                     error=error)                            # D-R15, D-R26
        pressure = self.static_pressure.update(per_group(pressure_requests), fan_on, flow,
                                               ix.group_slices)              # D-R16
        damper = zone.damper_command(flow, pressure[groups])                 # D-R09
        recirculation = ahu.economizer(return_temp, o.oat, sat, pressure, heat_g,
                                       k_per_pa=cfg.economizer_k_per_pa,
                                       max_recirculation=cfg.max_recirculation)  # D-R20

        plant_on = plant.plant_enable(int(plant_requests.sum()), bool(np.any(mode == WARM_UP)))  # D-R19
        if self.plant_reset is None:
            water_target = self.water.target(int(hot_water_requests.sum()), plant_on)  # D-R17
            dp = plant.official_pump_pressure(bool(fan_on.any()), dp_min=cfg.dp_min_bar,
                                              dp_max=cfg.dp_max_bar)          # D-R18 main
        else:
            dp, water_target = self.plant_reset.update(int(hot_water_requests.sum()), plant_on)  # D-R18
        water = self.water.limit_rise(water_target, o.water_setpoint)       # D-R17

        native = np.empty(ix.n_continuous)
        native[ix.sat_action] = sat
        native[ix.sp_action] = pressure
        native[ix.recirculation_action] = recirculation
        native[ix.water_temp_action] = water
        native[ix.water_dp_action] = dp
        native[ix.damper_action] = damper
        native[ix.valve_action] = valve
        discrete = np.zeros(ix.n_ahu + 1, dtype=np.int32)
        discrete[ix.ahu_run_action] = fan_on
        discrete[ix.hws_run_action] = int(plant_on)

        c = self.counters
        c["steps"] += 1
        c["fan_on_group_steps"] += int(fan_on.sum())
        c["plant_on_steps"] += int(plant_on)
        c["plant_switches"] += int(self.previous_plant_on is not None and plant_on != self.previous_plant_on)
        for g_mode in mode:
            c[f"mode_{MODE_NAMES[g_mode]}_group_steps"] += 1
        if self.previous_valve is not None:
            c["valve_jumps_over_0_9"] += int(np.sum(np.abs(valve - self.previous_valve) > 0.9))
        self.previous_valve = valve
        self.previous_plant_on = plant_on
        return {"continuous_action": ix.normalize(native), "discrete_action": discrete}


class G36Policy:
    """B1-R with the competition interface; registered as POLICIES["b1_r"]."""

    name = "b1_r"

    def __init__(self, building_meta_data, action_spec, config=None):
        started = time.perf_counter()
        self.config = config if config is not None else G36Config()
        continuous = action_spec["continuous_action"]
        discrete = action_spec["discrete_action"]
        self._shapes = (tuple(continuous.shape), tuple(discrete.shape))
        self._continuous_bounds = (np.asarray(continuous.low, dtype=float),
                                   np.asarray(continuous.high, dtype=float))
        self._discrete_bounds = (np.asarray(discrete.low), np.asarray(discrete.high))
        self.diagnostics = {"steps": 0, "fallback_steps": 0, "nonfinite_steps": 0,
                            "clipped_values": 0, "invalid_final_actions": 0, "errors": {},
                            "construction_error": None, "config": asdict(self.config)}
        self._durations = []
        self._logged = set()
        sink = os.environ.get(DIAGNOSTICS_ENV)
        self._sink = Path(sink) if sink else None
        self._fallback = None
        self._fallback_error = None
        try:
            # D-R23: official pid_water, used on any step where B1-R fails.
            from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
            self._fallback = PIDWaterResetPolicy(building_meta_data, action_spec)
        except Exception as error:  # noqa: BLE001 - recorded; only fatal if a fallback is needed
            self._fallback_error = error
        self._shadow_action = None
        self.diagnostics["shadow_errors"] = 0
        self._controller = None
        try:
            self._controller = G36Controller(building_meta_data, action_spec, self.config)
        except Exception as error:  # noqa: BLE001 - D-R23: B1-R unavailable, pid_water every step
            self.diagnostics["construction_error"] = "".join(traceback.format_exception(error))
            LOGGER.warning("B1-R could not be built; pid_water is used for every step:\n%s",
                           self.diagnostics["construction_error"])
        self.diagnostics["init_seconds"] = time.perf_counter() - started

    # -- D-T03 ------------------------------------------------------------

    def _finalize(self, action):
        """Check shapes and finite values, then clip to the action spec (D-T03)."""
        continuous = np.asarray(action["continuous_action"], dtype=float)
        discrete = np.asarray(action["discrete_action"])
        if (continuous.shape, discrete.shape) != self._shapes:
            raise ValueError(f"action shapes {continuous.shape}, {discrete.shape} != {self._shapes}")
        if not (np.all(np.isfinite(continuous)) and np.all(np.isfinite(discrete))):
            raise FloatingPointError("non-finite action")
        low, high = self._continuous_bounds
        clipped = np.clip(continuous, low, high)
        self.diagnostics["clipped_values"] += int(np.sum(clipped != continuous))
        return {"continuous_action": clipped.astype(np.float32),
                "discrete_action": np.clip(np.rint(discrete), *self._discrete_bounds).astype(np.int32)}

    def _record_error(self, error):
        kind = type(error).__name__
        errors = self.diagnostics["errors"]
        errors[kind] = errors.get(kind, 0) + 1
        if isinstance(error, FloatingPointError):
            self.diagnostics["nonfinite_steps"] += 1
        if kind not in self._logged:
            self._logged.add(kind)
            LOGGER.warning("B1-R failed at step %d; pid_water used (D-R23):\n%s",
                           self.diagnostics["steps"], "".join(traceback.format_exception(error)))

    def _shadow(self, obs):
        """D-R23: advance the pid_water shadow with this observation."""
        self._shadow_action = None
        if self._fallback is None:
            return
        try:
            self._shadow_action = self._fallback.get_action(obs)
        except Exception as error:  # noqa: BLE001 - only fatal on a step where B1-R also fails
            self.diagnostics["shadow_errors"] += 1
            self._fallback_error = error

    def _fallback_action(self):
        if self._shadow_action is None:
            raise RuntimeError("B1-R failed and pid_water is unavailable") from self._fallback_error
        self.diagnostics["fallback_steps"] += 1
        return self._finalize(self._shadow_action)

    def get_action(self, obs):
        started = time.perf_counter()
        self._shadow(obs)
        action = None
        if self._controller is not None:
            try:
                action = self._finalize(self._controller.act(obs))
            except Exception as error:  # noqa: BLE001 - D-T02 / D-R23
                self._record_error(error)
        if action is None:
            action = self._fallback_action()
        elapsed = time.perf_counter() - started
        d = self.diagnostics
        d["steps"] += 1
        low, high = self._continuous_bounds
        c = action["continuous_action"]
        if not (np.all(np.isfinite(c)) and np.all(c >= low) and np.all(c <= high)):
            d["invalid_final_actions"] += 1
        self._durations.append(elapsed)
        if self._sink is not None and d["steps"] % DUMP_EVERY == 0:
            self._dump()
        return action

    def _dump(self):
        durations = np.asarray(self._durations)
        summary = dict(self.diagnostics)
        if self._controller is not None:
            summary["counters"] = dict(self._controller.counters)
        summary["get_action_seconds"] = {
            "mean": float(durations.mean()), "p50": float(np.percentile(durations, 50)),
            "p99": float(np.percentile(durations, 99)), "max": float(durations.max()),
            "first_step": float(durations[0])}
        self._sink.mkdir(parents=True, exist_ok=True)
        stem = f"g36-{os.getpid()}-{id(self):x}"
        np.save(self._sink / f"{stem}-durations.npy", durations)
        with open(self._sink / f"{stem}.json", "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(summary, indent=2) + "\n")
