"""B1-R second version (b1_r_v2): the Guideline 36 style rule-based backbone (WP2 step 2.8).

The first version (b1_r, the first submission, ref 57242800) is frozen byte for
byte, apart from import paths, in smart_building.policies.g36_v1 and
smart_building.g36_v1. This module carries only b1_r_v2 and the registered WP3
variant switches; G36Config's defaults are b1_r_v2 (no switches for old
behaviour).

respond, group average) with the corrections of 2026-10-16:
variant, decision 19), and the same-step supply air of the inversion and the
"observed" is b1_r_v2 as frozen at 12:20).

actuator state back from the observation (equal to the last command); the
uses this step's commands.

       action, the official pid_water action is used for that step; pid_water
       runs in shadow mode on every observation.

b1_r_v3 = b1_r_v2 with plant_lockout_k None: no
outside air lockout of the heating plant; G36Config's defaults stay b1_r_v2.

(V3_CANDIDATES), as fixed-name G36PolicyV3 subclasses in this module, so that a
remote job or a submission notebook needs exactly the files of b1_r_v3:
                               enable / disable or reset (loop_activation
                               provision G36 section 5.3.4 allows; the zone
                               "same_step"), rebased from b1_r_v2_same_step_supply
                               b1_r_v2_dr06_ti_900

C3 = b1_r_v3_ti900 became the B1-R champion at G1 (submitted as ref 57299256). The
WP3 discrete variants on C3 are C3_VARIANTS, fixed-name

("g36_table" default | "occupied_row") and table_zero_airflow ("design_fraction" default |
"damper_floor"); the defaults keep every earlier policy bit for bit. The combination
candidate on C3 is C3_COMBINATIONS, fixed-name classes registered as b1_r_c3_<combination>:
b1_r_c3_laa_occrow_dampfloor = C3 + loops always active + occupied_row + damper_floor.

switches for setback and warm-up; the defaults keep every earlier policy bit for bit:
                            308.15 K) | "supply_floor" (WARM_UP as well; rejected, kept to reproduce the diagnosis)
                            groups the zones outside the heating state reheat the forced 0.1 airflow to 308.15 K)
                            | "neutral" (alpha: to the zone temperature)
                            (modes.SetbackSetupZoneHysteresis, the MBL per-zone hysteresis and latch)
b1_r_c3_laa_occrow_dampfloor_cold3 = v4 + "supply_floor_setback" + "supply" + "zone_hysteresis".

Diagnostics for local checks: with SMART_BUILDING_G36_DIAGNOSTICS set to a
directory, the policy writes its counters and per-step get_action times there
every 288 steps; unset (the default, and in submissions) nothing is written.
The counters include zone-steps outside the comfort window while the building
has occupants (observation num_occupants > 0, the interval just ended); they do
not feed back into the action."""
from collections import deque
import itertools
from dataclasses import asdict, dataclass, replace
import json
import logging
import os
from pathlib import Path
import time
import traceback

import numpy as np

from smart_building.g36 import ahu, plant, zone
from smart_building.g36 import requests as zone_requests
from smart_building.g36.building import BuildingIndex
from smart_building.g36.modes import (COOL_DOWN, MODE_NAMES, OCCUPIED, SETBACK, UNOCCUPIED, WARM_UP, FreezeProtection,
                                      SetbackSetup, SetbackSetupZoneHysteresis, is_occupied, operating_mode,
                                      seconds_to_next_occupied, zone_group_setpoints)
from smart_building.g36.optimal_start import OptimalStart
from smart_building.g36.requests import SetpointChangeSuppression
from smart_building.g36.trim_respond import TrimRespond, ignore_count

LOGGER = logging.getLogger(__name__)
DIAGNOSTICS_ENV = "SMART_BUILDING_G36_DIAGNOSTICS"
DUMP_EVERY = 288
_DUMP_SERIAL = itertools.count()
SETBACK_WARMUP_DISCHARGE = ("ramp", "supply_floor_setback", "supply_floor")
SETBACK_WARMUP_DEADBAND = ("valve_off", "supply", "neutral")
FREEZE_HEAT_MODES = ("rule", "reset_below_off", "reset_below_off_valve_off")
SETBACK_RELEASE = ("all_zones_now", "zone_hysteresis")
HW_SAWTOOTH = ("off", "band", "saw", "saw_pump_off")
HW_OFF_SETPOINT = ("max", "hold", "low_warm")
PUMP_DP_RESET = ("official", "trim_respond", "valve_inverse")
NIGHT_HEAT = ("off", "weekday", "always", "free_gas")
NIGHT_COOL = ("off", "night", "tree")


@dataclass(frozen=True)
class G36Config:
    setback_min_zones: int = 5
    mode_release_margin_k: float = 1.0
    setback_release: str = "all_zones_now"
    setback_status_band_k: float = 0.1
    freeze_on_k: float = 277.15
    freeze_off_k: float = 280.15
    freeze_band_k: float = 0.1
    unoccupied_conditioning: str = "always"
    unoccupied_lead_min: int = 120
    occupied_margin_k: float = 0.0
    economizer_mode: str = "official"
    occupied_cycling: bool = False
    cycle_charge_k: float = 0.8
    cycle_on_k: float = 0.25
    cycle_off_k: float = 0.6
    cycle_min_on_steps: int = 3
    cycle_window_min: int = 0
    evening_charge_min: int = 0
    evening_charge_k: float = 2.5
    night_flush: bool = False
    empty_day_delay_min: int = 0
    dcv_area_share: float = 1.0
    night_flush_window_s: float = 8 * 3600.0
    flush_delta_k: float = 3.0
    flush_damper: float = 0.6
    precool_until_hour: int = 0
    precool_k: float = 1.0
    sat_cost_optimal: bool = False
    sat_cost_max_k: float = 293.15
    sat_cost_guard_k: float = 0.0
    sat_cost_respect_requests: bool = False
    sat_cost_guard_ignore: int = 0
    zone_heat_ext_power: float = 0.0
    sat_cost_guard_large_q: float = 0.0
    zone_heat_redistribute: bool = False
    zone_heat_boost_max: float = 1.0
    zone_heat_boost_step: float = 0.05
    zone_min_ra_lps_m2: float = 0.0
    zone_min_rp_lps_m2: float = 0.0
    zone_min_scope: str = "all"
    zone_min_oa_divide: bool = False
    zone_heat_wall_ref_m: float = 0.0
    zone_heat_wall_power: float = 0.5
    opt_start_break_max_s: float = 0.0
    opt_start_continuous: bool = False
    opt_start_zone_stagger: bool = False
    arrival_hold_min: int = 0
    arrival_hold_k: float = 0.3
    ahu_heat_budget: float = 1.0
    ahu_heat_budget_hours: float = 3.0
    ahu_heat_budget_oat_k: float = 263.15
    ahu_heat_budget_scope: str = "occupied"
    setback_min_zone_frac: float = 0.0
    discharge_min_winter_k: float = 0.0
    cycle_vent_ledger: bool = False
    vent_height_m: float = 3.0
    vent_ra_lps_m2: float = 0.3
    vent_rp_lps_m2: float = 0.125
    vent_ez: float = 0.8
    vent_ev: float = 0.75
    vent_max_off_frac: float = 1.0 / 3.0
    midday_charge_k: float = 0.0
    midday_charge_start_hour: int = 11
    midday_charge_end_hour: int = 15
    midday_charge_oat_k: float = 273.15
    sp_max_setback_pa: float = 0.0
    sp_rogue_hours: float = 0.0
    sp_rogue_share: float = 0.7
    sat_cost_model: str = "v1"
    sat_chiller_lockout_k: float = 0.0
    sat_cost_model_min_oat_k: float = 0.0
    zone_temp_lead: float = 0.0
    night_flush_min_range_k: float = 0.0
    night_flush_min_max_k: float = 0.0
    precool_min_range_k: float = 0.0
    precool_min_max_k: float = 0.0
    opt_start_break_min_oat_k: float = 0.0
    freeze_min_airflow: bool = False
    ahu_vot_check: bool = False
    vot_rp_lps_person: float = 2.5
    vot_ra_lps_m2: float = 0.3
    cycle_no_heating: bool = False
    sp_options_max_oat_k: float = 0.0
    zone_temp_lead_steady: bool = False
    zone_temp_lead_max_k: float = 0.0
    cold_low_flow_oat_k: float = 0.0
    cold_low_flow_sp_pa: float = 200.0
    cold_low_flow_heat_max_scale: float = 1.0
    cold_low_flow_deadband_only: bool = False
    cold_low_flow_margin_k: float = 0.0
    hw_sawtooth: str = "off"
    hw_saw_refill_below_k: float = 340.15
    hw_saw_period: int = 6
    hw_off_setpoint: str = "max"
    hw_off_low_after_s: float = 3600.0
    hw_off_low_min_oat_k: float = 291.15
    no_reheat_oat_k: float = 0.0
    setback_heat_drop_k: float = 0.0
    pump_dp_reset: str = "official"
    pump_dp_min_bar: float = 2.0
    pump_dp_ignore: int = 2
    pump_dp_valve_target: float = 0.9
    warmup_target_add_k: float = 0.0
    freeze_coldest_only: bool = False
    freeze_hold_k: float = 0.5
    freeze_heat_mode: str = "rule"
    warmup_deadband_neutral_above_k: float = 0.0
    big_room_heat_max_ratio: float = 0.0
    big_room_min_area_m2: float = 250.0
    big_room_until_hour: int = 9
    big_room_oat24_k: float = 278.15
    big_room_need: bool = False
    sat_cost_fan_observed: bool = False
    margin_feedback_step_k: float = 0.0
    margin_feedback_max_k: float = 0.3
    sat_guard_trend_s: float = 0.0
    sat_guard_trend_k: float = 0.0
    sat_rate_limit_k: float = 0.0
    sat_guard_hysteresis_k: float = 0.0
    night_heat_mode: str = "off"
    night_heat_oat_k: float = 278.15
    night_cool_mode: str = "off"
    night_cool_oat_k: float = 293.15
    warmup_heat_max_ratio: float = 0.0
    arrival_keep_flow_min: int = 0
    sat_guard_rogue_hours: float = 0.0
    sat_guard_rogue_share: float = 0.7
    sat_cost_max_min_range_k: float = 0.0
    humid_low_sat_k: float = 0.0
    humid_max_range_k: float = 8.0
    economizer_hysteresis_k: float = 0.0
    sp_damper_pi: bool = False
    sp_pi_min_pa: float = 50.0
    sp_pi_max_pa: float = 410.0
    sp_smooth_alpha: float = 0.0
    low_load_max_loop: float = 0.0
    heating_ramp_from_supply: bool = False
    dcv_area_share_auto: bool = False
    cooldown_learn_step_k: float = 0.0
    cooldown_learn_max_k: float = 2.0
    return_air_area_weighted: bool = False
    hw_request_rogue: str = "off"
    hw_rogue_share: float = 0.7
    opt_start_fit: str = "slope"
    disturbance_ff_steps: float = 0.0
    disturbance_ff_cool_k: float = 0.0
    disturbance_forget: float = 0.995
    economic_target_k: float = 0.0
    economic_keep_k: float = 0.3
    economic_keep_cold_k: float = 0.0
    economic_keep_cold_oat24_k: float = 263.15
    empty_day_setup: str = "rule"
    empty_day_setup_lead_min: int = 660
    departure_learn: bool = False
    departure_default_min: float = 60.0
    arrival_learn: bool = False
    arrival_days: int = 5
    arrival_min_days: int = 2
    arrival_quantile: float = 0.0
    capacity_start_extra_s: float = 0.0
    start_offset_s: float = 0.0
    daily_start: bool = False
    daily_layer: bool = False
    daily_layer_parts: str = "all"
    daily_min_oat24_k: float = 0.0
    hourly_lp: bool = False
    hourly_lp_horizon_h: int = 12
    hourly_lp_every_steps: int = 12
    hourly_lp_time_limit_s: float = 0.5
    hourly_lp_comfort_weight: float = 100.0
    disturbance_highpass_steps: float = 0.0
    hw_saw_resume_dp: bool = False
    capacity_start_heat_only: bool = False
    arrival_interval_start: bool = False
    no_reheat_min_oat24_k: float = 0.0
    night_heat_scale: float = 1.0
    night_heat_max_h: float = 0.0
    night_heat_stop_k: float = 0.6
    night_cool_scale: float = 1.0
    night_cool_max_h: float = 0.0
    night_cool_floor_k: float = 0.0
    arrival_keep_below_only: bool = False
    sat_guard_trend_one_sided: bool = False
    sat_guard_keep_margin_k: float = 0.0
    arrival_keep_ramp: bool = False
    t_opt_max_s: float = 10800.0
    n_day: int = 3
    u_low_k: float = 0.0
    u_high_k: float = 0.5
    slope_floor_k_per_s: float = 1.0 / 3600.0
    thr_opt_on_s: float = 60.0
    zone_gain_per_k: float = 0.1
    zone_ti_s: float = 120.0
    anti_windup_ni: float = 0.9
    loop_enable_hysteresis_k: float = 0.25
    loop_near_zero: float = 0.01
    loop_near_zero_hysteresis: float = 0.008
    loop_reset_output: float = 0.0
    loop_activation: str = "enable_disable"
    command_readback: str = "own"
    zone_state_threshold: float = 0.01
    cooling_state_hysteresis: float = 0.005
    heating_state_hysteresis: float = 0.008
    zone_state_hold_steps: int = 2
    discharge_offset_k: float = 11.0
    discharge_limit_k: float = zone.MAX_DISCHARGE_TEMPERATURE
    discharge_zone_margin_k: float = 3.0
    discharge_gate_hysteresis_k: float = 0.25
    warm_supply_threshold_k: float = 0.25
    warm_supply_hysteresis_k: float = 0.125
    discharge_min_k: float = 283.15
    discharge_min_hysteresis_k: float = 0.25
    setback_warmup_discharge: str = "ramp"
    warmup_setback_supply_k: float = 308.15
    setback_warmup_deadband: str = "valve_off"
    valve_control: str = "inverse"
    plant_off_valve_inputs: str = "observed"
    inversion_supply_air: str = "observed"
    valve_gain_per_k: float = 0.5
    valve_ti_s: float = 300.0
    min_flow: float = 0.1
    cooling_max_flow: float = 1.0
    heating_max_ratio: float = 0.5
    setback_warmup_heating_airflow: str = "g36_table"
    table_zero_airflow: str = "design_fraction"
    cooling_three_k: float = 3.0
    cooling_two_k: float = 2.0
    cooling_loop_on: float = 0.95
    cooling_loop_off: float = 0.85
    damper_on: float = 0.95
    damper_off: float = 0.85
    flow_three_below: float = 0.5
    flow_two_below: float = 0.7
    hot_water_three_k: float = 17.0
    hot_water_two_k: float = 8.0
    valve_on: float = 0.95
    valve_off: float = 0.85
    plant_valve_on: float = 0.95
    plant_valve_off: float = 0.10
    ignore_override: int | None = None
    suppression_s_per_k: float = 540.0
    suppression_max_s: float = 1800.0
    suppression_threshold_k: float = 0.25
    sat_min_k: float = 285.15
    sat_max_k: float = 291.15
    sat_oat_low_k: float = 289.15
    sat_oat_high_k: float = 294.15
    sat_heating_mode_k: float = 305.0
    sat_trim_k: float = 0.1
    sat_respond_k: float = -0.2
    sat_respond_max_k: float = -0.6
    sat_period_s: float = 120.0
    sat_delay_s: float = 600.0
    cooldown_sat_k: float = 285.15
    static_pressure_reset: str = "trim_respond"
    sp_initial_pa: float = 120.0
    sp_min_pa: float = zone.RUNNING_MIN_STATIC_PRESSURE
    sp_max_pa: float = zone.DESIGN_STATIC_PRESSURE
    sp_trim_pa: float = -12.0
    sp_respond_pa: float = 15.0
    sp_respond_max_pa: float = 32.0
    sp_period_s: float = 120.0
    sp_delay_s: float = 600.0
    hw_initial_k: float = 350.0
    hw_min_k: float = 310.0
    hw_max_k: float = 350.0
    hw_trim_k: float = -1.1
    hw_respond_k: float = 1.7
    hw_respond_max_k: float = 3.9
    hw_period_s: float = 300.0
    hw_delay_s: float = 600.0
    hw_ignore: int = 2
    hw_max_rise_k: float = 2.5
    pump_pressure: str = "official"
    dp_min_bar: float = 0.0
    dp_max_bar: float = 20.0
    plant_reset_initial: float = 1.0
    plant_reset_trim: float = -0.02
    plant_reset_respond: float = 0.03
    plant_reset_respond_max: float = 0.07
    plant_reset_period_s: float = 300.0
    plant_reset_delay_s: float = 900.0
    plant_reset_ignore: int = 2
    plant_min_off_steps: int = 3
    plant_min_on_steps: int = 3
    plant_ignore_requests: int = 0
    plant_lockout_k: float | None = plant.HW_LOCKOUT_K
    plant_lockout_hysteresis_k: float = plant.HW_LOCKOUT_HYSTERESIS_K
    economizer_k_per_pa: float = 0.00322
    max_recirculation: float = 0.7
    official_kp: float = 0.5
    official_ki: float = 0.02
    official_kd: float = 0.1
    official_integral_limit: float = 10.0
    fan_control: str = "g36_modes"
    ahu_zone_aggregate: str = "average"


def inversion_water_dp(method, plant_on_observed, observed_dp, commanded_dp):
    """    (A) "observed": the measured pressure, 0 while the plant is off, so a box that
    needs heat asks for a full valve and a plant request (no water flow: the
    would give (WP3 variant, decision 19)."""
    if method == "observed" or plant_on_observed:
        return observed_dp
    if method == "commanded":
        return commanded_dp
    raise ValueError(f"unknown plant-off valve inputs {method!r}")


class G36Controller:
    """The b1_r_v2 sequences of one building (no exception handling here)."""

    COUNTERS = ("steps", "fan_on_group_steps", "plant_on_steps", "plant_switches", "valve_jumps_over_0_9",
                "discharge_gate_toggles", "loop_enable_edges", "loop_masked_zone_steps", "zone_state_changes",
                "freeze_protection_group_steps", "low_discharge_reheat_zone_steps", "low_discharge_reheat_switches",
                "unoccupied_zero_zone_steps", "warm_supply_min_flow_zone_steps", "suppression_starts",
                "economizer_gate_open_group_steps", "economizer_active_group_steps", "slope_updates",
                "occupied_steps", "occupied_zone_steps_above_comfort", "occupied_zone_steps_below_comfort",
                "occupied_kelvin_steps_above_comfort", "occupied_kelvin_steps_below_comfort")

    def __init__(self, metadata, action_spec, config):
        self.cfg = cfg = config
        self.index = ix = BuildingIndex(metadata, action_spec)
        n = ix.n_zones
        if cfg.valve_control not in ("inverse", "pi"):
            raise ValueError(f"unknown valve control {cfg.valve_control!r}")
        if cfg.plant_off_valve_inputs not in ("observed", "commanded"):
            raise ValueError(f"unknown plant-off valve inputs {cfg.plant_off_valve_inputs!r}")
        if cfg.pump_pressure not in ("official", "plant_reset"):
            raise ValueError(f"unknown pump pressure method {cfg.pump_pressure!r}")
        if cfg.economizer_mode not in ("official", "supply_air"):
            raise ValueError(f"unknown economizer mode {cfg.economizer_mode!r}")
        if cfg.inversion_supply_air not in ("observed", "same_step"):
            raise ValueError(f"unknown inversion supply air {cfg.inversion_supply_air!r}")
        if cfg.loop_activation not in ("enable_disable", "always_active"):
            raise ValueError(f"unknown loop activation {cfg.loop_activation!r}")
        if cfg.command_readback not in ("own", "observed"):
            raise ValueError(f"unknown command readback {cfg.command_readback!r}")
        if cfg.setback_warmup_heating_airflow not in zone.SETBACK_WARMUP_HEATING_AIRFLOW:
            raise ValueError(f"unknown setback / warm-up heating airflow {cfg.setback_warmup_heating_airflow!r}")
        if cfg.table_zero_airflow not in zone.TABLE_ZERO_AIRFLOW:
            raise ValueError(f"unknown table zero airflow {cfg.table_zero_airflow!r}")
        if cfg.setback_warmup_discharge not in SETBACK_WARMUP_DISCHARGE:
            raise ValueError(f"unknown setback / warm-up discharge {cfg.setback_warmup_discharge!r}")
        if cfg.setback_warmup_deadband not in SETBACK_WARMUP_DEADBAND:
            raise ValueError(f"unknown setback / warm-up deadband {cfg.setback_warmup_deadband!r}")
        if cfg.freeze_heat_mode not in FREEZE_HEAT_MODES:
            raise ValueError(f"unknown freeze heat mode {cfg.freeze_heat_mode!r}")
        if cfg.setback_release not in SETBACK_RELEASE:
            raise ValueError(f"unknown setback release {cfg.setback_release!r}")
        if cfg.sat_cost_model not in ("v1", "economizer"):
            raise ValueError(f"unknown cost-based SAT model {cfg.sat_cost_model!r}")
        if cfg.ahu_heat_budget_scope not in ("occupied", "prestart"):
            raise ValueError(f"unknown heating budget scope {cfg.ahu_heat_budget_scope!r}")
        if cfg.hw_sawtooth not in HW_SAWTOOTH:
            raise ValueError(f"unknown hot-water sawtooth {cfg.hw_sawtooth!r}")
        if cfg.hw_off_setpoint not in HW_OFF_SETPOINT:
            raise ValueError(f"unknown plant-off setpoint {cfg.hw_off_setpoint!r}")
        if cfg.pump_dp_reset not in PUMP_DP_RESET:
            raise ValueError(f"unknown pump pressure reset {cfg.pump_dp_reset!r}")
        if cfg.night_heat_mode not in NIGHT_HEAT or cfg.night_cool_mode not in NIGHT_COOL:
            raise ValueError(f"unknown night conditioning {cfg.night_heat_mode!r} / {cfg.night_cool_mode!r}")
        ignore = [cfg.ignore_override if cfg.ignore_override is not None else ignore_count(s)
                  for s in ix.group_sizes]
        heat_occ, cool_occ = ix.comfort_window[0] + cfg.occupied_margin_k, ix.comfort_window[1] - cfg.occupied_margin_k
        self.occupied_targets = (heat_occ, cool_occ)
        cool_down_target = ix.comfort_window[0] + cfg.precool_k if cfg.precool_until_hour > 0 else cool_occ
        self.optimal_start = OptimalStart(
            ix.zone_group, ix.group_slices, heating_setpoint=heat_occ, cooling_setpoint=cool_down_target,
            morning_hour=ix.morning_start_hour, evening_hour=ix.evening_start_hour,
            t_opt_max=cfg.t_opt_max_s, n_day=cfg.n_day, u_low=cfg.u_low_k, u_high=cfg.u_high_k,
            slope_floor=cfg.slope_floor_k_per_s, thr_opt_on=cfg.thr_opt_on_s,
            continuous=cfg.opt_start_continuous,
            fit=cfg.opt_start_fit)
        setback_zones = cfg.setback_min_zones
        if cfg.setback_min_zone_frac > 0.0:
            setback_zones = [max(1, int(round(cfg.setback_min_zone_frac * size))) for size in ix.group_sizes]
        if cfg.setback_release == "zone_hysteresis":
            self.setback_setup = SetbackSetupZoneHysteresis(
                ix.group_slices, n, min_zones=setback_zones, release_margin=cfg.mode_release_margin_k,
                band=cfg.setback_status_band_k)
        else:
            self.setback_setup = SetbackSetup(ix.group_slices, min_zones=setback_zones,
                                              release_margin=cfg.mode_release_margin_k)
        self.freeze_protection = FreezeProtection(ix.group_slices, on=cfg.freeze_on_k, off=cfg.freeze_off_k,
                                                  band=cfg.freeze_band_k)
        self.zone_loops = zone.ZoneLoops(
            n, cfg.zone_gain_per_k, cfg.zone_ti_s, hysteresis=cfg.loop_enable_hysteresis_k,
            near_zero=cfg.loop_near_zero, near_zero_hysteresis=cfg.loop_near_zero_hysteresis,
            reset_output=cfg.loop_reset_output, ni=cfg.anti_windup_ni,
            always_active=cfg.loop_activation == "always_active")
        self.zone_states = zone.ZoneStates(
            n, threshold=cfg.zone_state_threshold, cooling_hysteresis=cfg.cooling_state_hysteresis,
            heating_hysteresis=cfg.heating_state_hysteresis, hold_steps=cfg.zone_state_hold_steps)
        self.valve_loop = zone.PILoop(n, cfg.valve_gain_per_k, cfg.valve_ti_s, ni=cfg.anti_windup_ni)
        self.discharge_gate = np.zeros(n, dtype=bool)
        self.warm_supply = np.zeros(n, dtype=bool)
        self.low_discharge = np.zeros(n, dtype=bool)
        self.low_discharge_reheat = np.zeros(n, dtype=bool)
        self.cooling_latch = np.zeros(n, dtype=bool)
        self.pressure_latch = np.zeros(n, dtype=bool)
        self.hot_water_latch = np.zeros(n, dtype=bool)
        self.plant_latch = np.zeros(n, dtype=bool)
        self.suppression = SetpointChangeSuppression(
            n, seconds_per_kelvin=cfg.suppression_s_per_k, max_seconds=cfg.suppression_max_s,
            threshold=cfg.suppression_threshold_k)
        self.supply_air = ahu.SupplyAirTemperature(
            ignore, minimum=cfg.sat_min_k, maximum=cfg.sat_max_k, oat_low=cfg.sat_oat_low_k,
            oat_high=cfg.sat_oat_high_k, heating_mode=cfg.sat_heating_mode_k,
            cooldown=cfg.cooldown_sat_k, trim=cfg.sat_trim_k, respond=cfg.sat_respond_k,
            respond_max=cfg.sat_respond_max_k, period=cfg.sat_period_s, delay=cfg.sat_delay_s)
        self.static_pressure = ahu.StaticPressure(
            cfg.static_pressure_reset, ignore, initial=cfg.sp_initial_pa, minimum=cfg.sp_min_pa,
            maximum=cfg.sp_max_pa, trim=cfg.sp_trim_pa, respond=cfg.sp_respond_pa,
            respond_max=cfg.sp_respond_max_pa, period=cfg.sp_period_s, delay=cfg.sp_delay_s)
        self.official_loop = ahu.OfficialAhuLoop(
            ix.n_ahu, kp=cfg.official_kp, ki=cfg.official_ki, kd=cfg.official_kd,
            integral_limit=cfg.official_integral_limit)
        self.water = plant.SupplyWaterTemperature(
            initial=cfg.hw_initial_k, minimum=cfg.hw_min_k, maximum=cfg.hw_max_k, trim=cfg.hw_trim_k,
            respond=cfg.hw_respond_k, respond_max=cfg.hw_respond_max_k, period=cfg.hw_period_s,
            delay=cfg.hw_delay_s, ignore=cfg.hw_ignore, max_rise=cfg.hw_max_rise_k)
        self.plant_reset = None
        if cfg.pump_pressure == "plant_reset":
            self.plant_reset = plant.PlantReset(
                dp_min=cfg.dp_min_bar, dp_max=cfg.dp_max_bar, temp_low=cfg.hw_min_k,
                temp_high=cfg.hw_max_k, initial=cfg.plant_reset_initial, trim=cfg.plant_reset_trim,
                respond=cfg.plant_reset_respond, respond_max=cfg.plant_reset_respond_max,
                period=cfg.plant_reset_period_s, delay=cfg.plant_reset_delay_s,
                ignore=cfg.plant_reset_ignore)
        self.plant_enable = plant.PlantEnable(
            min_off_steps=cfg.plant_min_off_steps, min_on_steps=cfg.plant_min_on_steps,
            ignore=cfg.plant_ignore_requests, lockout=cfg.plant_lockout_k,
            lockout_hysteresis=cfg.plant_lockout_hysteresis_k)
        self.previous_valve = None
        self.previous_plant_on = None
        self.cycle_on = np.ones(ix.n_ahu, dtype=bool)
        self.cycle_steps = np.zeros(ix.n_ahu, dtype=int)
        self.counters = {name: 0 for name in self.COUNTERS}
        self.counters.update({f"mode_{name}_group_steps": 0 for name in MODE_NAMES})
        self.empty_day = self.arrived_today = False
        self.last_minute = None
        self.peak_occupants = 0.0
        self.occupied_min = self.cfg.min_flow
        self._geometry_metadata, self._zone_static = metadata, None
        self.zone_heat_factor, self.guard_zones = self._zone_geometry(metadata)
        self.heat_boost = np.ones(ix.n_zones)
        self.occupancy_ratio, self.room_peak_occupants = 0.0, 0.0
        self.room_error, self._rooms = None, None
        self.room_settings = None
        self.previous_day_people, self.day_people, self.break_minute = True, False, None
        self.vent_bank = np.zeros(ix.n_ahu)
        self._vent = None
        self.rogue_mean = np.zeros(n)
        self.previous_temp = None
        self.oat_today, self.oat_yesterday, self.oat_minute = None, None, None
        self.last_heating_state = np.zeros(n, dtype=bool)
        self.last_mode, self.last_fan = None, None
        self.prev_freeze_only = np.zeros(ix.n_ahu, dtype=bool)
        self.empty_hold = False
        self.oat_day = deque(maxlen=288)
        self.oat_24h = None
        self.saw_step, self.saw_charging = 0, False
        self.hw_on_setpoint, self.hw_off_steps = None, 0
        self.pump_dp_loop = TrimRespond(initial=cfg.dp_max_bar, minimum=min(cfg.pump_dp_min_bar, cfg.dp_max_bar),
                                        maximum=cfg.dp_max_bar, trim=-0.5, respond=1.0, respond_max=3.0,
                                        ignore=cfg.pump_dp_ignore, delay=2)
        self.margin_adjust, self.morning_penalty, self.morning_seen, self.feedback_minute = 0.0, 0.0, False, None
        self.temp_history = deque(maxlen=4)
        self.last_sat, self.sat_guarded = None, np.zeros(ix.n_ahu, dtype=bool)
        self.low_flow_groups = np.zeros(ix.n_ahu, dtype=bool)
        self.hw_pump_stop = False
        self.warmup_flow, self.keep_minute = np.zeros(n), None
        self.guard_rogue = np.zeros(n)
        self.econ_on = np.zeros(ix.n_ahu, dtype=bool)
        self.sp_pi, self.sp_last = None, None
        self.cooldown_offset, self.cool_hot, self.cool_seen, self.cool_minute = np.zeros(n), np.zeros(n, bool), \
            False, None
        self.hw_rogue = np.zeros(n)
        self.dist_theta = np.zeros((n, 3))
        self.dist_p = np.tile(np.eye(3) * 100.0, (n, 1, 1))
        self.dist_last, self.dist_hat = None, np.zeros(n)
        self.dist_slow = np.zeros(n)
        self.econ_counts, self.econ_side, self.econ_minute = np.zeros((n, 2)), np.zeros(n), None
        self.reserve_err = np.zeros(n)
        self.dep_max_absence, self.dep_zero_run, self.dep_seen, self.dep_off, self.dep_minute = None, 0.0, False, \
            False, None
        self.morning_hour = ix.morning_start_hour
        self.arr_history, self.arr_today, self.arr_minute = deque(maxlen=max(1, int(cfg.arrival_days))), None, None
        self.capacity_ready = False
        self._planner_on = cfg.daily_start or cfg.daily_layer or cfg.hourly_lp
        self.planner, self.planner_error = None, None
        self.ds_flow, self.lp_plan, self.layer_depth, self.pl_prev_occ, self.pl_steps = None, None, 0.0, False, 0
        self.lp_only = None
        self.lp_hour = 0
        if cfg.hw_request_rogue not in ("off", "share", "three"):
            raise ValueError(f"unknown hot-water request exclusion {cfg.hw_request_rogue!r}")

    GEOMETRY_FIELDS = ("zone_heat_ext_power", "zone_heat_redistribute", "sat_cost_guard_large_q")

    def set_geometry(self, **values):
        """The zone geometry switches between steps (stage 2B, the lookahead teacher's geometry options and a
        student that follows them); a key outside GEOMETRY_FIELDS raises."""
        unknown = set(values) - set(self.GEOMETRY_FIELDS)
        if unknown:
            raise KeyError(f"not a zone geometry switch: {sorted(unknown)}")
        if any(getattr(self.cfg, k) != v for k, v in values.items()):
            self.cfg = replace(self.cfg, **values)
            self.zone_heat_factor, self.guard_zones = self._zone_geometry(self._geometry_metadata)

    def _zone_geometry(self, metadata):
        """Per-zone heating maximum factors and the zones the A12 guard counts (G36Config zone_heat_ext_power,
        sat_cost_guard_large_q); (None, None) when both are off or the geometry cannot be built. The zone areas and
        exterior lengths are built once and kept."""
        cfg = self.cfg
        self.geometry_error = None
        if cfg.zone_heat_ext_power == 0.0 and cfg.sat_cost_guard_large_q <= 0.0:
            return None, None
        if self._zone_static is None:
            try:
                from smart_building.learned.tokens import ZONE, static_features
                self._zone_static = static_features(metadata, self.index)[ZONE]
            except Exception as error:
                self.geometry_error = repr(error)
                return None, None
        area, exterior = self._zone_static[:, 0], self._zone_static[:, 1]
        factor = None
        if cfg.zone_heat_ext_power != 0.0 and np.any(exterior > 0):
            reference = float(np.median(exterior[exterior > 0]))
            factor = np.maximum(1.0, exterior / reference) ** cfg.zone_heat_ext_power
        guard = area >= np.quantile(area, cfg.sat_cost_guard_large_q) if cfg.sat_cost_guard_large_q > 0.0 else None
        return factor, guard

    def _room_quantities(self):
        """(area m2, exterior wall length m) per zone for the per-room geometry switches (g36/rooms.py), built once;
        None when the geometry cannot be built (both switches then stay off, self.room_error says why)."""
        if self._rooms is None and self.room_error is None:
            try:
                from smart_building.g36.rooms import room_quantities
                self._rooms = room_quantities(self._geometry_metadata, self.index)
            except Exception as error:
                self.room_error = repr(error)
        return self._rooms

    def _redistribute(self, flow, heating_state, cooling_state, heating_max, cooling_max, minimum, active):
        """G36Config.zone_heat_redistribute: per AHU, the heating zones at the heating maximum with a factor above 1
        get up to factor x their flow (at most the cooling maximum), taken from the AHU's other active zones in
        proportion to how far they run above their minimum; the group's total flow is unchanged."""
        out = np.array(flow, dtype=float)
        heating_only = np.asarray(heating_state, dtype=bool) & ~np.asarray(cooling_state, dtype=bool)
        for s in self.index.group_slices:
            act = np.asarray(active[s], dtype=bool)
            if not act.any():
                continue
            f = out[s]
            want = act & heating_only[s] & (f >= heating_max[s] - 1e-9) & (self.zone_heat_factor[s] > 1.0)
            if not want.any():
                continue
            extra = np.where(want, np.maximum(np.minimum(cooling_max[s], f * self.zone_heat_factor[s]) - f, 0.0), 0.0)
            slack = np.where(act & ~want, np.maximum(f - minimum[s], 0.0), 0.0)
            need, have = float(extra.sum()), float(slack.sum())
            if need <= 0.0 or have <= 0.0:
                continue
            moved = min(need, have)
            out[s] = f + extra * (moved / need) - slack * (moved / have)
        return out

    def _dcv_minimum(self, occupants, static_pressure):
        """Occupancy-based minimum (G36Config.dcv_area_share): (zone minimum flow, airflow floor per group)."""
        cfg = self.cfg
        self.room_peak_occupants = max(self.room_peak_occupants, float(occupants))
        self.occupancy_ratio = (float(occupants) / self.room_peak_occupants if self.room_peak_occupants > 0
                                else 0.0)
        if cfg.dcv_area_share >= 1.0:
            return cfg.min_flow, zone.DAMPER_FLOOR
        self.peak_occupants = max(self.peak_occupants, float(occupants))
        ratio = float(occupants) / self.peak_occupants if self.peak_occupants > 0 else 0.0
        share = cfg.dcv_area_share
        if cfg.dcv_area_share_auto and self.peak_occupants > 0:
            rooms = self._room_quantities()
            if rooms is not None:
                ra_total = cfg.vot_ra_lps_m2 * float(np.sum(rooms[0]))
                share = ra_total / max(ra_total + cfg.vot_rp_lps_person * self.peak_occupants, 1e-9)
        minimum = cfg.min_flow * (share + (1.0 - share) * ratio)
        pressure = np.minimum(zone.effective_static_pressure(np.asarray(static_pressure, dtype=float)),
                              zone.DESIGN_STATIC_PRESSURE)
        return minimum, (zone.DAMPER_FLOOR * np.sqrt(pressure / zone.DESIGN_STATIC_PRESSURE))[:, None]

    def _departure(self, minute_of_day, occupants, occupied):
        """O503: the schedule's occupied flag, cleared for the rest of the day once today's occupants have been away
        longer than the longest midday absence learned so far (G36Config.departure_learn)."""
        minute = int(minute_of_day)
        if self.dep_minute is not None and minute < self.dep_minute:
            self.dep_zero_run, self.dep_seen, self.dep_off = 0.0, False, False
        self.dep_minute = minute
        if not occupied:
            return occupied
        if float(occupants) > 0:
            if self.dep_seen and self.dep_zero_run > 0.0:
                self.dep_max_absence = max(self.dep_max_absence or 0.0, self.dep_zero_run)
            self.dep_zero_run, self.dep_seen, self.dep_off = 0.0, True, False
        elif self.dep_seen:
            self.dep_zero_run += 5.0
            limit = self.dep_max_absence if self.dep_max_absence is not None else self.cfg.departure_default_min
            if self.dep_zero_run > limit:
                self.dep_off = True
        return bool(occupied) and not self.dep_off

    def _empty_day(self, minute_of_day, occupants, occupied):
        """Empty weekday: the schedule's occupied flag, cleared while today counts as empty (G36Config)."""
        minute = int(minute_of_day)
        if self.last_minute is not None and minute < self.last_minute:
            self.empty_day = self.arrived_today = False
        self.last_minute = minute
        if float(occupants) > 0:
            self.empty_day, self.arrived_today = False, True
        elif occupied and not self.arrived_today and \
                minute >= self.morning_hour * 60 + self.cfg.empty_day_delay_min:
            self.empty_day = True
        return bool(occupied) and not self.empty_day

    def _track_outside_air(self, minute_of_day, oat):
        """Climate gates: today's running (min, max, sum, n) of the mean outside air; at each new calendar day the
        finished day becomes oat_yesterday = (min, max, mean)."""
        minute, t = int(minute_of_day), float(np.mean(oat))
        if self.oat_minute is not None and minute < self.oat_minute and self.oat_today is not None:
            low, high, total, count = self.oat_today
            self.oat_yesterday, self.oat_today = (low, high, total / count), None
        self.oat_minute = minute
        if self.oat_today is None:
            self.oat_today = [t, t, t, 1]
        else:
            self.oat_today = [min(self.oat_today[0], t), max(self.oat_today[1], t), self.oat_today[2] + t,
                              self.oat_today[3] + 1]

    def _yesterday_gate(self, min_range, min_max):
        """True when yesterday's outdoor range and maximum reach the thresholds (0 = no condition); False before the
        first finished day when a threshold is set."""
        if min_range <= 0.0 and min_max <= 0.0:
            return True
        if self.oat_yesterday is None:
            return False
        low, high, _ = self.oat_yesterday
        return (min_range <= 0.0 or high - low >= min_range) and (min_max <= 0.0 or high >= min_max)

    def _vent_requirement(self):
        """        occupancy rate); None when the zone geometry cannot be built (then the AHU never coasts)."""
        if self._vent is None:
            rooms = self._room_quantities()
            if rooms is None:
                return None
            cfg, area = self.cfg, rooms[0]
            vbz = np.array([float(np.sum((cfg.vent_ra_lps_m2 + cfg.vent_rp_lps_m2) * area[s])) / 1000.0
                            for s in self.index.group_slices])
            volume = np.array([float(np.sum(area[s])) * cfg.vent_height_m for s in self.index.group_slices])
            self._vent = (vbz / (cfg.vent_ez * cfg.vent_ev), 3.0 * volume / np.maximum(vbz, 1e-9))
        return self._vent

    def _heat_budget(self, flow, temp, heat_sp, heating_state, cooling_state, heating_min, heating_max, apply,
                     budgets=None):
        """O3: per AHU in `apply` groups, the heating zones' airflow above their heating minimum is at most
        ahu_heat_budget (or budgets[g], O544) x the sum of (heating max - heating min), given to the zones with the
        largest deficit below their heating target first (the rest keep their heating minimum)."""
        out = np.array(flow, dtype=float)
        heating_only = np.asarray(heating_state, dtype=bool) & ~np.asarray(cooling_state, dtype=bool)
        for g, s in enumerate(self.index.group_slices):
            if not apply[g]:
                continue
            idx = np.flatnonzero(heating_only[s]) + s.start
            if idx.size == 0:
                continue
            excess = np.maximum(out[idx] - heating_min[idx], 0.0)
            share = self.cfg.ahu_heat_budget if budgets is None else float(budgets[g])
            budget = share * float(np.sum(np.maximum(heating_max[idx] - heating_min[idx], 0.0)))
            if excess.sum() <= budget:
                continue
            order = np.argsort(-(heat_sp[idx] - temp[idx]), kind="stable")
            left = budget
            given = np.zeros_like(excess)
            for k in order:
                given[k] = min(excess[k], left)
                left -= given[k]
            out[idx] = np.minimum(out[idx], heating_min[idx] + given)
        return out

    def _daily_gate(self):
        """O505 / O544 climate gate: the daily planners may act (no gate, or the 24 h mean outside air is at least
        daily_min_oat24_k)."""
        cfg = self.cfg
        return cfg.daily_min_oat24_k <= 0.0 or (self.oat_24h is not None and self.oat_24h >= cfg.daily_min_oat24_k)

    def _uses_oat_24h(self):
        """Pipeline 5: an option that reads the 24 h mean outside air (or its history) is on."""
        cfg = self.cfg
        return (cfg.cold_low_flow_oat_k > 0.0 or cfg.night_heat_mode != "off" or cfg.night_cool_mode != "off"
                or cfg.hourly_lp or cfg.no_reheat_min_oat24_k > 0.0 or cfg.daily_min_oat24_k > 0.0
                or cfg.big_room_heat_max_ratio > 0.0 or cfg.economic_keep_cold_k > 0.0)

    def _arrival(self, o):
        """O503 arrival half: at each new calendar day the morning start hour in force becomes the learned arrival
        (G36Config.arrival_learn); today's first arrival inside the schedule is recorded."""
        cfg, ix = self.cfg, self.index
        minute = int(o.minute_of_day)
        if self.arr_minute is not None and minute < self.arr_minute:
            if self.arr_today is not None:
                self.arr_history.append(float(self.arr_today))
            self.arr_today = None
            hour = ix.morning_start_hour
            if len(self.arr_history) >= cfg.arrival_min_days:
                first = float(np.quantile(np.array(self.arr_history), cfg.arrival_quantile))
                start = min(max(first, ix.morning_start_hour * 60.0), ix.evening_start_hour * 60.0 - 60.0)
                hour = 5.0 * np.floor(start / 5.0) / 60.0
            self.morning_hour = hour
            self.optimal_start.morning_hour = hour
        self.arr_minute = minute
        if self.arr_today is None and float(o.occupants) > 0 and is_occupied(
                o.minute_of_day, o.weekday, ix.morning_start_hour, ix.evening_start_hour):
            self.arr_today = max(minute - 5, 0) if cfg.arrival_interval_start else minute

    def _capacity_extra(self):
        """O506 / O505 fixed candidates: OptimalStart.group_extra (once): capacity_start_extra_s for the AHUs short of
        capacity (room geometry; none without it) plus start_offset_s for every AHU."""
        cfg = self.cfg
        self.capacity_ready = True
        extra = np.full(self.index.n_ahu, float(cfg.start_offset_s))
        if cfg.capacity_start_extra_s > 0.0:
            rooms = self._room_quantities()
            if rooms is not None:
                wall = np.bincount(self.index.zone_group, weights=rooms[1], minlength=self.index.n_ahu)
                per_zone = wall / np.maximum(self.index.group_sizes, 1)
                self.capacity_short = per_zone > float(np.median(per_zone))
                extra = extra + np.where(self.capacity_short, float(cfg.capacity_start_extra_s), 0.0)
        if cfg.capacity_start_heat_only:
            self.optimal_start.group_extra = extra
            self.optimal_start.group_extra_cool = np.full(self.index.n_ahu, float(cfg.start_offset_s))
        elif np.any(extra != 0.0):
            self.optimal_start.group_extra = extra

    def _oat_forecast(self, hours):
        """O545: hourly outside air for the next `hours` hours: yesterday's same hours (the last 288 step means) moved
        by today's difference to yesterday at this hour; the present value before a day of history."""
        hist = list(self.oat_day)
        now = hist[-1]
        if len(hist) < 288:
            return np.full(hours, now)
        bias = now - hist[0]
        return np.array([float(np.mean(hist[12 * k:12 * k + 12])) + bias if 12 * k < 288 else now
                         for k in range(hours)])

    def _planners(self, o, temp, occupied, targets, warm, cool):
        """O505 / O544 / O545 (g36/planner.py): the learned per-AHU model, the nightly start choice, the evening
        decisions and the hourly plan; returns the warm-up / cool-down flags in force (the rule's when nothing acts)."""
        cfg, ix = self.cfg, self.index
        if self.planner is None:
            from smart_building.g36 import planner
            self._pl = planner
            model = planner.GroupModel(ix)
            starter = planner.DailyStart(ix) if (cfg.daily_start or cfg.daily_layer) else None
            layer = planner.DailyLayer(ix, parts=cfg.daily_layer_parts) if cfg.daily_layer else None
            lp = None
            if cfg.hourly_lp:
                try:
                    lp = planner.HourlyLP(ix, horizon_h=cfg.hourly_lp_horizon_h, every_steps=cfg.hourly_lp_every_steps,
                                          time_limit_s=cfg.hourly_lp_time_limit_s,
                                          comfort_weight=cfg.hourly_lp_comfort_weight)
                except Exception as error:
                    self.planner_error = repr(error)
            self.planner = {"model": model, "starter": starter, "layer": layer, "lp": lp}
        pl = self.planner
        model, starter, layer, lp = pl["model"], pl["starter"], pl["layer"], pl["lp"]
        model.observe(o, temp)
        minute, weekday = int(o.minute_of_day), int(o.weekday)
        to_start = seconds_to_next_occupied(minute, weekday, self.morning_hour, ix.evening_start_hour)
        warm_out, cool_out = np.array(warm, dtype=bool), np.array(cool, dtype=bool)
        use_start = cfg.daily_start or (cfg.daily_layer and cfg.daily_layer_parts == "all")
        if layer is not None and self.pl_prev_occ and not occupied:
            if use_start:
                reach, flow = min(starter.max_lead_s, float(self.optimal_start.t_opt_max)
                                  + 3600.0 * max(starter.OFFSETS_H)), 1.0
            else:
                reach, flow = float(self.optimal_start.t_opt_max), None
            layer.evening(model, starter, temp, o.oat, ix.eco_window[0], targets[0], reach, cfg.heating_max_ratio,
                          cfg.sp_max_pa, flow)
            valid = model.valid()
            self.layer_depth = float(np.min(layer.depth[valid])) if valid.any() else 0.0
        self.pl_prev_occ = bool(occupied)
        self.ds_flow = None
        if starter is not None and use_start and self._daily_gate() and not occupied and to_start > 0:
            key = int(round(self.pl_steps + to_start / 300.0))
            if starter.decided_for != key and to_start <= starter.max_lead_s + 300.0:
                deviation = targets[0] - np.asarray(temp, dtype=float)
                start = self.optimal_start
                zone_lead = np.where(deviation > start.u_high,
                                     np.minimum(deviation / start.heating.average, start.t_opt_max), 0.0)
                rule_lead = np.array([float(zone_lead[s].max()) for s in ix.group_slices])
                t_now = np.asarray(temp, dtype=float)
                mean = model.mean_temp(t_now)
                spread = np.clip(mean - np.array([float(t_now[s].min()) for s in ix.group_slices]), 0.0,
                                 self._pl.SPREAD_MAX_K)
                floor = ix.eco_window[0] - cfg.setback_heat_drop_k - (self.layer_depth if cfg.daily_layer else 0.0)
                within = float("inf") if cfg.unoccupied_conditioning == "always" else 60.0 * cfg.unoccupied_lead_min
                starter.decide(model, temp, o.oat, to_start, targets[0], rule_lead, cfg.heating_max_ratio, spread,
                               floor, within)
                starter.decided_for = key
            if starter.decided_for == key and starter.choice is not None:
                lead, flow, valid = starter.choice
                planned = to_start <= lead
                warm_out = np.where(valid, planned, warm_out)
                self.ds_flow = np.where(valid & planned, flow, np.nan)
        self.lp_plan, self.lp_only = None, None
        if lp is not None:
            t_now = np.asarray(temp, dtype=float)
            if lp.due():
                occupied_h, hours_after = self._pl.hourly_schedule(minute, weekday, self.morning_hour,
                                                                   ix.evening_start_hour, lp.horizon)
                if not occupied and is_occupied(minute, weekday, self.morning_hour, ix.evening_start_hour):
                    k = 0
                    while k < len(occupied_h) and occupied_h[k]:
                        occupied_h[k] = False
                        k += 1
                    if k == len(occupied_h) and hours_after <= 0.0:
                        end = minute + 60 * len(occupied_h)
                        hours_after = max(0.0, (ix.evening_start_hour * 60 - end) / 60.0 + seconds_to_next_occupied(
                            int(ix.evening_start_hour * 60), weekday, self.morning_hour,
                            ix.evening_start_hour) / 3600.0)
                sizes = ix.group_sizes * zone.VAV_MAX_FLOW
                q_heat = self._pl.AIR_KJ * cfg.heating_max_ratio * cfg.cooling_max_flow * sizes * max(
                    cfg.discharge_limit_k - targets[0], 1.0)
                q_cool = self._pl.AIR_KJ * cfg.cooling_max_flow * sizes * max(targets[1] - cfg.sat_min_k, 1.0)
                mean = model.mean_temp(t_now)
                spread = self._pl.SPREAD_MAX_K
                heat_extra = np.clip(mean - np.array([float(t_now[s].min()) for s in ix.group_slices]), 0.0, spread)
                cool_extra = np.clip(np.array([float(t_now[s].max()) for s in ix.group_slices]) - mean, 0.0, spread)
                forecast = self._oat_forecast(lp.horizon)
                plan = lp.solve(model, mean, forecast, occupied_h, targets[0], targets[1], cfg.freeze_off_k, q_heat,
                                q_cool, hours_after, heat_extra, cool_extra)
                if plan is not None:
                    plan["step0"] = lp.steps - 1
                    before = plan["before"]
                    lead = min(float(to_start), 3600.0 * before)
                    may_warm, may_cool = np.zeros(ix.n_ahu, dtype=bool), np.zeros(ix.n_ahu, dtype=bool)
                    reach = cfg.t_opt_max_s + 300.0 * lp.every
                    if before > 0 and 0.0 < lead and to_start <= reach:
                        for g in np.flatnonzero(plan["valid"]):
                            heat_kwh = float(plan["heat_h"][g, :before].sum())
                            cool_kwh = float(plan["cool_h"][g, :before].sum())
                            may_warm[g] = heat_kwh > 0.0 and heat_kwh >= model.delivered_kwh(
                                g, mean[g], forecast[0], lead, targets[0] + heat_extra[g], q_heat[g])
                            may_cool[g] = cool_kwh > 0.0 and cool_kwh >= model.delivered_kwh(
                                g, mean[g], forecast[0], lead, targets[1] - cool_extra[g], q_cool[g], cooling=True)
                    plan["may_drop"] = (may_warm, may_cool)
            plan = lp.plan
            elapsed = (lp.steps - 1 - plan["step0"]) if plan is not None else 0
            hour, into = divmod(elapsed, 12)
            if plan is not None and hour < plan["heat_h"].shape[1]:
                self.lp_plan = plan
                if not occupied:
                    valid = plan["valid"]
                    left = (12 - into) / 12.0
                    heat_frac = plan["heat_h"][:, hour] / np.maximum(plan["heat_max"], 1e-9)
                    cool_frac = plan["cool_h"][:, hour] / np.maximum(plan["cool_max"], 1e-9)
                    heat_on = valid & (heat_frac > 0.05) & (left <= heat_frac + 1e-9)
                    cool_on = valid & (cool_frac > 0.05) & (left <= cool_frac + 1e-9) & ~heat_on
                    rule_warm, rule_cool = warm_out.copy(), cool_out.copy()
                    may_warm, may_cool = plan["may_drop"]
                    warm_out = heat_on | (rule_warm & ~(valid & may_warm))
                    cool_out = cool_on | (rule_cool & ~(valid & may_cool))
                    self.lp_only = (heat_on & ~rule_warm, cool_on & ~rule_cool)
                    self.lp_hour = hour
        self.pl_steps += 1
        return warm_out, cool_out

    def _night_power(self, o, temp, return_temp):
        """O507 / O522: (power W per zone, heating) for this step, or None (no option acting, no geometry)."""
        cfg, ix = self.cfg, self.index
        if self.oat_24h is None:
            return None
        heat = cfg.night_heat_mode != "off" and self.oat_24h < cfg.night_heat_oat_k
        cool = cfg.night_cool_mode != "off" and self.oat_24h >= cfg.night_cool_oat_k
        if not (heat or cool):
            return None
        rooms = self._room_quantities()
        if rooms is None:
            return None
        area, wall = rooms
        wpa = wall / np.maximum(area, 1e-9)
        q75, wmin, wmax = (float(np.quantile(wpa, 0.75)), float(wpa.min()), float(wpa.max()))
        to_start = seconds_to_next_occupied(o.minute_of_day, o.weekday, self.morning_hour, ix.evening_start_hour)
        hours = to_start / 3600.0
        minute = int(o.minute_of_day)
        night = minute >= 19 * 60 or minute < self.morning_hour * 60
        low = float(ix.comfort_window[0])
        t = np.asarray(temp, dtype=float)
        if heat:
            mode = cfg.night_heat_mode
            if hours < 2.0 or (mode == "weekday" and hours > 24.0):
                return None
            if cfg.night_heat_max_h > 0.0 and hours > cfg.night_heat_max_h:
                return None
            if mode == "free_gas":
                q = np.full(ix.n_zones, 26.0 if hours <= 18.0 else 4.0)
                if self.oat_24h < 255.15:
                    severe = wpa >= q75
                    q = np.where(severe, 57.0 + 41.0 * np.clip((wpa - q75) / max(wmax - q75, 1e-9), 0.0, 1.0), q)
            else:
                q = 5.0 + 27.0 * np.clip((wpa - wmin) / max(q75 - wmin, 1e-9), 0.0, 1.0)
            q = np.where(t > low + cfg.night_heat_stop_k, 0.0, q) * cfg.night_heat_scale
            return q * area, True
        if cfg.night_cool_max_h > 0.0 and hours > cfg.night_cool_max_h:
            return None
        if cfg.night_cool_mode == "night":
            if not night or hours > 11.0:
                return None
            q = np.where(t > low + 1.0, 7.5, 0.0)
            hotter = np.broadcast_to(np.asarray(o.oat, dtype=float), (ix.n_ahu,)) > np.asarray(return_temp)
            q = np.where(hotter[ix.zone_group], 0.0, q)
        else:
            if hours <= 8.0:
                q = np.where(wpa >= q75, 20.3, 12.5)
            else:
                q = np.full(ix.n_zones, 7.5 if night else 3.4)
            q = np.where((t - low > 4.7) & (hours >= 33.0), 0.0, q)
        q = np.where(t <= low + 0.5, 0.0, q)
        if cfg.night_cool_floor_k > 0.0:
            q = np.where(t < float(ix.comfort_window[1]) - cfg.night_cool_floor_k, 0.0, q)
        return q * cfg.night_cool_scale * area, False

    def _night_conditioning(self, o, temp, mode, return_temp):
        """O507 / O522: None, or the UNOCCUPIED groups that run and their (sat, pressure, recirculation per group;
        damper, valve, reheat zones per zone) commands."""
        cfg, ix = self.cfg, self.index
        power = self._night_power(o, temp, return_temp)
        if power is None:
            return None
        watts, heating = power
        groups = ix.zone_group
        run = (mode == UNOCCUPIED) & (np.bincount(groups, weights=(watts > 0.0).astype(float),
                                                  minlength=ix.n_ahu) > 0.0)
        if not run.any():
            return None
        t = np.asarray(temp, dtype=float)
        oat = np.broadcast_to(np.asarray(o.oat, dtype=float), (ix.n_ahu,))
        pressure = np.full(ix.n_ahu, float(cfg.sp_min_pa))
        floor = zone.DAMPER_FLOOR * np.sqrt(zone.effective_static_pressure(pressure) / zone.DESIGN_STATIC_PRESSURE)
        air = zone.AIR_DENSITY * zone.AIR_HEAT_CAPACITY * zone.VAV_MAX_FLOW
        if heating:
            recirculation = np.full(ix.n_ahu, float(cfg.max_recirculation))
            supply = recirculation * return_temp + (1.0 - recirculation) * oat
            sat = np.full(ix.n_ahu, float(cfg.sat_heating_mode_k))
            target = np.where(watts > 0.0, float(cfg.discharge_limit_k), t)
            flow = watts / (air * np.maximum(target - t, 1.0))
        else:
            recirculation = np.where(oat < return_temp, 0.0, float(cfg.max_recirculation))
            sat = np.array(return_temp, dtype=float)
            for g in np.flatnonzero(run):
                s = ix.group_slices[g]
                want = watts[s] > 0.0
                gap = float(np.median(watts[s][want])) / (air * float(floor[g]))
                sat[g] = float(np.clip(np.median(t[s][want]) - gap, cfg.sat_min_k, max(return_temp[g] - 1.0,
                                                                                         cfg.sat_min_k)))
            supply_mix = recirculation * return_temp + (1.0 - recirculation) * oat
            supply = np.minimum(supply_mix, sat)
            flow = watts / (air * np.maximum(t - supply[groups], 1.0))
            target = None
        flow = np.clip(flow, floor[groups], float(cfg.cooling_max_flow))
        damper = zone.damper_command(flow, pressure[groups])
        valve = np.zeros(ix.n_zones)
        if heating:
            applied = zone.applied_damper(damper)
            box_pressure = zone.applied_static_pressure(pressure, run)[groups]
            ratio = zone.supply_flow_ratio(zone.box_flow_demand(applied, box_pressure), groups, ix.n_ahu)
            water_dp = float(o.water_dp) if bool(o.hws_on) else 0.0
            valve = zone.reheat_valve_for(target, applied, box_pressure, ratio[groups], supply[groups],
                                          o.water_temp, water_dp)
        return {"run": run, "sat": sat, "pressure": pressure, "recirculation": recirculation, "damper": damper,
                "valve": valve}

    def _margin_feedback(self, o, temp):
        """O543: comfort kelvin-steps of the first two occupied hours while the building has occupants; at each new
        calendar day, a day that had occupants then moves the margin by +step (any kelvin-step) or -step (none)."""
        cfg, ix = self.cfg, self.index
        minute = int(o.minute_of_day)
        if self.feedback_minute is not None and minute < self.feedback_minute:
            if self.morning_seen:
                step = cfg.margin_feedback_step_k if self.morning_penalty > 0.0 else -cfg.margin_feedback_step_k
                margin = float(np.clip(cfg.occupied_margin_k + self.margin_adjust + step, 0.0,
                                       cfg.margin_feedback_max_k))
                self.margin_adjust = margin - cfg.occupied_margin_k
            self.morning_penalty, self.morning_seen = 0.0, False
        self.feedback_minute = minute
        start = self.morning_hour * 60
        if start <= minute < start + 120 and float(o.occupants) > 0:
            low, high = ix.comfort_window
            t = np.asarray(temp, dtype=float)
            self.morning_penalty += float(np.sum(np.maximum(low - t, 0.0) + np.maximum(t - high, 0.0)))
            self.morning_seen = True

    def _disturbance(self, o, temp):
        """O530: update each zone's model with the step that just ended and return the smoothed disturbance (K per
        step). The box air heat of that step comes from the observed (applied) dampers, valves and pressure."""
        cfg, ix = self.cfg, self.index
        groups = ix.zone_group
        t = np.asarray(temp, dtype=float)
        if self.dist_last is not None:
            t_prev = self.dist_last
            oat = np.broadcast_to(np.asarray(o.oat, dtype=float), (ix.n_ahu,))[groups]
            return_temp = np.array([t_prev[s].mean() for s in ix.group_slices])
            supply = zone.estimate_supply_air(return_temp, o.oat, o.recirculation, o.sat_setpoint, o.ahu_on)
            pressure = o.static_pressure[groups]
            demand = zone.box_flow_demand(o.damper, pressure)
            ratio = zone.supply_flow_ratio(demand, groups, ix.n_ahu)
            discharge = zone.estimate_discharge(o.damper, o.valve, pressure, ratio[groups], supply[groups],
                                                o.water_temp, o.water_dp)
            on = np.asarray(o.ahu_on, dtype=bool)[groups]
            q = np.where(on, zone.AIR_DENSITY * zone.AIR_HEAT_CAPACITY * demand * ratio[groups]
                         * (discharge - t_prev), 0.0) / 1000.0
            u = oat - t_prev
            x = np.stack([q, u, np.ones_like(q)], axis=1)
            y = t - t_prev
            px = np.einsum("nij,nj->ni", self.dist_p, x)
            gain = px / (cfg.disturbance_forget + np.einsum("ni,ni->n", x, px))[:, None]
            error = y - np.einsum("ni,ni->n", x, self.dist_theta)
            self.dist_theta = self.dist_theta + gain * error[:, None]
            self.dist_p = (self.dist_p - np.einsum("ni,nj->nij", gain, px)) / cfg.disturbance_forget
            trace = np.einsum("nii->n", self.dist_p)
            self.dist_p = self.dist_p * np.minimum(1.0, 300.0 / np.maximum(trace, 1e-12))[:, None, None]
            bad = ~(np.all(np.isfinite(self.dist_theta), axis=1) & np.all(np.isfinite(self.dist_p), axis=(1, 2)))
            if bad.any():
                self.dist_theta[bad] = 0.0
                self.dist_p[bad] = np.eye(3) * 100.0
            residual = y - self.dist_theta[:, 0] * q - self.dist_theta[:, 1] * u
            self.dist_hat = self.dist_hat + 0.3 * (np.clip(residual, -1.0, 1.0) - self.dist_hat)
        self.dist_last = t.copy()
        return self.dist_hat

    def _cooldown_learning(self, o, temp):
        """O536: zones above the comfort window in the first occupied hour (with occupants); at each new calendar day
        a day that had occupants then lowers those zones' cool-down target by a step, the others' by half a step
        back up, within [0, cooldown_learn_max_k]."""
        cfg, ix = self.cfg, self.index
        minute = int(o.minute_of_day)
        if self.cool_minute is not None and minute < self.cool_minute:
            if self.cool_seen:
                self.cooldown_offset = np.clip(np.where(self.cool_hot, self.cooldown_offset + cfg.cooldown_learn_step_k,
                                                        self.cooldown_offset - 0.5 * cfg.cooldown_learn_step_k),
                                               0.0, cfg.cooldown_learn_max_k)
            self.cool_hot[:], self.cool_seen = False, False
        self.cool_minute = minute
        start = self.morning_hour * 60
        if start <= minute < start + 60 and float(o.occupants) > 0 and int(o.weekday) < 5:
            self.cool_hot |= np.asarray(temp, dtype=float) > ix.comfort_window[1]
            self.cool_seen = True

    def _occupied_targets(self):
        """The occupied (heating, cooling) targets in force: the A3 margin, moved by the O543 feedback when it is on
        (then also the optimal start's targets)."""
        cfg = self.cfg
        if cfg.margin_feedback_step_k <= 0.0:
            return self.occupied_targets
        margin = float(np.clip(cfg.occupied_margin_k + self.margin_adjust, 0.0, cfg.margin_feedback_max_k))
        window = self.index.comfort_window
        targets = (window[0] + margin, window[1] - margin)
        self.optimal_start.heating_setpoint = float(targets[0])
        if cfg.precool_until_hour <= 0:
            self.optimal_start.cooling_setpoint = float(targets[1])
        return targets

    def _hw_sawtooth(self, rule_water, valve, tank):
        """O546: (hot-water setpoint K, stop the pump this step) while any reheat valve is commanded open."""
        cfg = self.cfg
        if not np.any(np.asarray(valve) > 0.01):
            return rule_water, False
        low, high = float(cfg.hw_min_k), float(cfg.hw_max_k)
        if cfg.hw_sawtooth == "band":
            if float(tank) < cfg.hw_saw_refill_below_k:
                self.saw_charging = True
            elif float(tank) >= high - 0.05:
                self.saw_charging = False
            return (high if self.saw_charging else low), False
        period = max(1, int(cfg.hw_saw_period))
        charge = self.saw_step % period >= period - max(1, -(-period // 6))
        self.saw_step += 1
        return (high if charge else low), bool(charge and cfg.hw_sawtooth == "saw_pump_off")

    def _cost_based_sat(self, sat, temp, cooling, cooling_state, return_temp, o, fan_on, mode, cool_sp=None,
                        requests=None):
        """Baseline v2 A12: per running OCCUPIED group, the SAT candidate with the lowest estimated score cost.

        Flows are fractions of each zone's design flow (zones taken as equal in size). A cooling zone keeps the
        cooling it gets now, flow x (zone - SAT), so a warmer SAT needs more flow; a zone at minimum flow that is
        reheating (observed valve > 0.05) needs reheat for every kelvin the SAT falls. Electricity: chiller flow x
        (mixed - SAT) / COP 3.5 plus fan about 0.4 K-equivalent per unit flow; gas: reheat / 0.9; one unit of
        electricity weighs 2.55 units of gas in the score. With the comfort guard on (sat_cost_guard_k > 0 with
        cool_sp per zone, or sat_cost_respect_requests with requests per group), a guarded group's SAT is at most
        G36's own value `sat`. The minimum flow a zone actually gets is at least the simulator's damper floor: a 0.1
        damper POSITION (vav.py:490-495), i.e. 0.1 x sqrt(SP / 2000) of the design flow at the group's observed static
        pressure (2026-10-29; with min_flow >= 0.1 this changes nothing)."""
        cfg = self.cfg
        pressure = np.minimum(zone.effective_static_pressure(np.asarray(o.static_pressure, dtype=float)),
                              zone.DESIGN_STATIC_PRESSURE)
        floor_flow = zone.DAMPER_FLOOR * np.sqrt(pressure / zone.DESIGN_STATIC_PRESSURE)
        near_k = cfg.sat_cost_guard_k if cool_sp is not None else 0.0
        asked = np.asarray(requests) > 0 if cfg.sat_cost_respect_requests and requests is not None else None
        out = np.array(sat, dtype=float)
        top = cfg.sat_cost_max_k
        if cfg.sat_cost_max_min_range_k > 0.0 and not (
                self.oat_yesterday is not None
                and self.oat_yesterday[1] - self.oat_yesterday[0] >= cfg.sat_cost_max_min_range_k):
            top = min(top, 293.15)
        candidates = np.arange(cfg.sat_min_k, top + 1e-9, 0.5)
        recirculation = np.clip(np.asarray(o.recirculation, dtype=float), 0.0, cfg.max_recirculation)
        mixed = recirculation * return_temp + (1.0 - recirculation) * np.asarray(o.oat, dtype=float)
        reheating_now = np.asarray(o.valve, dtype=float) > 0.05
        economizer_model = cfg.sat_cost_model == "economizer"
        if economizer_model:
            oat = np.broadcast_to(np.asarray(o.oat, dtype=float), np.shape(return_temp))
            mixed_max = cfg.max_recirculation * return_temp + (1.0 - cfg.max_recirculation) * oat
            reachable = np.where((oat < return_temp) & (cfg.economizer_mode == "supply_air"), oat, mixed_max)
        if economizer_model or cfg.sat_cost_fan_observed:
            fan_k = (np.asarray(o.static_pressure, dtype=float) * (2.0 - recirculation)) / (0.9 * 1206.0)
        guard_temp = None
        if cfg.sat_guard_trend_s > 0.0 and len(self.temp_history) >= 2:
            history = self.temp_history
            slope = (history[-1] - history[0]) / (300.0 * (len(history) - 1))
            if cfg.sat_guard_trend_one_sided:
                slope = np.maximum(slope, 0.0)
            guard_temp = np.asarray(temp, dtype=float) + slope * cfg.sat_guard_trend_s
        hysteresis = cfg.sat_guard_hysteresis_k > 0.0
        guarded_now = np.zeros(self.index.n_ahu, dtype=bool)
        for g, s in enumerate(self.index.group_slices):
            if not fan_on[g] or mode[g] != OCCUPIED:
                continue
            zone_t = temp[s]
            loop = np.where(cooling_state[s], cooling[s], 0.0)
            cools = loop > 0.0
            min_flow = max(self.occupied_min, float(floor_flow[g]))
            if self.low_flow_groups[g]:
                min_flow = float(floor_flow[g])
            flow_now = min_flow + loop * (cfg.cooling_max_flow - min_flow)
            delivered = flow_now * np.maximum(zone_t - out[g], 0.1)
            reheats = ~cools & reheating_now[s]
            best, best_cost = out[g], None
            eco = economizer_model and float(oat[g]) >= cfg.sat_cost_model_min_oat_k
            for candidate in candidates:
                supply = min(candidate, float(mixed_max[g])) if eco else candidate
                if (eco and cfg.sat_chiller_lockout_k > 0.0 and float(oat[g]) < cfg.sat_chiller_lockout_k
                        and float(reachable[g]) > candidate):
                    continue
                flow = np.where(cools, np.maximum(delivered / np.maximum(zone_t - supply, 0.1), min_flow),
                                min_flow)
                if np.any(flow > cfg.cooling_max_flow + 1e-9):
                    continue
                if eco:
                    electricity = flow.sum() * (max(float(reachable[g]) - candidate, 0.0) / 3.5 + float(fan_k[g]))
                    gas = min_flow * reheats.sum() * (out[g] - supply) / 0.9
                    cost = 2.55 * electricity + gas
                    if best_cost is None or cost < best_cost:
                        best, best_cost = candidate, cost
                    continue
                fan_term = float(fan_k[g]) if cfg.sat_cost_fan_observed else 0.4
                electricity = flow.sum() * (max(mixed[g] - candidate, 0.0) / 3.5 + fan_term)
                gas = min_flow * reheats.sum() * (out[g] - candidate) / 0.9
                cost = 2.55 * electricity + gas
                if best_cost is None or cost < best_cost:
                    best, best_cost = candidate, cost
            width = near_k
            if cfg.sat_guard_trend_k > 0.0 and 0.0 < near_k < 5.0:
                width = cfg.sat_guard_trend_k
            guard_t = zone_t if guard_temp is None else guard_temp[s]
            near = guard_t > cool_sp[s] - width
            if cfg.sat_guard_rogue_hours > 0.0 and 0.0 < near_k < 5.0:
                self.guard_rogue[s] += (near.astype(float) - self.guard_rogue[s]) / (cfg.sat_guard_rogue_hours * 12.0)
                near = near & (self.guard_rogue[s] <= cfg.sat_guard_rogue_share)
            if self.guard_zones is not None:
                near = near & self.guard_zones[s]
            near_guard = near_k > 0.0 and int(np.sum(near)) > cfg.sat_cost_guard_ignore
            if hysteresis and near_k > 0.0 and self.sat_guarded[g] and not near_guard:
                hold = guard_t > cool_sp[s] - width - cfg.sat_guard_hysteresis_k
                if self.guard_zones is not None:
                    hold = hold & self.guard_zones[s]
                near_guard = int(np.sum(hold)) > cfg.sat_cost_guard_ignore
            guarded_now[g] = near_guard
            guarded = near_guard or (asked is not None and bool(asked[g]))
            out[g] = min(best, float(sat[g])) if guarded else best
        if hysteresis:
            self.sat_guarded = guarded_now
        return out

    def set_parameters(self, **values):
        """Baseline v2 B2-RL slow layer:
        change SLOW_PARAMETER_RANGES parameters from the next act() on, keeping every other state of the controller.
        A key outside the table or a value outside its range raises; setting the current values changes nothing.
        sp_max_pa becomes every static pressure loop's maximum (the setpoint is held inside it from the next update);
        a call without it leaves the loops alone."""
        for key, value in values.items():
            low, high = SLOW_PARAMETER_RANGES[key]
            if not low <= value <= high:
                raise ValueError(f"{key} = {value!r} outside [{low}, {high}]")
        if "unoccupied_lead_min" in values:
            values["unoccupied_lead_min"] = int(round(values["unoccupied_lead_min"]))
        self.cfg = cfg = replace(self.cfg, **values)
        if "sp_max_pa" in values:
            self.static_pressure.maximum = float(cfg.sp_max_pa)
            for loop in self.static_pressure.loops:
                loop.maximum = max(loop.minimum, float(cfg.sp_max_pa))
        window = self.index.comfort_window
        self.occupied_targets = (window[0] + cfg.occupied_margin_k, window[1] - cfg.occupied_margin_k)
        self.optimal_start.heating_setpoint = float(self.occupied_targets[0])
        self.optimal_start.cooling_setpoint = float(window[0] + cfg.precool_k if cfg.precool_until_hour > 0
                                                    else self.occupied_targets[1])
        self.optimal_start.t_opt_max = float(cfg.t_opt_max_s)

    def act(self, obs):
        cfg, ix, c = self.cfg, self.index, self.counters
        groups = ix.zone_group
        o = ix.parse(obs)
        temp = o.zone_temp
        pump_stopped_last = self.hw_pump_stop
        if cfg.arrival_learn:
            self._arrival(o)
        if (cfg.capacity_start_extra_s > 0.0 or cfg.start_offset_s != 0.0) and not self.capacity_ready:
            self._capacity_extra()
        occupied = is_occupied(o.minute_of_day, o.weekday, self.morning_hour,
                               ix.evening_start_hour)
        if cfg.arrival_learn and float(o.occupants) > 0 and is_occupied(
                o.minute_of_day, o.weekday, ix.morning_start_hour, ix.evening_start_hour):
            occupied = True
        if cfg.empty_day_delay_min > 0:
            occupied = self._empty_day(o.minute_of_day, o.occupants, occupied)
        if cfg.departure_learn:
            occupied = self._departure(o.minute_of_day, o.occupants, occupied)
        climate_gates = (cfg.night_flush_min_range_k > 0.0 or cfg.night_flush_min_max_k > 0.0
                         or cfg.precool_min_range_k > 0.0 or cfg.precool_min_max_k > 0.0
                         or cfg.opt_start_break_min_oat_k > 0.0 or cfg.sp_options_max_oat_k > 0.0
                         or cfg.hw_off_setpoint == "low_warm" or cfg.sat_cost_max_min_range_k > 0.0
                         or cfg.humid_low_sat_k > 0.0)
        if climate_gates:
            self._track_outside_air(o.minute_of_day, o.oat)
        if self._uses_oat_24h():
            self.oat_day.append(float(np.mean(o.oat)))
            self.oat_24h = sum(self.oat_day) / len(self.oat_day)
        if cfg.margin_feedback_step_k > 0.0:
            self._margin_feedback(o, temp)
        if cfg.cooldown_learn_step_k > 0.0:
            self._cooldown_learning(o, temp)
        if cfg.opt_start_break_max_s > 0.0:
            minute = int(o.minute_of_day)
            if self.break_minute is not None and minute < self.break_minute:
                self.previous_day_people, self.day_people = self.day_people, False
            self.break_minute = minute
            self.day_people = self.day_people or float(o.occupants) > 0
            after_break = not self.previous_day_people
            if cfg.opt_start_break_min_oat_k > 0.0:
                after_break = after_break and self.oat_yesterday is not None and \
                    self.oat_yesterday[2] >= cfg.opt_start_break_min_oat_k
            self.optimal_start.t_opt_max = float(max(cfg.t_opt_max_s, cfg.opt_start_break_max_s) if after_break
                                                 else cfg.t_opt_max_s)
        if cfg.precool_until_hour > 0 and (cfg.precool_min_range_k > 0.0 or cfg.precool_min_max_k > 0.0):
            precool_today = self._yesterday_gate(cfg.precool_min_range_k, cfg.precool_min_max_k)
            self.optimal_start.cooling_setpoint = float(ix.comfort_window[0] + cfg.precool_k if precool_today
                                                        else self.occupied_targets[1])
        occupied_targets = self._occupied_targets()
        if cfg.capacity_start_heat_only:
            self.optimal_start.heat_extra_gate = (
                np.broadcast_to(np.asarray(o.oat, dtype=float), (ix.n_ahu,)) < occupied_targets[0])
        warm, cool = self.optimal_start.update(temp, o.minute_of_day, o.weekday)
        if self._planner_on:
            try:
                warm, cool = self._planners(o, temp, occupied, occupied_targets, warm, cool)
            except Exception as error:
                self.planner_error, self._planner_on = repr(error), False
                self.ds_flow, self.lp_plan, self.lp_only = None, None, None
        eco_heat, eco_cool = ix.eco_window
        setback_drop = cfg.setback_heat_drop_k + (self.layer_depth if cfg.daily_layer and self._daily_gate()
                                                  else 0.0)
        if setback_drop > 0.0:
            eco_heat = eco_heat - setback_drop
        setback, setup = self.setback_setup.update(temp, occupied, eco_heat, eco_cool)
        raw_setup = setup
        if cfg.unoccupied_conditioning != "always" and o.occupants <= 0:
            if cfg.unoccupied_conditioning == "occupancy_hours":
                start = self.morning_hour * 60 - cfg.unoccupied_lead_min
                allowed = start <= o.minute_of_day < ix.evening_start_hour * 60
            else:
                allowed = seconds_to_next_occupied(o.minute_of_day, o.weekday, self.morning_hour,
                                                   ix.evening_start_hour) <= cfg.unoccupied_lead_min * 60
            if not allowed:
                setback, setup = np.zeros_like(setback), np.zeros_like(setup)
        if cfg.empty_day_setup == "night":
            if float(o.occupants) > 0:
                self.empty_hold = False
            elif self.empty_day:
                self.empty_hold = True
            to_start = seconds_to_next_occupied(o.minute_of_day, o.weekday, self.morning_hour,
                                                ix.evening_start_hour)
            if self.empty_hold and float(o.occupants) <= 0 and 0 < to_start <= cfg.empty_day_setup_lead_min * 60:
                setup = raw_setup
            elif self.empty_day:
                setup = np.zeros_like(setup)
        freeze = self.freeze_protection.update(temp, np.asarray(occupied) | warm | cool)
        mode = operating_mode(occupied, cool, warm, setup, setback | freeze)
        heat_g, cool_g = zone_group_setpoints(mode, occupied_targets, ix.eco_window)
        freeze_only = (mode == SETBACK) & np.asarray(freeze, dtype=bool) & ~np.asarray(setback, dtype=bool)
        if setback_drop > 0.0:
            heat_g = np.where((mode == SETBACK) & ~freeze_only, eco_heat, heat_g)
        heat_sp, cool_sp = heat_g[groups], cool_g[groups]
        if (cfg.freeze_min_airflow or cfg.freeze_heat_mode != "rule") and freeze_only.any():
            heat_sp = np.where(freeze_only[groups], np.minimum(heat_sp, cfg.freeze_off_k + 0.5), heat_sp)
            if cfg.freeze_coldest_only and cfg.freeze_heat_mode == "rule":
                coldest = np.zeros(ix.n_zones, dtype=bool)
                for g in np.flatnonzero(freeze_only):
                    s = ix.group_slices[g]
                    coldest[s.start + int(np.argmin(temp[s]))] = True
                heat_sp = np.where(freeze_only[groups], np.where(coldest, np.minimum(
                    heat_sp, cfg.freeze_on_k + cfg.freeze_hold_k), np.minimum(heat_sp, cfg.freeze_on_k)), heat_sp)
        error = ahu.group_errors(temp, ix.group_slices, heat_g, cool_g, cfg.ahu_zone_aggregate)
        fan_on = ahu.fan_command(cfg.fan_control, mode=mode, error=error)
        unoccupied = (mode == UNOCCUPIED)[groups]
        if (cfg.evening_charge_min > 0 and o.weekday < 4 and float(np.mean(o.oat)) < 283.15
                and o.minute_of_day >= ix.evening_start_hour * 60 - cfg.evening_charge_min):
            charge = (mode == OCCUPIED)[groups]
            heat_sp = np.where(charge, np.maximum(heat_sp, ix.comfort_window[0] + cfg.evening_charge_k), heat_sp)
        if (cfg.precool_until_hour > 0 and o.weekday < 5 and o.minute_of_day < cfg.precool_until_hour * 60
                and self._yesterday_gate(cfg.precool_min_range_k, cfg.precool_min_max_k)):
            warm = ((np.asarray(o.oat, dtype=float) > 291.15) & (mode == OCCUPIED))[groups]
            cool_sp = np.where(warm, np.minimum(cool_sp, ix.comfort_window[0] + cfg.precool_k), cool_sp)
        room_stagger = self.room_settings.get("stagger") if self.room_settings is not None else None
        room_stagger = room_stagger if room_stagger is not None and np.any(room_stagger > 0) else None
        if cfg.opt_start_zone_stagger or room_stagger is not None:
            to_start = seconds_to_next_occupied(o.minute_of_day, o.weekday, self.morning_hour,
                                                ix.evening_start_hour)
            waiting = (mode == WARM_UP)[groups] & (to_start > self.optimal_start.heating.zone_t_opt)
            if not cfg.opt_start_zone_stagger:
                waiting = waiting & (room_stagger > 0)
            heat_sp = np.where(waiting, np.minimum(heat_sp, ix.eco_window[0]), heat_sp)
            heating_start = self.optimal_start.heating
            restart = waiting & heating_start.tracking
            heating_start.elapsed = np.where(restart, 0.0, heating_start.elapsed)
            heating_start.start_deviation = np.where(restart, self.optimal_start.heating_setpoint - temp,
                                                     heating_start.start_deviation)
        day_start = self.morning_hour * 60
        if cfg.arrival_hold_min > 0 and day_start <= o.minute_of_day < day_start + cfg.arrival_hold_min:
            hold = (mode == OCCUPIED)[groups]
            heat_sp = np.where(hold, np.maximum(heat_sp, np.minimum(heat_sp + cfg.arrival_hold_k, cool_sp - 0.5)),
                               heat_sp)
        if cfg.warmup_target_add_k > 0.0:
            warming = (mode == WARM_UP)[groups]
            heat_sp = np.where(warming, np.maximum(heat_sp, np.minimum(heat_sp + cfg.warmup_target_add_k,
                                                                       cool_sp - 0.5)), heat_sp)
        if cfg.cooldown_learn_step_k > 0.0:
            cooling_down = (mode == COOL_DOWN)[groups]
            cool_sp = np.where(cooling_down, np.maximum(cool_sp - self.cooldown_offset, heat_sp + 0.5), cool_sp)
        if (cfg.midday_charge_k > 0.0 and cfg.midday_charge_start_hour * 60 <= o.minute_of_day
                < cfg.midday_charge_end_hour * 60 and float(np.mean(o.oat)) < cfg.midday_charge_oat_k):
            charge = (mode == OCCUPIED)[groups]
            target = np.minimum(ix.comfort_window[0] + cfg.midday_charge_k, cool_sp - 0.5)
            heat_sp = np.where(charge, np.maximum(heat_sp, target), heat_sp)
        if cfg.occupied_cycling:
            low, high = ix.comfort_window
            worst = np.array([np.minimum(temp[s] - low, high - temp[s]).min() for s in ix.group_slices])
            occupied_group = mode == OCCUPIED
            if cfg.cycle_window_min > 0:
                occupied_group = occupied_group & (
                    o.minute_of_day >= ix.evening_start_hour * 60 - cfg.cycle_window_min)
            start = occupied_group & ~self.cycle_on & (worst < cfg.cycle_on_k)
            stop = (occupied_group & self.cycle_on & (worst > cfg.cycle_off_k)
                    & (self.cycle_steps >= cfg.cycle_min_on_steps))
            if cfg.cycle_vent_ledger:
                vent = self._vent_requirement()
                if vent is None:
                    stop = np.zeros_like(stop)
                else:
                    vot, t_avg = vent
                    max_off = cfg.vent_max_off_frac * t_avg
                    delivered = np.where(o.ahu_on, np.asarray(o.ahu_flow, dtype=float) * (1.0 - np.clip(
                        np.asarray(o.recirculation, dtype=float), 0.0, cfg.max_recirculation)), 0.0)
                    self.vent_bank = np.where(mode == OCCUPIED, np.minimum(
                        self.vent_bank + (delivered - vot) * 300.0, vot * max_off), 0.0)
                    credit = self.vent_bank >= vot * 300.0
                    stop = stop & credit
                    start = start | (occupied_group & ~self.cycle_on
                                     & (~credit | (self.cycle_steps * 300.0 >= max_off)))
            if cfg.cycle_no_heating:
                heating_group = np.bincount(groups, weights=self.last_heating_state.astype(float),
                                            minlength=ix.n_ahu) > 0.0
                stop = stop & ~heating_group
            self.cycle_on = np.where(occupied_group, (self.cycle_on | start) & ~stop, True)
            self.cycle_steps = np.where(start | stop | ~occupied_group, 0, self.cycle_steps + 1)
            coast = occupied_group & ~self.cycle_on
            charging = (occupied_group & self.cycle_on)[groups]
            if cfg.cycle_charge_k > 0.0:
                heat_sp = np.where(charging, np.maximum(heat_sp, low + cfg.cycle_charge_k), heat_sp)
                cool_sp = np.where(charging, np.minimum(cool_sp, high - cfg.cycle_charge_k), cool_sp)
            fan_on = fan_on & ~coast
            unoccupied = unoccupied | coast[groups]
        if self.room_settings is not None and "heat_margin" in self.room_settings:
            occupied_room = (mode == OCCUPIED)[groups]
            heat_sp = np.where(occupied_room, heat_sp + self.room_settings["heat_margin"], heat_sp)
            cool_sp = np.where(occupied_room, cool_sp - self.room_settings["cool_margin"], cool_sp)
            heat_sp = np.minimum(heat_sp, cool_sp - 0.1)
        reserve = self.room_settings.get("margin_reserve") if self.room_settings is not None else None
        if reserve is not None and np.any(reserve > 0):
            occ = (mode == OCCUPIED)[groups]
            t_now = np.asarray(temp, dtype=float)
            short = np.maximum(np.maximum(heat_sp - t_now, t_now - cool_sp), 0.0)
            self.reserve_err = np.where(occ, 0.94 * self.reserve_err + 0.06 * short, self.reserve_err)
            extra = np.clip(0.5 * (self.reserve_err - 0.08), 0.0, 0.25) * np.clip(reserve, 0.0, 1.0)
            heat_sp = np.where(occ, heat_sp + extra, heat_sp)
            cool_sp = np.where(occ, cool_sp - extra, cool_sp)
            heat_sp = np.minimum(heat_sp, cool_sp - 0.1)
        if cfg.economic_target_k > 0.0:
            minute = int(o.minute_of_day)
            if self.econ_minute is not None and minute < self.econ_minute:
                cool_n, heat_n = self.econ_counts[:, 0], self.econ_counts[:, 1]
                self.econ_side = np.where(cool_n > heat_n, 1.0, np.where(heat_n > cool_n, -1.0, 0.0))
                self.econ_counts[:] = 0.0
            self.econ_minute = minute
            occupied_econ = (mode == OCCUPIED)[groups]
            low, high = ix.comfort_window
            keep = cfg.economic_keep_k
            if (cfg.economic_keep_cold_k > 0.0 and self.oat_24h is not None
                    and self.oat_24h < cfg.economic_keep_cold_oat24_k):
                keep = cfg.economic_keep_cold_k
            cool_up = np.maximum(cool_sp, np.minimum(cool_sp + cfg.economic_target_k, high - keep))
            heat_down = np.minimum(heat_sp, np.maximum(heat_sp - cfg.economic_target_k, low + keep))
            cool_sp = np.where(occupied_econ & (self.econ_side > 0), cool_up, cool_sp)
            heat_sp = np.where(occupied_econ & (self.econ_side < 0), heat_down, heat_sp)
        if self.lp_plan is not None and self.lp_only is not None:
            plan = self.lp_plan
            warm_only, cool_only = self.lp_only
            warm_z = ((mode == WARM_UP) & warm_only)[groups]
            cool_z = ((mode == COOL_DOWN) & cool_only)[groups]
            t_end = plan["t_h"][:, self.lp_hour]
            heat_cap = np.maximum(t_end - plan["heat_extra"], ix.eco_window[0])[groups]
            cool_cap = np.minimum(t_end + plan["cool_extra"], ix.eco_window[1])[groups]
            heat_sp = np.where(warm_z, np.minimum(heat_sp, heat_cap), heat_sp)
            cool_sp = np.where(cool_z, np.maximum(cool_sp, cool_cap), cool_sp)

        loop_temp = temp
        if cfg.zone_temp_lead > 0.0:
            if self.previous_temp is not None:
                correction = cfg.zone_temp_lead * (temp - self.previous_temp)
                if cfg.zone_temp_lead_max_k > 0.0:
                    correction = np.clip(correction, -cfg.zone_temp_lead_max_k, cfg.zone_temp_lead_max_k)
                if cfg.zone_temp_lead_steady:
                    steady = (np.zeros(ix.n_ahu, dtype=bool) if self.last_mode is None else
                              (mode == self.last_mode) & np.asarray(fan_on, dtype=bool)
                              & np.asarray(self.last_fan, dtype=bool))
                    correction = np.where(steady[groups], correction, 0.0)
                loop_temp = temp + correction
            self.previous_temp = np.array(temp, dtype=float, copy=True)
        if cfg.disturbance_ff_steps > 0.0:
            disturbance = self._disturbance(o, temp)
            occupied_ff = (mode == OCCUPIED)[groups]
            if cfg.disturbance_highpass_steps > 0.0:
                self.dist_slow = self.dist_slow + (disturbance - self.dist_slow) / cfg.disturbance_highpass_steps
                disturbance = disturbance - self.dist_slow
            lead = np.clip(cfg.disturbance_ff_steps * disturbance, -0.5, 0.5)
            loop_temp = np.where(occupied_ff, loop_temp + lead, loop_temp)
            if cfg.disturbance_ff_cool_k > 0.0:
                cool_sp = np.where(occupied_ff, np.minimum(cool_sp + cfg.disturbance_ff_cool_k,
                                                           ix.comfort_window[1]), cool_sp)
        if cfg.freeze_heat_mode != "rule":
            start = np.asarray(freeze_only, dtype=bool) & ~self.prev_freeze_only
            if start.any():
                self.zone_loops.heating.reset(start[groups] & (np.asarray(temp, dtype=float) >= heat_sp), 0.0)
            self.prev_freeze_only = np.array(freeze_only, dtype=bool, copy=True)
        heating, cooling = self.zone_loops.update(loop_temp, heat_sp, cool_sp)
        cooling_state, heating_state = self.zone_states.update(heating, cooling)
        if cfg.economic_target_k > 0.0:
            occupied_econ = (mode == OCCUPIED)[groups]
            self.econ_counts[:, 0] += occupied_econ & np.asarray(cooling_state, dtype=bool)
            self.econ_counts[:, 1] += occupied_econ & np.asarray(heating_state, dtype=bool)

        return_temp = np.array([temp[s].mean() for s in ix.group_slices])
        if cfg.return_air_area_weighted:
            rooms_area = self._room_quantities()
            if rooms_area is not None:
                area = rooms_area[0]
                return_temp = np.array([float(np.sum(temp[s] * area[s])) / max(float(np.sum(area[s])), 1e-9)
                                        for s in ix.group_slices])
        supply_est = zone.estimate_supply_air(return_temp, o.oat, o.recirculation, o.sat_setpoint,
                                              o.ahu_on)
        box_pressure = o.static_pressure[groups]
        ratio = zone.supply_flow_ratio(zone.box_flow_demand(o.damper, box_pressure), groups, ix.n_ahu)
        discharge_est = zone.estimate_discharge(o.damper, o.valve, box_pressure, ratio[groups],
                                                supply_est[groups], o.water_temp, o.water_dp)

        suppressed = self.suppression.update(cool_sp, temp)
        cooling_requests, self.cooling_latch = zone_requests.cooling_sat_requests(
            temp - cool_sp, cooling, self.cooling_latch, suppressed,
            three_above=cfg.cooling_three_k, two_above=cfg.cooling_two_k,
            loop_on=cfg.cooling_loop_on, loop_off=cfg.cooling_loop_off)

        def per_group(values):
            return np.bincount(groups, weights=values, minlength=ix.n_ahu)

        self.occupied_min, airflow_floor = self._dcv_minimum(o.occupants, o.static_pressure)
        low_flow_on = cfg.cold_low_flow_oat_k > 0.0 or cfg.humid_low_sat_k > 0.0 or cfg.low_load_max_loop > 0.0
        cold_flow = humid = None
        if low_flow_on:
            occupied_group = mode == OCCUPIED
            cold_flow = occupied_group & (self.oat_24h is not None and cfg.cold_low_flow_oat_k > 0.0
                                          and self.oat_24h < cfg.cold_low_flow_oat_k)
            low = cold_flow.copy()
            if cfg.humid_low_sat_k > 0.0:
                oat_g = np.broadcast_to(np.asarray(o.oat, dtype=float), (ix.n_ahu,))
                narrow = (self.oat_yesterday is not None
                          and self.oat_yesterday[1] - self.oat_yesterday[0] < cfg.humid_max_range_k)
                humid = occupied_group & np.asarray(fan_on, dtype=bool) & (oat_g > return_temp + 2.0) & narrow
                low = low | humid
            if cfg.low_load_max_loop > 0.0:
                heating_any = np.bincount(groups, weights=np.asarray(heating_state, dtype=float),
                                          minlength=ix.n_ahu) > 0.0
                load = np.array([float(np.max(np.where(cooling_state[s], cooling[s], 0.0)))
                                 for s in ix.group_slices])
                low = low | (occupied_group & np.asarray(fan_on, dtype=bool) & ~heating_any
                             & (load < cfg.low_load_max_loop))
            self.low_flow_groups = low
        if cfg.sat_guard_trend_s > 0.0:
            self.temp_history.append(np.array(temp, dtype=float, copy=True))
        sat = self.supply_air.update(per_group(cooling_requests), fan_on, o.oat, mode=mode,
                                     error=error)
        if cfg.sat_cost_optimal:
            guard_sp = cool_sp
            relax = self.room_settings.get("cool_relax") if self.room_settings is not None else None
            if relax is not None and np.any(relax > 0.0):
                guard_sp = np.where((mode == OCCUPIED)[groups], cool_sp - relax, cool_sp)
            keep = max(cfg.sat_guard_keep_margin_k - cfg.occupied_margin_k, 0.0) if cfg.sat_guard_keep_margin_k > 0.0 \
                else 0.0
            if keep > 0.0:
                guard_sp = np.where((mode == OCCUPIED)[groups], guard_sp - keep, guard_sp)
            sat = self._cost_based_sat(sat, temp, cooling, cooling_state, return_temp, o, fan_on, mode,
                                       cool_sp=guard_sp, requests=per_group(cooling_requests))
        if cfg.sat_rate_limit_k > 0.0:
            running = np.asarray(fan_on, dtype=bool) & (mode == OCCUPIED)
            if self.last_sat is not None:
                last, was_running = self.last_sat
                sat = np.where(running & was_running, np.clip(sat, last - cfg.sat_rate_limit_k,
                                                              last + cfg.sat_rate_limit_k), sat)
            self.last_sat = (np.array(sat, dtype=float, copy=True), running)
        if humid is not None and humid.any():
            sat = np.where(humid, np.minimum(sat, cfg.humid_low_sat_k), sat)
        sat_zone = sat[groups]

        gate = zone.discharge_gate(discharge_est, temp + cfg.discharge_zone_margin_k, self.discharge_gate,
                                   cfg.discharge_gate_hysteresis_k)
        warm_supply = zone.greater(supply_est[groups] - temp, cfg.warm_supply_threshold_k, self.warm_supply,
                                   cfg.warm_supply_hysteresis_k)
        cooling_max, minimum, heating_min, heating_max = (limit[groups] for limit in zone.mode_airflow_limits(
            mode, cooling_max=cfg.cooling_max_flow, heating_max=cfg.heating_max_ratio * cfg.cooling_max_flow,
            occupied_min=self.occupied_min, heating_min=self.occupied_min, floor=airflow_floor,
            setback_warmup_heating=cfg.setback_warmup_heating_airflow,
            zero_entries=cfg.table_zero_airflow))
        low_flow_zone = self.low_flow_groups[groups] if low_flow_on else None
        if low_flow_zone is not None and low_flow_zone.any():
            floor_zone = (np.full(ix.n_zones, float(airflow_floor)) if np.ndim(airflow_floor) == 0
                          else np.asarray(airflow_floor, dtype=float)[:, 0][groups])
            if cfg.cold_low_flow_deadband_only:
                lowered = low_flow_zone & ~np.asarray(heating_state, dtype=bool)
                if cfg.cold_low_flow_margin_k > 0.0:
                    lowered = lowered & (np.asarray(temp, dtype=float) >= heat_sp + cfg.cold_low_flow_margin_k)
                minimum = np.where(lowered, floor_zone, minimum)
            else:
                minimum = np.where(low_flow_zone, floor_zone, minimum)
                heating_min = np.where(low_flow_zone, floor_zone, heating_min)
        if cfg.warmup_heat_max_ratio > 0.0:
            warming_zone = (mode == WARM_UP)[groups]
            heating_max = np.where(warming_zone, np.maximum(cfg.warmup_heat_max_ratio * cfg.cooling_max_flow,
                                                            heating_min), heating_max)
        if (cfg.big_room_heat_max_ratio > 0.0 and self.oat_24h is not None
                and self.oat_24h < cfg.big_room_oat24_k):
            rooms = self._room_quantities()
            if rooms is not None:
                big = np.asarray(rooms[0], dtype=float) >= cfg.big_room_min_area_m2
                window = ((mode == WARM_UP)[groups] | ((mode == OCCUPIED)[groups]
                                                       & (o.minute_of_day < cfg.big_room_until_hour * 60)))
                if cfg.big_room_need:
                    window = window & (np.asarray(heating_state, dtype=bool) & (np.asarray(heating) >= 0.99)
                                       & (np.asarray(temp, dtype=float) < heat_sp))
                heating_max = np.where(big & window, np.maximum(heating_max, cfg.big_room_heat_max_ratio
                                                                * cfg.cooling_max_flow), heating_max)
        if self.ds_flow is not None:
            chosen = (mode == WARM_UP)[groups] & np.isfinite(self.ds_flow)[groups]
            if chosen.any():
                heating_max = np.where(chosen, np.maximum(np.nan_to_num(self.ds_flow)[groups] * cfg.cooling_max_flow,
                                                          heating_min), heating_max)
        geometry_modes = np.isin(mode, (OCCUPIED, COOL_DOWN))[groups]
        if self.zone_heat_factor is not None and not cfg.zone_heat_redistribute:
            heating_max = np.where(geometry_modes, np.clip(heating_max * self.zone_heat_factor, heating_min,
                                                           cooling_max), heating_max)
        if cfg.zone_heat_boost_max > 1.0:
            boost_zone = (mode == OCCUPIED)[groups]
            short = (boost_zone & np.asarray(heating_state, dtype=bool) & (np.asarray(heating) >= 0.99)
                     & (temp < heat_sp))
            step = np.where(short, cfg.zone_heat_boost_step, -cfg.zone_heat_boost_step)
            self.heat_boost = np.where(boost_zone, np.clip(self.heat_boost + step, 1.0, cfg.zone_heat_boost_max),
                                       self.heat_boost)
            heating_max = np.where(boost_zone, np.clip(heating_max * self.heat_boost, heating_min, cooling_max),
                                   heating_max)
        rs = self.room_settings
        discharge_offset = cfg.discharge_offset_k
        if rs is not None and "heat_max_ratio" in rs and np.any(rs["heat_max_ratio"] > 0.0):
            on_ratio = (mode == OCCUPIED)[groups] & (rs["heat_max_ratio"] > 0.0)
            heating_max = np.where(on_ratio, np.clip(rs["heat_max_ratio"] * cooling_max, heating_min, cooling_max),
                                   heating_max)
        if rs is not None and "discharge_offset" in rs and np.any(rs["discharge_offset"] > 0.0):
            discharge_offset = np.where((mode == OCCUPIED)[groups] & (rs["discharge_offset"] > 0.0),
                                        rs["discharge_offset"], cfg.discharge_offset_k)
        rooms = (self._room_quantities() if rs is not None or cfg.zone_min_ra_lps_m2 > 0.0
                 or cfg.zone_heat_wall_ref_m > 0.0 else None)
        if rooms is not None:
            area, wall = rooms
            occupied_now = (mode == OCCUPIED)[groups]
            ra, rp, ref, power = ((rs["min_ra"], rs["min_rp"], rs["wall_ref"], rs["wall_power"]) if rs is not None
                                  else (cfg.zone_min_ra_lps_m2, cfg.zone_min_rp_lps_m2, cfg.zone_heat_wall_ref_m,
                                        cfg.zone_heat_wall_power))
            on_min = np.broadcast_to(np.asarray(ra) > 0.0, area.shape)
            if on_min.any():
                floor = (np.full(ix.n_zones, float(airflow_floor)) if np.ndim(airflow_floor) == 0
                         else np.asarray(airflow_floor, dtype=float)[:, 0][groups])
                lps = (ra + rp * self.occupancy_ratio) * area
                divide = (np.broadcast_to(np.asarray(rs["oa_divide"], dtype=bool), area.shape) if rs is not None
                          else np.full(ix.n_zones, bool(cfg.zone_min_oa_divide)))
                if divide.any():
                    oa_fraction = np.clip(1.0 - np.asarray(o.recirculation, dtype=float),
                                          1.0 - cfg.max_recirculation, 1.0)[groups]
                    ez = np.where(np.asarray(heating_state, dtype=bool), 0.8, 1.0)
                    lps = np.where(divide, lps / (ez * oa_fraction), lps)
                room_min = np.minimum(np.maximum(lps / (1000.0 * zone.VAV_MAX_FLOW), floor), cooling_max)
                scoped = (occupied_now & on_min
                          & {"all": True, "core": wall <= 0.0, "perimeter": wall > 0.0}[cfg.zone_min_scope])
                minimum = np.where(scoped, room_min, minimum)
                heating_min = np.where(scoped, room_min, heating_min)
                heating_max = np.where(scoped, np.maximum(heating_max, room_min), heating_max)
            on_wall = np.broadcast_to(np.asarray(ref) > 0.0, area.shape)
            if on_wall.any():
                factor = np.where(on_wall, np.maximum(1.0, wall / np.where(on_wall, ref, 1.0)) ** power, 1.0)
                heating_max = np.where(geometry_modes & on_wall,
                                       np.clip(heating_max * factor, heating_min, cooling_max), heating_max)
        if cold_flow is not None and cfg.cold_low_flow_heat_max_scale != 1.0 and cold_flow.any():
            cold_zone = cold_flow[groups]
            heating_max = np.where(cold_zone, np.maximum(heating_max * cfg.cold_low_flow_heat_max_scale,
                                                         heating_min), heating_max)
        ramp_sat = sat_zone
        if cfg.heating_ramp_from_supply:
            ramp_sat = np.minimum(sat_zone, np.maximum(supply_est[groups], 283.15))
        flow, heating_discharge_sp = zone.dual_maximum(
            heating, cooling, ramp_sat, heat_sp, cooling_state=cooling_state, heating_state=heating_state,
            gate=gate, supply_hot=warm_supply, min_flow=minimum,
            cooling_max_flow=cooling_max, heating_min_flow=heating_min, heating_max_flow=heating_max,
            discharge_offset=discharge_offset, discharge_limit=cfg.discharge_limit_k)
        if self.zone_heat_factor is not None and cfg.zone_heat_redistribute:
            flow = self._redistribute(flow, heating_state, cooling_state, heating_max, cooling_max, minimum,
                                      geometry_modes)
        if cfg.ahu_heat_budget < 1.0 and float(np.mean(o.oat)) < cfg.ahu_heat_budget_oat_k:
            if cfg.ahu_heat_budget_scope == "occupied":
                start_min = self.morning_hour * 60
                in_window = start_min <= o.minute_of_day < start_min + cfg.ahu_heat_budget_hours * 60
                apply = (mode == OCCUPIED) & in_window
            else:
                lead = seconds_to_next_occupied(o.minute_of_day, o.weekday, self.morning_hour,
                                                ix.evening_start_hour)
                apply = (np.isin(mode, (WARM_UP, SETBACK)) & ~(freeze & ~setback)
                         & (0 < lead <= cfg.ahu_heat_budget_hours * 3600.0))
            if np.any(apply):
                flow = self._heat_budget(flow, temp, heat_sp, heating_state, cooling_state, heating_min,
                                         heating_max, apply)
        layer = self.planner["layer"] if (cfg.daily_layer and self.planner is not None) else None
        if layer is not None:
            deficit = np.array([float(np.max(heat_sp[s] - temp[s])) for s in ix.group_slices])
            apply = (mode == OCCUPIED) & (layer.budget < 1.0) & (deficit <= 0.5)
            apply = apply & self._daily_gate()
            if np.any(apply):
                flow = self._heat_budget(flow, temp, heat_sp, heating_state, cooling_state, heating_min,
                                         heating_max, apply, budgets=layer.budget)
            heating_only = np.asarray(heating_state, dtype=bool) & ~np.asarray(cooling_state, dtype=bool)
            span = np.where(heating_only, np.maximum(heating_max - heating_min, 0.0), 0.0)
            used = np.where(heating_only, np.maximum(flow - heating_min, 0.0), 0.0)
            span_g = np.bincount(groups, weights=span, minlength=ix.n_ahu)
            share_g = np.where(span_g > 1e-9, np.bincount(groups, weights=used, minlength=ix.n_ahu)
                               / np.maximum(span_g, 1e-9), np.nan)
        if cfg.arrival_keep_flow_min > 0:
            minute = int(o.minute_of_day)
            if self.keep_minute is not None and minute < self.keep_minute:
                self.warmup_flow = np.zeros(ix.n_zones)
            self.keep_minute = minute
            self.warmup_flow = np.where((mode == WARM_UP)[groups], flow, self.warmup_flow)
            start_min = self.morning_hour * 60
            if start_min <= minute < start_min + cfg.arrival_keep_flow_min:
                keep = (mode == OCCUPIED)[groups] & np.asarray(heating_state, dtype=bool)
                if cfg.arrival_keep_below_only:
                    keep = keep & (temp < heat_sp)
                kept = self.warmup_flow
                if cfg.arrival_keep_ramp:
                    end_min = start_min + cfg.arrival_keep_flow_min
                    frac = min(max((end_min - minute) / (cfg.arrival_keep_flow_min / 3.0), 0.0), 1.0)
                    kept = flow + frac * np.maximum(self.warmup_flow - flow, 0.0)
                flow = np.where(keep, np.minimum(np.maximum(flow, kept), cooling_max), flow)
        if cfg.setback_warmup_discharge != "ramp":
            floor_modes = (SETBACK,) if cfg.setback_warmup_discharge == "supply_floor_setback" else (WARM_UP, SETBACK)
            floored = np.isin(mode, floor_modes)[groups]
            heating_discharge_sp = np.where(floored, np.maximum(heating_discharge_sp, cfg.warmup_setback_supply_k),
                                            heating_discharge_sp)

        vot_need = None
        if cfg.ahu_vot_check:
            rooms_q = self._room_quantities()
            if rooms_q is not None:
                area = rooms_q[0]
                people = float(o.occupants) * area / max(float(area.sum()), 1e-9)
                ez = np.where(np.asarray(heating_state, dtype=bool), 0.8, 1.0)
                voz = (cfg.vot_rp_lps_person * people + cfg.vot_ra_lps_m2 * area) / 1000.0 / ez
                running = (mode == OCCUPIED) & np.asarray(fan_on, dtype=bool)
                vot_need = np.where(running, np.bincount(groups, weights=voz, minlength=ix.n_ahu), 0.0)
                supply = np.bincount(groups, weights=flow * zone.VAV_MAX_FLOW, minlength=ix.n_ahu)
                short = running & (supply < vot_need) & (supply > 0.0)
                if short.any():
                    scale = np.where(short, vot_need / np.maximum(supply, 1e-9), 1.0)[groups]
                    flow = np.where(short[groups], np.minimum(flow * scale, cooling_max), flow)
                    supply = np.bincount(groups, weights=flow * zone.VAV_MAX_FLOW, minlength=ix.n_ahu)
                vot_supply = supply

        readback = cfg.command_readback == "observed"
        pressure_in_force = o.static_pressure[groups] if readback else self.static_pressure.setpoint[groups]
        demanded = zone.damper_command(flow, pressure_in_force)
        pressure_requests, self.pressure_latch = zone_requests.static_pressure_requests(
            demanded, self.pressure_latch, damper_on=cfg.damper_on, damper_off=cfg.damper_off,
            three_below=cfg.flow_three_below, two_below=cfg.flow_two_below)
        sp_options_on = (cfg.sp_options_max_oat_k <= 0.0 or (self.oat_yesterday is not None
                                                              and self.oat_yesterday[2] < cfg.sp_options_max_oat_k))
        if cfg.sp_rogue_hours > 0.0 and sp_options_on:
            alpha = 1.0 / (cfg.sp_rogue_hours * 12.0)
            asked = (pressure_requests > 0).astype(float)
            self.rogue_mean = np.where(fan_on[groups], self.rogue_mean + alpha * (asked - self.rogue_mean),
                                       self.rogue_mean)
            pressure_requests = np.where(self.rogue_mean > cfg.sp_rogue_share, 0, pressure_requests)
        if cfg.sp_max_setback_pa > 0.0 or cfg.freeze_min_airflow or low_flow_on or layer is not None:
            for g, sp_loop in enumerate(self.static_pressure.loops):
                top = float(cfg.sp_max_pa)
                if cfg.sp_max_setback_pa > 0.0 and sp_options_on and mode[g] in (WARM_UP, SETBACK):
                    top = min(top, cfg.sp_max_setback_pa)
                if cfg.freeze_min_airflow and freeze_only[g]:
                    top = sp_loop.minimum
                if low_flow_on and self.low_flow_groups[g]:
                    top = min(top, cfg.cold_low_flow_sp_pa)
                if (layer is not None and mode[g] == OCCUPIED and np.isfinite(layer.sp_cap[g])
                        and self._daily_gate()):
                    top = min(top, float(layer.sp_cap[g]))
                sp_loop.maximum = max(sp_loop.minimum, top)
        if readback and cfg.static_pressure_reset == "trim_respond":
            for g, loop in enumerate(self.static_pressure.loops):
                if o.ahu_on[g] and loop.steps_on is not None:
                    loop.setpoint = float(o.static_pressure[g])
        pressure = self.static_pressure.update(per_group(pressure_requests), fan_on, flow,
                                               ix.group_slices)
        if layer is not None:
            layer.record((mode == OCCUPIED) & np.asarray(fan_on, dtype=bool), pressure, share_g)
        if cfg.sp_damper_pi:
            if self.sp_pi is None:
                self.sp_pi = np.full(ix.n_ahu, float(cfg.sp_min_pa))
            pi_damper = zone.damper_command(flow, self.sp_pi[groups])
            top = np.array([float(np.max(pi_damper[s])) for s in ix.group_slices])
            caps = np.array([loop.maximum for loop in self.static_pressure.loops])
            moved = np.clip(self.sp_pi + 400.0 * (top - 0.9), cfg.sp_pi_min_pa, np.minimum(cfg.sp_pi_max_pa, caps))
            self.sp_pi = np.where(fan_on, moved, float(cfg.sp_min_pa))
            pressure = np.where(fan_on, self.sp_pi, pressure)
        if cfg.sp_smooth_alpha > 0.0:
            running_sp = np.asarray(fan_on, dtype=bool)
            if self.sp_last is not None:
                last_p, last_on = self.sp_last
                pressure = np.where(running_sp & last_on, cfg.sp_smooth_alpha * pressure
                                    + (1.0 - cfg.sp_smooth_alpha) * last_p, pressure)
            self.sp_last = (np.array(pressure, dtype=float, copy=True), running_sp)
        damper = np.where(unoccupied, 0.0, zone.damper_command(flow, pressure[groups]))

        work_hours = o.weekday < 5 and ix.morning_start_hour <= o.minute_of_day // 60 < ix.evening_start_hour
        u = self.official_loop.update(temp, ix.group_slices, work_hours=work_hours,
                                      comfort_window=ix.comfort_window, eco_window=ix.eco_window)
        gate_open = np.nan_to_num(u, nan=0.0) < 0.0
        supply_air_economizer = cfg.economizer_mode == "supply_air"
        recirculation = ahu.economizer(return_temp, o.oat, sat, pressure, heat_g,
                                       allowed=np.ones_like(gate_open) if supply_air_economizer else gate_open,
                                       k_per_pa=cfg.economizer_k_per_pa,
                                       max_recirculation=cfg.max_recirculation,
                                       heating_floor=not supply_air_economizer)
        if cfg.economizer_hysteresis_k > 0.0:
            oat_g = np.broadcast_to(np.asarray(o.oat, dtype=float), (ix.n_ahu,))
            self.econ_on = np.where(oat_g < return_temp - cfg.economizer_hysteresis_k, True,
                                    np.where(oat_g > return_temp, False, self.econ_on))
            usable = self.econ_on & np.asarray(fan_on, dtype=bool) & ~freeze_only
            recirculation = np.where(usable, recirculation, cfg.max_recirculation)
        if vot_need is not None:
            limit = 1.0 - vot_need / np.maximum(vot_supply, 1e-9)
            recirculation = np.where(vot_need > 0.0,
                                     np.clip(np.minimum(recirculation, limit), 0.0, cfg.max_recirculation),
                                     recirculation)
            self.vot_last = (vot_need, vot_supply, np.array(recirculation, dtype=float))

        if cfg.inversion_supply_air == "same_step":
            supply_now = zone.estimate_supply_air(return_temp, o.oat,
                                                  np.clip(recirculation, 0.0, cfg.max_recirculation), sat,
                                                  fan_on)[groups]
            discharge_test = zone.estimate_discharge(o.damper, o.valve, box_pressure, ratio[groups], supply_now,
                                                     o.water_temp, o.water_dp)
        else:
            supply_now, discharge_test = supply_est[groups], discharge_est

        low_discharge = zone.less(discharge_test, cfg.discharge_min_k, self.low_discharge,
                                  cfg.discharge_min_hysteresis_k)
        occupied_zone = (mode == OCCUPIED)[groups]
        low_discharge_reheat = ~heating_state & occupied_zone & low_discharge
        valve_enabled = heating_state | low_discharge_reheat
        floor_target = cfg.discharge_min_k
        if cfg.discharge_min_winter_k > 0.0:
            floor_target = np.where(supply_now < cfg.discharge_min_k, cfg.discharge_min_winter_k, cfg.discharge_min_k)
        discharge_sp = np.where(heating_state, heating_discharge_sp, floor_target)
        if cfg.setback_warmup_deadband != "valve_off":
            forced = np.isin(mode, (WARM_UP, SETBACK))[groups] & ~heating_state
            target = temp if cfg.setback_warmup_deadband == "neutral" else np.full(np.shape(temp),
                                                                                   cfg.warmup_setback_supply_k)
            if cfg.freeze_min_airflow or cfg.freeze_heat_mode != "rule":
                target = np.where(freeze_only[groups], temp, target)
            if cfg.warmup_deadband_neutral_above_k > 0.0:
                warm_enough = (mode == WARM_UP)[groups] & (
                    np.asarray(temp, dtype=float) >= heat_sp + cfg.warmup_deadband_neutral_above_k)
                target = np.where(warm_enough, temp, target)
            valve_enabled = valve_enabled | forced
            discharge_sp = np.where(forced, target, discharge_sp)
        if cfg.freeze_heat_mode == "reset_below_off_valve_off":
            shut = freeze_only[groups] & ~heating_state & (np.asarray(temp, dtype=float) > cfg.freeze_off_k + 3.0)
            valve_enabled = valve_enabled & ~shut

        if cfg.valve_control == "inverse":
            if self.plant_reset is None:
                commanded_dp = plant.official_pump_pressure(bool(fan_on.any()), dp_min=cfg.dp_min_bar,
                                                            dp_max=cfg.dp_max_bar)
            else:
                commanded_dp = self.plant_reset.pressure_in_force()
            water_dp = inversion_water_dp(cfg.plant_off_valve_inputs, o.hws_on, o.water_dp, commanded_dp)
            if cfg.hw_saw_resume_dp and pump_stopped_last:
                water_dp = commanded_dp
            inversion_damper = zone.applied_damper(damper)
            inversion_pressure = zone.applied_static_pressure(pressure, fan_on)[groups]
            inversion_ratio = zone.supply_flow_ratio(zone.box_flow_demand(inversion_damper, inversion_pressure),
                                                     groups, ix.n_ahu)
            valve = zone.reheat_valve_for(discharge_sp, inversion_damper, inversion_pressure,
                                          inversion_ratio[groups], supply_now, o.water_temp, water_dp)
        else:
            valve = self.valve_loop.update(discharge_sp - discharge_est)
        valve = np.where(valve_enabled & ~unoccupied, valve, 0.0)
        if cfg.no_reheat_oat_k > 0.0:
            warm_air = np.broadcast_to(np.asarray(o.oat, dtype=float), (ix.n_ahu,)) > cfg.no_reheat_oat_k
            if cfg.no_reheat_min_oat24_k > 0.0:
                warm_air = warm_air & bool(self.oat_24h is not None and self.oat_24h >= cfg.no_reheat_min_oat24_k)
            valve = np.where(warm_air[groups], 0.0, valve)
        pump_dp = None
        request_source = valve
        if cfg.pump_dp_reset != "official" and cfg.valve_control == "inverse" and self.plant_reset is None:
            full_dp = plant.official_pump_pressure(bool(fan_on.any()), dp_min=cfg.dp_min_bar, dp_max=cfg.dp_max_bar)
            enabled = valve_enabled & ~unoccupied
            if cfg.no_reheat_oat_k > 0.0:
                enabled = enabled & ~warm_air[groups]
            at_full = np.where(enabled, zone.reheat_valve_for(discharge_sp, inversion_damper, inversion_pressure,
                                                              inversion_ratio[groups], supply_now, o.water_temp,
                                                              full_dp), 0.0)
            if water_dp > 0.0:
                request_source = at_full
            if not fan_on.any():
                pump_dp = full_dp
            elif cfg.pump_dp_reset == "trim_respond":
                pump_dp = self.pump_dp_loop.update(int(np.sum(valve > 0.95)), True)
            else:
                top = float(np.max(at_full)) if at_full.size else 0.0
                pump_dp = float(np.clip(full_dp * (top / cfg.pump_dp_valve_target) ** 2, cfg.pump_dp_min_bar,
                                        cfg.dp_max_bar))
                if bool(o.hws_on):
                    valve = np.where(enabled, zone.reheat_valve_for(discharge_sp, inversion_damper,
                                                                    inversion_pressure, inversion_ratio[groups],
                                                                    supply_now, o.water_temp, pump_dp), 0.0)
            if not fan_on.any():
                self.pump_dp_loop.update(0, False)
        if cfg.night_heat_mode != "off" or cfg.night_cool_mode != "off":
            plan = self._night_conditioning(o, temp, mode, return_temp)
            if plan is not None:
                run, zones_run = plan["run"], plan["run"][groups]
                fan_on = fan_on | run
                sat = np.where(run, plan["sat"], sat)
                pressure = np.where(run, plan["pressure"], pressure)
                recirculation = np.where(run, plan["recirculation"], recirculation)
                damper = np.where(zones_run, plan["damper"], damper)
                valve = np.where(zones_run, plan["valve"], valve)
                request_source = np.where(zones_run, plan["valve"], request_source)
                unoccupied = unoccupied & ~zones_run
                self.counters["night_conditioning_group_steps"] = (
                    self.counters.get("night_conditioning_group_steps", 0) + int(run.sum()))

        request_valve = o.valve if readback else request_source
        hot_water_requests, self.hot_water_latch = zone_requests.hot_water_reset_requests(
            discharge_sp - discharge_est, request_valve, self.hot_water_latch,
            three_above=cfg.hot_water_three_k, two_above=cfg.hot_water_two_k,
            valve_on=cfg.valve_on, valve_off=cfg.valve_off)
        if cfg.hw_request_rogue != "off":
            asking = (hot_water_requests > 0) if cfg.hw_request_rogue == "share" else (hot_water_requests >= 3)
            if bool(o.hws_on):
                self.hw_rogue = self.hw_rogue + (asking.astype(float) - self.hw_rogue) / 288.0
            hot_water_requests = np.where(self.hw_rogue > cfg.hw_rogue_share, 0, hot_water_requests)
        plant_requests, self.plant_latch = zone_requests.heating_plant_requests(
            request_valve, self.plant_latch, valve_on=cfg.plant_valve_on, valve_off=cfg.plant_valve_off)

        if readback and bool(o.hws_on) != self.plant_enable.enabled:
            self.plant_enable.enabled, self.plant_enable.steps_in_state = bool(o.hws_on), 1
        plant_on = self.plant_enable.update(int(plant_requests.sum()), float(np.mean(o.oat)))
        if self.plant_reset is None:
            if readback and o.hws_on and self.water.loop.steps_on is not None:
                self.water.loop.setpoint = float(o.water_setpoint)
            water_target = self.water.target(int(hot_water_requests.sum()), plant_on)
            dp = plant.official_pump_pressure(bool(fan_on.any()), dp_min=cfg.dp_min_bar,
                                              dp_max=cfg.dp_max_bar)
            if pump_dp is not None:
                dp = pump_dp
        else:
            dp, water_target = self.plant_reset.update(int(hot_water_requests.sum()), plant_on)
        water = self.water.limit_rise(water_target, o.water_setpoint)
        if cfg.hw_off_setpoint != "max":
            if plant_on:
                self.hw_on_setpoint, self.hw_off_steps = float(water), 0
            else:
                self.hw_off_steps += 1
                if self.hw_on_setpoint is not None:
                    water = min(water, self.hw_on_setpoint)
                if (cfg.hw_off_setpoint == "low_warm" and self.hw_off_steps * 300.0 >= cfg.hw_off_low_after_s
                        and self.oat_yesterday is not None and self.oat_yesterday[2] >= cfg.hw_off_low_min_oat_k):
                    water = float(cfg.hw_min_k)
        self.hw_pump_stop = False
        if cfg.hw_sawtooth != "off":
            water, self.hw_pump_stop = self._hw_sawtooth(water, valve, o.water_temp)

        if cfg.night_flush:
            soon = seconds_to_next_occupied(o.minute_of_day, o.weekday, self.morning_hour,
                                            ix.evening_start_hour) <= cfg.night_flush_window_s
            flush = ((mode == UNOCCUPIED) & soon & (return_temp > ix.comfort_window[0] + 1.0)
                     & (np.asarray(o.oat, dtype=float) < return_temp - cfg.flush_delta_k)
                     & self._yesterday_gate(cfg.night_flush_min_range_k, cfg.night_flush_min_max_k))
            if flush.any():
                zones_flush = flush[groups]
                fan_on = fan_on | flush
                sat = np.where(flush, 305.0, sat)
                recirculation = np.where(flush, 0.0, recirculation)
                pressure = np.where(flush, cfg.sp_min_pa, pressure)
                damper = np.where(zones_flush, cfg.flush_damper, damper)
                valve = np.where(zones_flush, 0.0, valve)
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
        discrete[ix.hws_run_action] = int(plant_on and not self.hw_pump_stop)

        c["steps"] += 1
        c["fan_on_group_steps"] += int(fan_on.sum())
        c["plant_on_steps"] += int(plant_on)
        c["plant_switches"] += int(self.previous_plant_on is not None and plant_on != self.previous_plant_on)
        c["discharge_gate_toggles"] += int(np.sum(gate != self.discharge_gate))
        c["loop_enable_edges"] = self.zone_loops.enable_edges
        c["loop_masked_zone_steps"] = self.zone_loops.masked_zone_steps
        c["zone_state_changes"] = self.zone_states.changes
        c["freeze_protection_group_steps"] += int(freeze.sum())
        c["low_discharge_reheat_zone_steps"] += int(low_discharge_reheat.sum())
        c["low_discharge_reheat_switches"] += int(np.sum(low_discharge_reheat != self.low_discharge_reheat))
        c["unoccupied_zero_zone_steps"] += int(unoccupied.sum())
        c["warm_supply_min_flow_zone_steps"] += int(np.sum(cooling_state & warm_supply))
        c["suppression_starts"] = self.suppression.starts
        c["economizer_gate_open_group_steps"] += int(gate_open.sum())
        c["economizer_active_group_steps"] += int(np.sum(recirculation < cfg.max_recirculation))
        c["slope_updates"] = self.optimal_start.heating.updates + self.optimal_start.cooling.updates
        for g_mode in mode:
            c[f"mode_{MODE_NAMES[g_mode]}_group_steps"] += 1
        if self.previous_valve is not None:
            c["valve_jumps_over_0_9"] += int(np.sum(np.abs(valve - self.previous_valve) > 0.9))
        if o.occupants > 0:
            comfort_low, comfort_high = ix.comfort_window
            c["occupied_steps"] += 1
            c["occupied_zone_steps_above_comfort"] += int(np.sum(temp > comfort_high))
            c["occupied_zone_steps_below_comfort"] += int(np.sum(temp < comfort_low))
            c["occupied_kelvin_steps_above_comfort"] += float(np.sum(np.maximum(temp - comfort_high, 0.0)))
            c["occupied_kelvin_steps_below_comfort"] += float(np.sum(np.maximum(comfort_low - temp, 0.0)))
        self.discharge_gate, self.warm_supply, self.low_discharge = gate, warm_supply, low_discharge
        self.low_discharge_reheat = low_discharge_reheat
        self.previous_valve = valve
        self.previous_plant_on = plant_on
        self.last_heating_state = np.asarray(heating_state, dtype=bool).copy()
        self.last_mode, self.last_fan = np.array(mode, copy=True), np.array(fan_on, dtype=bool, copy=True)
        return {"continuous_action": ix.normalize(native), "discrete_action": discrete}


V3_OVERRIDES = {"plant_lockout_k": None}

V3_CANDIDATES = {
    "loops_always_active": {"loop_activation": "always_active"},
    "same_step_supply": {"inversion_supply_air": "same_step"},
    "ti900": {"zone_ti_s": 900.0},
}


def v3_candidate_config(name):
    """G36Config of a WP3 candidate: b1_r_v3 with the candidate's switch."""
    return replace(G36Config(), **{**V3_OVERRIDES, **V3_CANDIDATES[name]})


class G36PolicyV2:
    """b1_r_v2 with the competition interface; registered as POLICIES["b1_r_v2"]."""

    name = "b1_r_v2"

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
        self._stem = None
        if self._sink is not None:
            self._stem = f"g36-{os.getpid()}-{next(_DUMP_SERIAL)}-{id(self):x}"
            self.diagnostics["policy"] = self.name
            try:
                self.diagnostics["building"] = {
                    "time_zone": str(building_meta_data.get("time_zone")),
                    "zones": len(building_meta_data["zone_temp_indices"]),
                    "ahus": len(building_meta_data["ahu_zone_assignments"])}
            except Exception:
                self.diagnostics["building"] = None
        self._fallback = None
        self._fallback_error = None
        try:
            from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
            self._fallback = PIDWaterResetPolicy(building_meta_data, action_spec)
        except Exception as error:
            self._fallback_error = error
        self._shadow_action = None
        self.diagnostics["shadow_errors"] = 0
        self._controller = None
        try:
            self._controller = G36Controller(building_meta_data, action_spec, self.config)
        except Exception as error:
            self.diagnostics["construction_error"] = "".join(traceback.format_exception(error))
            LOGGER.warning("B1-R could not be built; pid_water is used for every step:\n%s",
                           self.diagnostics["construction_error"])
        self.diagnostics["init_seconds"] = time.perf_counter() - started


    def _finalize(self, action):
        """Check shapes and finite values, then clip to the action spec."""
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
            LOGGER.warning("B1-R failed at step %d; pid_water used:\n%s",
                           self.diagnostics["steps"], "".join(traceback.format_exception(error)))

    def _shadow(self, obs):
        """See README.md."""
        self._shadow_action = None
        if self._fallback is None:
            return
        try:
            self._shadow_action = self._fallback.get_action(obs)
        except Exception as error:
            self.diagnostics["shadow_errors"] += 1
            self._fallback_error = error

    def _fallback_action(self):
        if self._shadow_action is None:
            raise RuntimeError("B1-R failed and pid_water is unavailable") from self._fallback_error
        self.diagnostics["fallback_steps"] += 1
        return self._finalize(self._shadow_action)

    def set_rule_parameters(self, **values):
        """Baseline v2 B2-RL: G36Controller.set_parameters on this policy's controller (RULE_PARAMETER_RANGES)."""
        if self._controller is None:
            raise RuntimeError("B1-R could not be built; there are no rule parameters to set")
        self._controller.set_parameters(**values)

    def get_action(self, obs):
        started = time.perf_counter()
        self._shadow(obs)
        action = None
        if self._controller is not None:
            try:
                action = self._finalize(self._controller.act(obs))
            except Exception as error:
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
        stem = self._stem
        np.save(self._sink / f"{stem}-durations.npy", durations)
        with open(self._sink / f"{stem}.json", "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(summary, indent=2) + "\n")


class G36PolicyV3(G36PolicyV2):
    """    POLICIES["b1_r_v3"]. Every other setting is the b1_r_v2 default."""

    name = "b1_r_v3"

    def __init__(self, building_meta_data, action_spec, config=None):
        super().__init__(building_meta_data, action_spec,
                         config if config is not None else replace(G36Config(), **V3_OVERRIDES))


class _V3Candidate(G36PolicyV3):
    """b1_r_v3 with one WP3 candidate switch of V3_CANDIDATES (competition constructor signature)."""

    candidate = None

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec, v3_candidate_config(self.candidate))


class G36PolicyV3LoopsAlwaysActive(_V3Candidate):
    """b1_r_v3_loops_always_active: the zone heating and cooling loops run at all"""

    candidate = "loops_always_active"
    name = "b1_r_v3_loops_always_active"


class G36PolicyV3SameStepSupply(_V3Candidate):
    """    this step's recirculation and SAT commands (b1_r_v2_same_step_supply rebased on b1_r_v3)."""

    candidate = "same_step_supply"
    name = "b1_r_v3_same_step_supply"


class G36PolicyV3Ti900(_V3Candidate):
    """b1_r_v3_ti900: zone loop integral time 900 s (ControlLoops.mo:4-15) instead of
    120 s (b1_r_v2_dr06_ti_900 rebased on b1_r_v3)."""

    candidate = "ti900"
    name = "b1_r_v3_ti900"


V3_CANDIDATE_CLASSES = {cls.name: cls for cls in (
    G36PolicyV3LoopsAlwaysActive, G36PolicyV3SameStepSupply, G36PolicyV3Ti900)}


C3_VARIANTS = {
    "dr12_ignore_0": {"ignore_override": 0},
    "dr12_ignore_2": {"ignore_override": 2},
    "dr16_closed_form": {"static_pressure_reset": "closed_form"},
    "req013_b_commanded_pressure": {"plant_off_valve_inputs": "commanded"},
    "dr25_observed_readback": {"command_readback": "observed"},
    "same_step_supply": {"inversion_supply_air": "same_step"},
    "loops_always_active": {"loop_activation": "always_active"},
}


def c3_variant_config(name):
    """G36Config of a WP3 variant on C3: b1_r_v3_ti900 with the variant's switch."""
    return replace(v3_candidate_config("ti900"), **C3_VARIANTS[name])


class _C3Variant(G36PolicyV3):
    """C3 (b1_r_v3_ti900) with one WP3 switch of C3_VARIANTS (competition constructor signature)."""

    variant = None

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec, c3_variant_config(self.variant))


class G36PolicyC3Dr12Ignore0(_C3Variant):
    variant = "dr12_ignore_0"
    name = "b1_r_c3_dr12_ignore_0"


class G36PolicyC3Dr12Ignore2(_C3Variant):
    variant = "dr12_ignore_2"
    name = "b1_r_c3_dr12_ignore_2"


class G36PolicyC3Dr16ClosedForm(_C3Variant):
    variant = "dr16_closed_form"
    name = "b1_r_c3_dr16_closed_form"


class G36PolicyC3Req013BCommandedPressure(_C3Variant):
    variant = "req013_b_commanded_pressure"
    name = "b1_r_c3_req013_b_commanded_pressure"


class G36PolicyC3Dr25ObservedReadback(_C3Variant):
    """See README.md."""
    variant = "dr25_observed_readback"
    name = "b1_r_c3_dr25_observed_readback"


class G36PolicyC3SameStepSupply(_C3Variant):
    variant = "same_step_supply"
    name = "b1_r_c3_same_step_supply"


class G36PolicyC3LoopsAlwaysActive(_C3Variant):
    variant = "loops_always_active"
    name = "b1_r_c3_loops_always_active"


C3_VARIANT_CLASSES = {cls.name: cls for cls in (
    G36PolicyC3Dr12Ignore0, G36PolicyC3Dr12Ignore2, G36PolicyC3Dr16ClosedForm, G36PolicyC3Req013BCommandedPressure,
    G36PolicyC3Dr25ObservedReadback, G36PolicyC3SameStepSupply, G36PolicyC3LoopsAlwaysActive)}


C3_CONFIGURABLE = {
    "ignore_override": (0, 2),
    "static_pressure_reset": ("closed_form",),
    "plant_off_valve_inputs": ("commanded",),
    "command_readback": ("observed",),
    "inversion_supply_air": ("same_step",),
    "loop_activation": ("always_active",),
    "setback_warmup_heating_airflow": ("occupied_row",),
    "table_zero_airflow": ("damper_floor",),
    "setback_warmup_discharge": ("supply_floor_setback",),
    "setback_warmup_deadband": ("supply", "neutral"),
    "setback_release": ("zone_hysteresis",),
    "unoccupied_conditioning": ("occupancy_hours", "next_occupied"),
    "unoccupied_lead_min": (60, 120, 180, 240, 360),
    "min_flow": (0.0, 0.05),
    "plant_ignore_requests": (1, 2),
    "fan_control": ("official_pid",),
    "ahu_zone_aggregate": ("worst", "average"),
    "t_opt_max_s": (21600.0, 28800.0),
    "slope_floor_k_per_s": (1.0 / 7200.0,),
    "occupied_margin_k": (0.25, 0.5),
    "discharge_offset_k": (14.0,),
    "heating_max_ratio": (0.3,),
    "economizer_mode": ("supply_air",),
    "sat_oat_low_k": (373.15,),
    "sat_oat_high_k": (373.15,),
    "occupied_cycling": (True,),
    "cycle_charge_k": (0.0,),
    "cycle_off_k": (0.5, 0.75),
    "cycle_window_min": (90,),
    "evening_charge_min": (120,),
    "night_flush": (True,),
    "sat_max_k": (293.15,),
    "precool_until_hour": (12,),
    "sat_cost_optimal": (True,),
    "sp_max_pa": (600.0, 200.0),
    "zone_gain_per_k": (0.05, 0.2),
    "zone_ti_s": (450.0, 1800.0),
    "freeze_on_k": (275.15, 273.15),
    "freeze_off_k": (278.15, 276.15),
    "sat_cost_guard_k": (0.5, 1.0, 0.25, 0.75),
    "sat_cost_respect_requests": (True,),
    "sat_cost_guard_ignore": (1, 2, 3),
    "zone_heat_ext_power": (0.5, 1.0, 1.5, -0.5, -1.0),
    "sat_cost_guard_large_q": (0.5, 0.75),
    "zone_heat_redistribute": (True,),
    "zone_heat_boost_max": (1.5, 2.0, 2.2),
    "zone_heat_boost_step": (0.02, 0.05, 0.2),
    "zone_min_ra_lps_m2": (0.3, 0.45, 0.6),
    "zone_min_rp_lps_m2": (0.125, 0.1875, 0.25),
    "zone_min_scope": ("core", "perimeter"),
    "zone_min_oa_divide": (True,),
    "zone_heat_wall_ref_m": (20.0, 30.0, 50.0),
    "zone_heat_wall_power": (0.5, 1.0),
    "empty_day_delay_min": (60, 120),
    "dcv_area_share": (0.7, 0.5, 0.3),
    "opt_start_break_max_s": (7200.0, 10800.0),
    "opt_start_continuous": (True,),
    "opt_start_zone_stagger": (True,),
    "arrival_hold_min": (60, 120),
    "arrival_hold_k": (0.3, 0.5),
    "ahu_heat_budget": (0.8, 0.7, 0.6),
    "ahu_heat_budget_scope": ("prestart",),
    "setback_min_zone_frac": (0.7,),
    "mode_release_margin_k": (2.0,),
    "discharge_min_winter_k": (281.15, 282.15),
    "cycle_vent_ledger": (True,),
    "midday_charge_k": (1.0,),
    "precool_k": (0.3, 2.5),
    "sp_max_setback_pa": (200.0,),
    "sp_rogue_hours": (24.0,),
    "sat_cost_model": ("economizer",),
    "sat_chiller_lockout_k": (288.15,),
    "sat_cost_model_min_oat_k": (288.15,),
    "night_flush_min_range_k": (12.0,),
    "night_flush_min_max_k": (298.15,),
    "precool_min_range_k": (10.0,),
    "precool_min_max_k": (303.15,),
    "opt_start_break_min_oat_k": (297.15,),
    "freeze_min_airflow": (True,),
    "ahu_vot_check": (True,),
    "cycle_no_heating": (True,),
    "sp_options_max_oat_k": (288.15,),
    "sp_rogue_share": (0.9,),
    "zone_temp_lead_steady": (True,),
    "zone_temp_lead_max_k": (0.3,),
    "zone_temp_lead": (0.5,),
    "cold_low_flow_oat_k": (278.15, 273.15, 283.15, 288.15),
    "cold_low_flow_sp_pa": (200.0, 2000.0),
    "cold_low_flow_heat_max_scale": (0.5,),
    "hw_sawtooth": ("band", "saw", "saw_pump_off"),
    "hw_saw_refill_below_k": (340.15,),
    "hw_saw_period": (6, 12),
    "hw_off_setpoint": ("hold", "low_warm"),
    "no_reheat_oat_k": (288.15,),
    "setback_heat_drop_k": (1.0, 2.0, 3.0),
    "pump_dp_reset": ("trim_respond", "valve_inverse"),
    "warmup_target_add_k": (0.3,),
    "freeze_coldest_only": (True,),
    "freeze_heat_mode": ("reset_below_off", "reset_below_off_valve_off"),
    "warmup_deadband_neutral_above_k": (1.0, 0.5),
    "big_room_heat_max_ratio": (0.7, 0.6, 0.8, 1.0),
    "big_room_need": (True,),
    "cold_low_flow_deadband_only": (True,),
    "cold_low_flow_margin_k": (0.5, 1.0, 1.5),
    "economic_keep_cold_k": (0.6,),
    "empty_day_setup": ("night",),
    "big_room_min_area_m2": (150.0, 100.0, 0.0),
    "sat_cost_fan_observed": (True,),
    "margin_feedback_step_k": (0.05,),
    "sat_guard_trend_s": (1800.0, 900.0),
    "sat_guard_trend_k": (0.25, 0.5),
    "sat_rate_limit_k": (0.5,),
    "sat_guard_hysteresis_k": (0.25,),
    "night_heat_mode": ("weekday", "always", "free_gas"),
    "night_cool_mode": ("night", "tree"),
    "warmup_heat_max_ratio": (0.6, 1.0),
    "arrival_keep_flow_min": (30, 60),
    "sat_guard_rogue_hours": (24.0,),
    "sat_cost_max_min_range_k": (12.0,),
    "humid_low_sat_k": (286.15,),
    "economizer_hysteresis_k": (1.0,),
    "sp_damper_pi": (True,),
    "sp_smooth_alpha": (0.5,),
    "low_load_max_loop": (0.5,),
    "heating_ramp_from_supply": (True,),
    "dcv_area_share_auto": (True,),
    "cooldown_learn_step_k": (0.2,),
    "return_air_area_weighted": (True,),
    "hw_request_rogue": ("share", "three"),
    "opt_start_fit": ("affine",),
    "disturbance_ff_steps": (3.0,),
    "disturbance_ff_cool_k": (0.2,),
    "economic_target_k": (0.6,),
    "departure_learn": (True,),
    "arrival_learn": (True,),
    "capacity_start_extra_s": (1800.0, 3600.0, 5400.0),
    "start_offset_s": (-3600.0, 3600.0, 7200.0),
    "daily_start": (True,),
    "daily_layer": (True,),
    "daily_layer_parts": ("setback", "sp_cap", "budget"),
    "daily_min_oat24_k": (291.15,),
    "hourly_lp": (True,),
    "hourly_lp_horizon_h": (6, 12, 24),
    "hourly_lp_every_steps": (12, 24),
    "disturbance_highpass_steps": (24.0,),
    "hw_saw_resume_dp": (True,),
    "capacity_start_heat_only": (True,),
    "arrival_interval_start": (True,),
    "no_reheat_min_oat24_k": (293.15,),
    "night_heat_scale": (0.25,),
    "night_heat_max_h": (6.0, 12.0),
    "night_heat_stop_k": (0.0,),
    "night_cool_scale": (0.5,),
    "night_cool_max_h": (8.0,),
    "night_cool_floor_k": (1.0,),
    "arrival_keep_below_only": (True,),
    "sat_guard_trend_one_sided": (True,),
    "pump_dp_ignore": (0,),
    "sat_guard_keep_margin_k": (0.1,),
    "arrival_keep_ramp": (True,),
}
C3_CONFIGURABLE["unoccupied_conditioning"] = C3_CONFIGURABLE["unoccupied_conditioning"] + ("always",)
for _key, _extra in (("empty_day_delay_min", (30,)), ("precool_until_hour", (6,)), ("precool_k", (1.0,)),
                     ("zone_ti_s", (900.0, 630.0)), ("zone_gain_per_k", (0.07,)), ("sat_cost_max_k", (295.15,)),
                     ("freeze_on_k", (173.15,)), ("freeze_off_k", (176.15,))):
    C3_CONFIGURABLE[_key] = tuple(C3_CONFIGURABLE.get(_key, ())) + _extra
del _key, _extra

RULE_PARAMETER_RANGES = {
    "occupied_margin_k": (0.0, 1.0),
    "discharge_offset_k": (8.0, 14.0),
    "heating_max_ratio": (0.1, 0.5),
    "unoccupied_lead_min": (0, 360),
    "t_opt_max_s": (3600.0, 21600.0),
}
SLOW_PARAMETER_RANGES = {**RULE_PARAMETER_RANGES, "sp_max_pa": (200.0, 2000.0), "sat_cost_guard_k": (0.0, 30.0)}


def c3_configured_config(overrides):
    """C3's G36Config with whitelisted switch overrides (C3_CONFIGURABLE); raises on anything else."""
    overrides = dict(overrides or {})
    for key, value in overrides.items():
        if key not in C3_CONFIGURABLE:
            raise KeyError(f"{key!r} is not a configurable C3 switch (allowed: {sorted(C3_CONFIGURABLE)})")
        if not any(type(value) is type(allowed) and value == allowed for allowed in C3_CONFIGURABLE[key]):
            raise ValueError(f"{key} = {value!r} is not allowed (allowed: {C3_CONFIGURABLE[key]})")
    return replace(v3_candidate_config("ti900"), **overrides)


class G36PolicyC3Configured(G36PolicyV3):
    """C3 (b1_r_v3_ti900) with a dict of whitelisted switch overrides; no overrides = C3.

    For a remote job that instantiates classes as (building_meta_data, action_spec), bind the
    overrides with c3_configured_class (usable as a policy factory with policy_params)."""

    name = "b1_r_c3_configured"

    def __init__(self, building_meta_data, action_spec, overrides=None):
        self.overrides = dict(overrides or {})
        super().__init__(building_meta_data, action_spec, c3_configured_config(self.overrides))


def c3_configured_class(overrides):
    """A competition policy class: G36PolicyC3Configured bound to `overrides` (checked here)."""
    bound = dict(overrides or {})
    c3_configured_config(bound)

    class BoundC3Configured(G36PolicyC3Configured):
        def __init__(self, building_meta_data, action_spec):
            super().__init__(building_meta_data, action_spec, bound)

    return BoundC3Configured


B2R2_CHANGES = {"setback_warmup_discharge": 'supply_floor_setback', "setback_warmup_deadband": 'supply', "setback_release": 'zone_hysteresis', "economizer_mode": 'supply_air'}
C3_COMBINATIONS = {
    "laa_occrow_dampfloor": {"loop_activation": "always_active",
                             "setback_warmup_heating_airflow": "occupied_row",
                             "table_zero_airflow": "damper_floor"},
    "laa_occrow_dampfloor_cold3": {"loop_activation": "always_active",
                                   "setback_warmup_heating_airflow": "occupied_row",
                                   "table_zero_airflow": "damper_floor",
                                   "setback_warmup_discharge": "supply_floor_setback",
                                   "setback_warmup_deadband": "supply",
                                   "setback_release": "zone_hysteresis"},
    "laa_occrow_dampfloor_b2a1": {"loop_activation": "always_active", "setback_warmup_heating_airflow": "occupied_row",
                                  "table_zero_airflow": "damper_floor", "unoccupied_conditioning": "next_occupied",
                                  "unoccupied_lead_min": 120},
    "laa_occrow_dampfloor_b2a4": {"loop_activation": "always_active", "setback_warmup_heating_airflow": "occupied_row",
                                  "table_zero_airflow": "damper_floor", "fan_control": "official_pid",
                                  "ahu_zone_aggregate": "worst"},
    "laa_occrow_dampfloor_b2a14": {"loop_activation": "always_active", "setback_warmup_heating_airflow": "occupied_row",
                                   "table_zero_airflow": "damper_floor", "unoccupied_conditioning": "next_occupied",
                                   "unoccupied_lead_min": 120, "fan_control": "official_pid",
                                   "ahu_zone_aggregate": "worst"},
    "laa_occrow_dampfloor_b2a124": {"loop_activation": "always_active",
                                    "setback_warmup_heating_airflow": "occupied_row",
                                    "table_zero_airflow": "damper_floor", "unoccupied_conditioning": "next_occupied",
                                    "unoccupied_lead_min": 120, "fan_control": "official_pid",
                                    "ahu_zone_aggregate": "worst", "t_opt_max_s": 21600.0},
    "laa_occrow_dampfloor_b2r2": {"loop_activation": "always_active", "setback_warmup_heating_airflow": "occupied_row",
                                  "table_zero_airflow": "damper_floor", "unoccupied_conditioning": "next_occupied",
                                  "unoccupied_lead_min": 120, **B2R2_CHANGES},
}


class _C3Combination(G36PolicyC3Configured):
    """C3 with one fixed C3_COMBINATIONS entry (competition constructor signature)."""

    combination = None

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec, C3_COMBINATIONS[self.combination])


class G36PolicyC3LaaOccrowDampfloor(_C3Combination):
    """b1_r_c3_laa_occrow_dampfloor: C3 + loop_activation "always_active" +
    setback_warmup_heating_airflow "occupied_row" + table_zero_airflow "damper_floor"."""

    combination = "laa_occrow_dampfloor"
    name = "b1_r_c3_laa_occrow_dampfloor"


class G36PolicyC3LaaOccrowDampfloorCold3(_C3Combination):
    """b1_r_c3_laa_occrow_dampfloor_cold3: v4 + setback_warmup_discharge "supply_floor_setback"
    + setback_warmup_deadband "supply" + setback_release "zone_hysteresis"; own command readback, as v4."""

    combination = "laa_occrow_dampfloor_cold3"
    name = "b1_r_c3_laa_occrow_dampfloor_cold3"


class G36PolicyB2A1(_C3Combination):
    """b1_r_c3_laa_occrow_dampfloor_b2a1 (baseline v2 A1): v4 + no conditioning when nobody is in (next_occupied)."""

    combination = "laa_occrow_dampfloor_b2a1"
    name = "b1_r_c3_laa_occrow_dampfloor_b2a1"


class G36PolicyB2A4(_C3Combination):
    """b1_r_c3_laa_occrow_dampfloor_b2a4 (baseline v2 A4): v4 + AHU only while its group is outside the band."""

    combination = "laa_occrow_dampfloor_b2a4"
    name = "b1_r_c3_laa_occrow_dampfloor_b2a4"


class G36PolicyB2A14(_C3Combination):
    """b1_r_c3_laa_occrow_dampfloor_b2a14 (baseline v2 A1 + A4)."""

    combination = "laa_occrow_dampfloor_b2a14"
    name = "b1_r_c3_laa_occrow_dampfloor_b2a14"


class G36PolicyB2A124(_C3Combination):
    """b1_r_c3_laa_occrow_dampfloor_b2a124 (baseline v2 A1 + A2 + A4)."""

    combination = "laa_occrow_dampfloor_b2a124"
    name = "b1_r_c3_laa_occrow_dampfloor_b2a124"


class G36PolicyB2R2(_C3Combination):
    """b1_r_c3_laa_occrow_dampfloor_b2r2 (baseline v2 round 2): A1 + B2R2_CHANGES."""

    combination = "laa_occrow_dampfloor_b2r2"
    name = "b1_r_c3_laa_occrow_dampfloor_b2r2"


C3_COMBINATION_CLASSES = {cls.name: cls for cls in (G36PolicyC3LaaOccrowDampfloor, G36PolicyC3LaaOccrowDampfloorCold3,
                                                    G36PolicyB2A1, G36PolicyB2A4, G36PolicyB2A14, G36PolicyB2A124,
                                                    G36PolicyB2R2)}
