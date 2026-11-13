"""Zone control of b1_r_v2: PI loops, loop enable/disable, VAV dual maximum, damper
conversion and the discharge air estimate (WP2 step 2.4; corrections of 2026-10-16).

The first version (b1_r, as submitted) is frozen in smart_building.g36_v1.

D-R06  Zone heating and cooling loops are PI with k = 0.1 per K; Ti 120 s
       (Reheat/Controller.mo:51-66, D-R24) or 900 s (ControlLoops.mo:4-15, WP3
       variant). Outputs 0-1.
D-R27  PI discretisation (item 2 of the audit decisions): the CDL PID integrator,
       dI/dt = (k/Ti) [e - (y_u - y) / (k Ni)] with y_u = k e + I, y = clip(y_u)
       and Ni = 0.9 (CDL Reals/PID.mo:32-34, 62-64, 129, 177-181, 217), solved
       exactly over one 300 s step with the error held (zero-order hold). Inside
       the limits dI/dt = k e / Ti (I += k e 300 / Ti, as before); above the
       upper (below the lower) limit dI/dt = (I* - I) / (Ni Ti) with
       I* = limit - (1 - Ni) k e, so I <- I* + (I - I*) exp(-t / (Ni Ti)). The
       step is split where y_u crosses a limit; with e constant the trajectory
       crosses at most twice (upper -> inside -> lower or the reverse), see
       PILoop.update. The step convention is unchanged: the integral is advanced
       with this step's error, then the output is formed from it.
D-R31  Loop enable / disable (G36 section 5.3.4, MBL ThermalZones/ControlLoops.mo,
       read in full at 11e34d2): the heating loop is enabled by CDL Less(TZon,
       THeaSet, h = 0.25 K) and its PID (PIDWithReset) is reset on the rising edge
       of that signal; its output is multiplied by 0 while the loop is disabled,
       i.e. while LessThreshold(y, t = 0.01, h = 0.008) of the PID output holds
       for timChe = 30 s (one step, D-R10 precedent) and the zone is not below the
       setpoint (lines 71-122, 124-179). The PID keeps integrating while its
       output is masked. The cooling loop is the mirror image (Less(TCooSet,
       TZon)). The "larger branch" rule is gone. Reset (REQ-016 point 1, as the
       source): the rising edge resets the controller OUTPUT to y_reset = 0, i.e.
       the integral to y_reset - k e (PIDWithReset.mo:46-48, 143-148, 186); the
       step then integrates as usual.
       Zone state (REQ-016 point 6, DamperValves.mo:269-272, 301-304, 423-431,
       643-657): cooling = GreaterThreshold(uCoo, 0.01, h 0.005), heating =
       GreaterThreshold(uHea, 0.01, h 0.008), each through TrueFalseHold(600 s
       true, 0 s false); cooling has priority, deadband when neither holds.
       TrueFalseHold at 300 s steps: a rising edge of the threshold output at
       step k keeps the state true at steps k and k + 1 (600 s); from k + 2 the
       state follows the threshold output again (ZoneStates).
       WP3 candidate b1_r_v3_loops_always_active (reviewer, D-R31 row,
       2026-10-17): ZoneLoops(always_active=True) runs both loops at all times
       with no enable / disable and no reset, the D-R27 anti-windup being the
       provision G36 section 5.3.4 asks for; the zone states are unchanged.
D-R07  Dual maximum (DamperValves.mo:862-919, 438-575): cooling airflow from the
       minimum to the cooling maximum; deadband minimum; heating 0-50 % ramps the
       discharge setpoint from the AHU supply air setpoint TSupSet (line 531,
       item 6) to min(heating setpoint + 11 K, 308.15 K); 50-100 % raises the
       airflow from heating minimum to heating maximum while the gate
       Greater(TDis, TZon + 3 K, h = 0.25 K) holds (lines 57, 236-238, 511-515;
       REQ-015 F3). TDis is the estimate of the pair sent last step.
D-R30  Airflow limits by mode (ActiveAirFlow.mo:301-324), 0 raised to the damper
       floor; the zone state (D-R31, ZoneStates) picks the branch: cooling state
       first, then deadband (minimum), else heating. REQ-019 (2026-10-18) adds two
       switches, defaults unchanged (mode_airflow_limits): (A) warm-up and setback
       heating limits from the occupied row; (B) a 0 entry stays 0 and the simulator
       applies its 0.1 damper position floor.
D-R34  In the cooling state, if the AHU supply air (estimate) exceeds the zone by
       GreaterThreshold(t = 0.25 K, h = 0.125 K), the cooling airflow is the
       minimum (DamperValves.mo:274-281, 463-471, 454-460; documentation 878-880).
D-R35  Valve enable (DamperValves.mo:260-261, 292-294, 543-563): the coil runs in
       the heating state, or in occupied mode, in any other state, while
       LessThreshold(TDis, t = TDisMin = 283.15 K, h = 0.25 K) holds. Outside the
       heating state the discharge setpoint is TDisMin = 283.15 K (REQ-016 point
       2: the G36 text quoted in the MBL documentation wins over the wiring's
       max(TSupSet, TDisMin), which assumes an AHU that heats to TSupSet).
       Otherwise the valve is 0.
D-R33  Unoccupied mode: valve 0 and damper 0 (damPosUno, watValPosUno; lines
       282-285, 344-349, 551-575); the simulator raises the damper to its floor.
D-R08  Airflow fractions of the design flow Qmax at 2000 Pa (vav.py:279-286):
       minimum 0.1, cooling maximum 1.0, heating maximum 0.5.
D-R09  Damper command = fraction / sqrt(SP / 2000); discharge air estimated with
       the simulator's reheat equations from the actuator state read back.
D-R28  Valve: the D-R09 equations solved for the valve at this step's damper
       command and static pressure as the simulator applies them (REQ-015 F1).
"""
import math

import numpy as np

from smart_building.g36.modes import COOL_DOWN, OCCUPIED, SETBACK, SETUP, UNOCCUPIED, WARM_UP
from smart_building.g36.trim_respond import STEP_SECONDS

# Simulator constants (official facts; the scenario generator fixes the VAV
# ones, scenario_generator.py:278-340, notes/04 A6).
AIR_DENSITY = 1.2                    # kg/m3, smart_control/simulator/constants.py:103
AIR_HEAT_CAPACITY = 1006.0           # J/kg/K, constants.py:104
WATER_DENSITY = 1000.0               # kg/m3, constants.py:106
WATER_HEAT_CAPACITY = 4180.0         # J/kg/K, constants.py:107
VAV_MAX_FLOW = 2.0                   # m3/s, config.gin:152 vav_max_air_flowrate
DESIGN_STATIC_PRESSURE = 2000.0      # Pa, vav.py:58 max_air_flow_static_pressure
REHEAT_KV = 0.108                    # config.gin:153 vav_reheat_water_flowrate
MAX_DISCHARGE_TEMPERATURE = 308.15   # K, vav.py:79
MIN_FLOW_DEMAND = 1e-5               # m3/s, vav.py:288-290
RUNNING_MIN_STATIC_PRESSURE = 200.0  # Pa, air_handler.py:53-59, 227-231; config.gin:163
AHU_MAX_FLOW = 20.0                  # m3/s, config.gin:176 (2.0 m3/s x 10 boxes, scenario_generator.py:179-181)
DAMPER_FLOOR = 0.1                   # vav.py:65 min_damper_setting; applied as max(command, floor), vav.py:490-495
SECONDS_PER_HOUR = 3600.0            # Kv in m3/h, hot_water_system.py:58, 307-318


# -- CDL comparison blocks with hysteresis (Reals/Greater.mo:88, Less.mo:85,
#    GreaterThreshold, LessThreshold.mo:85); `previous` is pre(y), initially false.

def greater(u1, u2, previous, h):
    """CDL Greater: (not pre(y) and u1 > u2) or (pre(y) and u1 > u2 - h)."""
    return np.where(previous, u1 > u2 - h, u1 > u2)


def less(u1, u2, previous, h):
    """CDL Less: (not pre(y) and u1 < u2) or (pre(y) and u1 < u2 + h)."""
    return np.where(previous, u1 < u2 + h, u1 < u2)


def discharge_gate(discharge_temp, threshold, previous, hysteresis):
    """D-R07 stage-2 gate: Greater(TDis, TZon + 3 K, h = dTHys)."""
    return greater(discharge_temp, threshold, previous, hysteresis)


class PILoop:
    """D-R27: discrete PI over a vector, output limited to [minimum, maximum], with the
    exact zero-order-hold solution of the CDL anti-windup over each step."""

    def __init__(self, n, gain, ti_seconds, step_seconds=STEP_SECONDS, *, minimum=0.0, maximum=1.0, ni=0.9):
        self.gain = float(gain)
        self.ti = float(ti_seconds)
        self.step = float(step_seconds)
        self.tau = float(ni) * self.ti
        self.ni = float(ni)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.integral = np.zeros(n)

    def reset(self, mask, value=0.0):
        """Set the integral where mask is true (PIDWithReset trigger)."""
        self.integral = np.where(mask, value, self.integral)

    def advance(self, integral, error, duration):
        """Integral after `duration` seconds with the error held (exact, piecewise).

        a = k e. Inside the limits I(t) = I + a t / Ti until a + I reaches a limit.
        Saturated at limit L (a + I beyond L, or at L and moving outward):
        I(t) = I* + (I - I*) exp(-t / tau), I* = L - (1 - Ni) a, tau = Ni Ti. It
        leaves the saturated region only if I* lies inside (a < 0 at the upper
        limit, a > 0 at the lower one), at t = tau ln((I - I*) / (I_b - I*)) with
        I_b = L - a. The regime is tracked explicitly across a crossing; with e held
        there are at most three segments (upper, inside, lower or the reverse).
        """
        a = self.gain * np.asarray(error, dtype=float)
        integral = np.array(integral, dtype=float, copy=True)
        left = np.full(integral.shape, float(duration))
        y = a + integral
        regime = np.where((y > self.maximum) | ((y >= self.maximum) & (a > 0.0)), 1,
                          np.where((y < self.minimum) | ((y <= self.minimum) & (a < 0.0)), -1, 0))
        for _ in range(4):
            if not (left > 0.0).any():
                break
            for side, limit, leaves in ((1, self.maximum, a < 0.0), (-1, self.minimum, a > 0.0)):
                m = (left > 0.0) & (regime == side)
                if not m.any():
                    continue
                target = limit - (1.0 - self.ni) * a
                boundary = limit - a
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = (integral - target) / (boundary - target)
                    t_hit = np.where(leaves, self.tau * np.log(np.maximum(np.nan_to_num(ratio, nan=1.0), 1.0)),
                                     np.inf)
                stay = m & (t_hit >= left)
                cross = m & (t_hit < left)
                integral = np.where(stay, target + (integral - target) * np.exp(-left / self.tau), integral)
                integral = np.where(cross, boundary, integral)
                left = np.where(stay, 0.0, np.where(cross, left - t_hit, left))
                regime = np.where(cross, 0, regime)
            m = (left > 0.0) & (regime == 0)
            if m.any():
                rate = a / self.ti
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_up = np.where(rate > 0.0, (self.maximum - a - integral) / rate, np.inf)
                    t_down = np.where(rate < 0.0, (self.minimum - a - integral) / rate, np.inf)
                t_hit = np.maximum(np.minimum(t_up, t_down), 0.0)
                stay = m & (t_hit >= left)
                up = m & (t_hit < left) & (rate > 0.0)
                down = m & (t_hit < left) & (rate < 0.0)
                integral = np.where(stay, integral + rate * left, integral)
                integral = np.where(up, self.maximum - a, np.where(down, self.minimum - a, integral))
                left = np.where(stay, 0.0, np.where(up | down, left - t_hit, left))
                regime = np.where(up, 1, np.where(down, -1, regime))
        return integral

    def update(self, error):
        """Advance one step with this step's error; returns the limited output."""
        self.integral = self.advance(self.integral, error, self.step)
        return np.clip(self.gain * np.asarray(error, dtype=float) + self.integral, self.minimum, self.maximum)


class ZoneLoops:
    """D-R06 + D-R31: heating and cooling loops of every zone with the ControlLoops.mo
    enable / disable logic. update() returns (heating, cooling) outputs 0-1.

    always_active (WP3 candidate b1_r_v3_loops_always_active, D-R31 row 2026-10-17): no
    enable / disable and no reset; both PI loops run every step from the start and their
    outputs are used as they are. G36 section 5.3.4 allows this when provisions are made to
    minimize integral windup (ControlLoops.mo documentation); the provision is the D-R27
    CDL anti-windup of PILoop. The default False is b1_r_v2 / b1_r_v3."""

    def __init__(self, n, gain, ti_seconds, *, hysteresis=0.25, near_zero=0.01, near_zero_hysteresis=0.008,
                 reset_output=0.0, ni=0.9, step_seconds=STEP_SECONDS, always_active=False):
        self.heating = PILoop(n, gain, ti_seconds, step_seconds, ni=ni)
        self.cooling = PILoop(n, gain, ti_seconds, step_seconds, ni=ni)
        self.always_active = bool(always_active)
        self.hysteresis = float(hysteresis)
        self.near_zero = float(near_zero)
        self.near_zero_hysteresis = float(near_zero_hysteresis)
        self.reset_output = float(reset_output)            # PIDWithReset y_reset
        self.heat_enabled = np.zeros(n, dtype=bool)        # enaHeaLoo pre(y)
        self.cool_enabled = np.zeros(n, dtype=bool)        # enaCooLoo pre(y)
        self.heat_near_zero = np.zeros(n, dtype=bool)      # zerCon1 pre(y)
        self.cool_near_zero = np.zeros(n, dtype=bool)      # zerCon pre(y)
        self.enable_edges = 0                              # diagnostics: rising edges (resets)
        self.masked_zone_steps = 0                         # diagnostics: outputs forced to 0

    def update(self, zone_temp, heating_setpoint, cooling_setpoint):
        if self.always_active:
            # No enable, no mask, no reset: the D-R27 limited outputs as they are.
            return (self.heating.update(heating_setpoint - zone_temp),
                    self.cooling.update(zone_temp - cooling_setpoint))
        heat_en = less(zone_temp, heating_setpoint, self.heat_enabled, self.hysteresis)
        cool_en = less(cooling_setpoint, zone_temp, self.cool_enabled, self.hysteresis)
        heat_rise, cool_rise = heat_en & ~self.heat_enabled, cool_en & ~self.cool_enabled
        heat_error, cool_error = heating_setpoint - zone_temp, zone_temp - cooling_setpoint
        # PIDWithReset: on the rising edge the output is reset to y_reset, I = y_reset - k e.
        self.heating.reset(heat_rise, self.reset_output - self.heating.gain * heat_error)
        self.cooling.reset(cool_rise, self.reset_output - self.cooling.gain * cool_error)
        y_heat = self.heating.update(heat_error)
        y_cool = self.cooling.update(cool_error)
        limit = self.near_zero + self.near_zero_hysteresis
        heat_nz = np.where(self.heat_near_zero, y_heat < limit, y_heat < self.near_zero)
        cool_nz = np.where(self.cool_near_zero, y_cool < limit, y_cool < self.near_zero)
        heat_off, cool_off = heat_nz & ~heat_en, cool_nz & ~cool_en
        heating = np.where(heat_off, 0.0, y_heat)
        cooling = np.where(cool_off, 0.0, y_cool)
        self.heat_enabled, self.cool_enabled = heat_en, cool_en
        self.heat_near_zero, self.cool_near_zero = heat_nz, cool_nz
        self.enable_edges += int(heat_rise.sum() + cool_rise.sum())
        self.masked_zone_steps += int(np.sum(heat_off & (y_heat > 0.0)) + np.sum(cool_off & (y_cool > 0.0)))
        return heating, cooling


class ZoneStates:
    """D-R31 (REQ-016 point 6): the zone state of DamperValves.mo (lines 269-272, 301-304,
    423-431, 643-657): cooling = GreaterThreshold(uCoo, t 0.01, h 0.005) and heating =
    GreaterThreshold(uHea, t 0.01, h 0.008), each held by TrueFalseHold(trueHoldDuration
    600 s, falseHoldDuration 0). Discretisation of the hold at 300 s steps: a rising edge
    of the threshold output at step k holds the state true at steps k and k + 1
    (hold_steps = 2 = 600 s / 300 s); from step k + 2 it follows the threshold output.
    update() returns (cooling state, heating state); the caller gives cooling priority.
    """

    def __init__(self, n, *, threshold=0.01, cooling_hysteresis=0.005, heating_hysteresis=0.008, hold_steps=2):
        self.threshold = float(threshold)
        self.cooling_hysteresis = float(cooling_hysteresis)
        self.heating_hysteresis = float(heating_hysteresis)
        self.hold_steps = int(hold_steps)
        self.cool_raw = np.zeros(n, dtype=bool)        # greThr1 pre(y)
        self.heat_raw = np.zeros(n, dtype=bool)        # greThr2 pre(y)
        self.cool_hold = np.zeros(n, dtype=int)        # steps the cooling state is still held
        self.heat_hold = np.zeros(n, dtype=int)
        self.state = np.zeros(n, dtype=int)            # 0 deadband, 1 cooling, 2 heating (cooling first)
        self.changes = 0                               # diagnostics: zone-state changes

    def update(self, heating_output, cooling_output):
        cool_raw = greater(cooling_output, self.threshold, self.cool_raw, self.cooling_hysteresis)
        heat_raw = greater(heating_output, self.threshold, self.heat_raw, self.heating_hysteresis)
        self.cool_hold = np.where(cool_raw & ~self.cool_raw, self.hold_steps, self.cool_hold)
        self.heat_hold = np.where(heat_raw & ~self.heat_raw, self.hold_steps, self.heat_hold)
        cooling = cool_raw | (self.cool_hold > 0)
        heating = heat_raw | (self.heat_hold > 0)
        self.cool_hold = np.maximum(self.cool_hold - 1, 0)
        self.heat_hold = np.maximum(self.heat_hold - 1, 0)
        self.cool_raw, self.heat_raw = cool_raw, heat_raw
        state = np.where(cooling, 1, np.where(heating, 2, 0))
        self.changes += int(np.sum(state != self.state))
        self.state = state
        return cooling, heating


SETBACK_WARMUP_HEATING_AIRFLOW = ("g36_table", "occupied_row")   # D-R30 switch (A), REQ-019
TABLE_ZERO_AIRFLOW = ("design_fraction", "damper_floor")          # D-R30 switch (B), REQ-019


def mode_airflow_limits(mode, *, cooling_max, heating_max, occupied_min, heating_min, floor=DAMPER_FLOOR,
                        setback_warmup_heating="g36_table", zero_entries="design_fraction"):
    """D-R30: (cooling maximum, minimum, heating minimum, heating maximum) of each zone
    group by operating mode, fractions of Qmax (MBL ActiveAirFlow.mo:301-324, G36
    section 5.6.4). VCooMax = cooling_max, VHeaMax = heating_max, VOccMin =
    occupied_min, VHeaMin = heating_min; the cooling minimum is the minimum.

    The two REQ-019 switches (reviewer 2026-10-18 03:20) default to the behaviour frozen
    in C3 and every earlier policy:
    setback_warmup_heating  "g36_table": warm-up and setback heat from VHeaMax to VCooMax
                            (the table). "occupied_row" (A, derived from the source): they
                            use the occupied row, heating minimum max(VHeaMin, VOccMin) and
                            heating maximum max(VHeaMax, VOccMin) - the table's rows for
                            35 C AHU supply air (SupplyTemperature.mo:35-39, 393-395) do not
                            fit an AHU that only cools; the freeze-protection setback of
                            D-R32 runs as SETBACK and is included. The cooling maximum, the
                            cooling minimum and the minimum of these rows are unchanged.
    zero_entries            "design_fraction": every entry is raised to 0.1 of the design
                            flow (the floor), so a 0 asks for 0.1 of Qmax at any static
                            pressure. "damper_floor" (B, the D-R30 text): an entry that is 0
                            stays 0, its damper command is 0 and the simulator holds the box
                            at its 0.1 position floor (vav.py:490-495); non-zero entries are
                            raised to the floor exactly as in "design_fraction".
    """
    if setback_warmup_heating not in SETBACK_WARMUP_HEATING_AIRFLOW:
        raise ValueError(f"unknown setback / warm-up heating airflow {setback_warmup_heating!r}")
    if zero_entries not in TABLE_ZERO_AIRFLOW:
        raise ValueError(f"unknown table zero airflow {zero_entries!r}")
    table = np.zeros((6, 4))
    table[OCCUPIED] = (cooling_max, occupied_min, max(heating_min, occupied_min), max(heating_max, occupied_min))
    table[COOL_DOWN] = (cooling_max, 0.0, heating_min, heating_max)
    table[SETUP] = (cooling_max, 0.0, 0.0, 0.0)
    table[WARM_UP] = (0.0, 0.0, heating_max, cooling_max)
    table[SETBACK] = (0.0, 0.0, heating_max, cooling_max)
    table[UNOCCUPIED] = (0.0, 0.0, 0.0, 0.0)
    if setback_warmup_heating == "occupied_row":
        for row in (WARM_UP, SETBACK):
            table[row, 2:] = table[OCCUPIED, 2:]
    rows = table[np.asarray(mode, dtype=int)]
    limits = np.maximum(rows, floor)
    if zero_entries == "damper_floor":
        limits = np.where(rows == 0.0, 0.0, limits)
    return limits[..., 0], limits[..., 1], limits[..., 2], limits[..., 3]


def dual_maximum(heating, cooling, sat_setpoint, heating_setpoint, *, cooling_state, heating_state, gate,
                 supply_hot, min_flow, cooling_max_flow, heating_min_flow, heating_max_flow, discharge_offset=11.0,
                 discharge_limit=MAX_DISCHARGE_TEMPERATURE):
    """D-R07, D-R30, D-R34: airflow setpoint (fraction of Qmax) and the heating discharge
    air setpoint (K) of each zone.

    heating, cooling: loop outputs 0-1; cooling_state, heating_state: ZoneStates (the
    cooling state has priority, deadband when neither holds, DamperValves.mo swi,
    swi1, swi2); sat_setpoint: the AHU supply air setpoint TSupSet reaching the box;
    gate: D-R07 Greater output; supply_hot: D-R34 GreaterThreshold output. The
    discharge setpoint returned is the heating ramp (used in the heating state).
    """
    cooling_flow = np.where(supply_hot, min_flow, min_flow + cooling * (cooling_max_flow - min_flow))
    first_half = np.clip(heating / 0.5, 0.0, 1.0)
    discharge_max = np.minimum(heating_setpoint + discharge_offset, discharge_limit)
    discharge_setpoint = sat_setpoint + first_half * (discharge_max - sat_setpoint)
    second_half = np.clip((heating - 0.5) / 0.5, 0.0, 1.0)
    heating_flow = np.where(gate, heating_min_flow + second_half * (heating_max_flow - heating_min_flow),
                            heating_min_flow)
    flow = np.where(cooling_state, cooling_flow, np.where(heating_state, heating_flow, min_flow))
    return flow, discharge_setpoint


def applied_damper(damper_command):
    """Damper position the box takes for a command: clipped to [0, 1], then raised to
    the floor (vav.py:217-225, 490-495)."""
    return np.maximum(np.clip(damper_command, 0.0, 1.0), DAMPER_FLOOR)


def applied_static_pressure(static_pressure_command, fan_on):
    """Static pressure the boxes see for this step's command: max(command, 200 Pa)
    while the AHU runs, 0 while it is off (air_handler.py:227-231, 242-243)."""
    return np.where(fan_on, effective_static_pressure(static_pressure_command), 0.0)


def effective_static_pressure(static_pressure):
    """Pressure a running fan holds: max(command, 200 Pa) (air_handler.py:227-231)."""
    return np.maximum(static_pressure, RUNNING_MIN_STATIC_PRESSURE)


def damper_command(flow_fraction, static_pressure):
    """D-R09: damper = q / (Qmax * sqrt(SP / 2000)) with q = fraction * Qmax.

    static_pressure is the commanded value; the running floor of 200 Pa is
    applied, so the command also stays finite while the fan is off.
    """
    return flow_fraction / np.sqrt(effective_static_pressure(static_pressure) / DESIGN_STATIC_PRESSURE)


def estimate_supply_air(return_temp, oat, recirculation, sat_setpoint, ahu_on):
    """D-R09: AHU supply air temperature (air_handler.py:257-289).

    Mixed air r * T_return + (1 - r) * T_outside; a running AHU cools it down to
    its setpoint and never heats.
    """
    mixed = recirculation * return_temp + (1.0 - recirculation) * oat
    return np.where(ahu_on & (mixed > sat_setpoint), sat_setpoint, mixed)


def box_flow_demand(damper, static_pressure):
    """Unrestricted VAV flow, m3/s (vav.py:279-290)."""
    pressure = np.maximum(static_pressure, 0.0)
    return np.maximum(damper * VAV_MAX_FLOW * np.sqrt(pressure / DESIGN_STATIC_PRESSURE), MIN_FLOW_DEMAND)


def supply_flow_ratio(demand, zone_group, n_groups):
    """Share of the demanded flow each AHU can move: min(1, AHU maximum / demand)
    (simulator_flexible_floor_plan.py:368-380). With at most 10 boxes of 2 m3/s
    per AHU and a 20 m3/s fan the ratio is 1 in generated buildings; it is kept
    for fidelity to the simulator equations.
    """
    total = np.bincount(zone_group, weights=demand, minlength=n_groups)
    return np.minimum(1.0, AHU_MAX_FLOW / np.maximum(total, MIN_FLOW_DEMAND))


def reheat_valve_for(discharge_setpoint, damper, static_pressure, flow_ratio, supply_temp, water_temp,
                     water_dp):
    """D-R28: valve position at which the D-R09 reheat equations give the setpoint.

    Inverse of estimate_discharge for the same inputs (vav.py:316-373): the
    effectiveness needed is e* = (T_set - T_supply) / (T_water - T_supply) and
    e = C_water / C_air below saturation, so valve = e* * C_air / C_water(valve 1).
    No heat needed (T_set <= T_supply) gives 0; a setpoint beyond the reachable
    effectiveness min(1, headroom / gap), or no water flow, gives 1. The result
    is clipped to [0, 1].
    """
    air_rate = AIR_HEAT_CAPACITY * AIR_DENSITY * box_flow_demand(damper, static_pressure) * flow_ratio
    water_rate_open = (WATER_HEAT_CAPACITY * WATER_DENSITY * REHEAT_KV * np.sqrt(max(water_dp, 0.0))
                       / SECONDS_PER_HOUR)
    gap = water_temp - supply_temp
    need = discharge_setpoint - supply_temp
    safe_gap = np.where(gap > 0.0, gap, 1.0)
    wanted = need / safe_gap
    reachable = np.minimum(1.0, (MAX_DISCHARGE_TEMPERATURE - supply_temp) / safe_gap)
    if water_rate_open > 0.0:
        valve = wanted * air_rate / water_rate_open
    else:
        valve = np.full(np.shape(wanted), np.inf)
    valve = np.where(wanted > reachable, 1.0, valve)
    valve = np.where((need <= 0.0) | (gap <= 0.0), 0.0, valve)
    return np.clip(valve, 0.0, 1.0)


def estimate_discharge(damper, valve, static_pressure, flow_ratio, supply_temp, water_temp, water_dp):
    """D-R09: discharge air temperature of each box, K (vav.py:316-373, 397-413).

    All inputs are per zone except water_temp (tank temperature, K) and
    water_dp (pump differential pressure, bar). Each box's water flow is
    valve * Kv * sqrt(dp) / 3600 m3/s (header resistance 0,
    hot_water_system.py:294-326; vav.py:308-314).
    """
    air_rate = AIR_HEAT_CAPACITY * AIR_DENSITY * box_flow_demand(damper, static_pressure) * flow_ratio
    water_flow = valve * REHEAT_KV * np.sqrt(max(water_dp, 0.0)) / SECONDS_PER_HOUR
    water_rate = WATER_HEAT_CAPACITY * WATER_DENSITY * water_flow
    effectiveness = np.minimum(1.0, water_rate / air_rate)
    gap = water_temp - supply_temp
    headroom = MAX_DISCHARGE_TEMPERATURE - supply_temp
    limited = np.where(headroom <= 0.0, 0.0,
                       np.minimum(effectiveness, headroom / np.where(gap > 0.0, gap, 1.0)))
    effectiveness = np.where(gap > 0.0, limited, effectiveness)
    return supply_temp + effectiveness * gap
