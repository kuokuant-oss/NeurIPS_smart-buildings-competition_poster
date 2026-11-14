"""Zone control: PI loops, VAV dual maximum, damper conversion and the discharge
air temperature estimate.

       time Ti is an A/B switch: 900 s (G36/ThermalZones/ControlLoops.mo:4-15,
       rises with the cooling loop from minimum to cooling maximum; deadband
       holds the minimum; heating loop 0-50 % raises the discharge air setpoint
       linearly from the supply air temperature to min(heating setpoint + 11 K,
       ablation); heating loop
       50-100 % raises airflow from heating minimum to heating maximum while the
       discharge air is more than 3 K above the zone.
       maximum = damper 1.0, heating maximum = 50 % of cooling maximum. Read as
       fractions of the design flow Qmax at the design pressure 2000 Pa
       (vav.py:279-286, q = damper * Qmax * sqrt(SP / 2000)): 0.1, 1.0, 0.5.
       Airflow setpoints are absolute so that the static pressure requests
 and the closed-form reset can ask for more pressure.
       airflow as a fraction of Qmax this is fraction / sqrt(SP / 2000). The
       box has no discharge air sensor, so the discharge temperature is
       estimated with the simulator's reheat equations (vav.py:253-373,
       air_handler.py:257-289, hot_water_system.py:294-326) from the observed
       actuator positions; it plays the role of the G36 discharge sensor."""
import numpy as np

from smart_building.g36_v1.trim_respond import STEP_SECONDS

AIR_DENSITY = 1.2
AIR_HEAT_CAPACITY = 1006.0
WATER_DENSITY = 1000.0
WATER_HEAT_CAPACITY = 4180.0
VAV_MAX_FLOW = 2.0
DESIGN_STATIC_PRESSURE = 2000.0
REHEAT_KV = 0.108
MAX_DISCHARGE_TEMPERATURE = 308.15
MIN_FLOW_DEMAND = 1e-5
RUNNING_MIN_STATIC_PRESSURE = 200.0
AHU_MAX_FLOW = 20.0
SECONDS_PER_HOUR = 3600.0


class PILoop:
    """Discrete PI loop over a vector of zones, output limited to 0-1.

    integral term advances by gain * error * step / Ti each step; when
    gain * error + integral leaves [0, 1] the integral is set to the bound minus
    gain * error, the step-size limit of the CDL back-calculation anti-windup
    (MBL CDL/Reals/PID.mo:32-34, 373-390). The integral itself is not clamped."""

    def __init__(self, n, gain, ti_seconds, step_seconds=STEP_SECONDS, *, minimum=0.0, maximum=1.0):
        self.gain = float(gain)
        self.integral_rate = float(gain) * float(step_seconds) / float(ti_seconds)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.integral = np.zeros(n)

    def update(self, error):
        proportional = self.gain * error
        integral = self.integral + self.integral_rate * error
        output = proportional + integral
        integral = np.where(output > self.maximum, self.maximum - proportional,
                            np.where(output < self.minimum, self.minimum - proportional, integral))
        self.integral = integral
        return np.clip(proportional + integral, self.minimum, self.maximum)


def dual_maximum(heating, cooling, supply_temp, heating_setpoint, zone_temp, discharge_temp, *,
                 min_flow, cooling_max_flow, heating_max_flow, discharge_offset=11.0,
                 discharge_limit=MAX_DISCHARGE_TEMPERATURE, zone_margin=3.0):
    """    heating, cooling: zone loop outputs 0-1; supply_temp: the AHU supply air"""
    cooling_flow = min_flow + cooling * (cooling_max_flow - min_flow)
    first_half = np.clip(heating / 0.5, 0.0, 1.0)
    discharge_max = np.minimum(heating_setpoint + discharge_offset, discharge_limit)
    discharge_setpoint = supply_temp + first_half * (discharge_max - supply_temp)
    second_half = np.clip((heating - 0.5) / 0.5, 0.0, 1.0)
    heating_flow = np.where(discharge_temp > zone_temp + zone_margin,
                            min_flow + second_half * (heating_max_flow - min_flow), min_flow)
    flow = np.maximum(cooling_flow, heating_flow)
    return flow, discharge_setpoint


def effective_static_pressure(static_pressure):
    """Pressure a running fan holds: max(command, 200 Pa) (air_handler.py:227-231)."""
    return np.maximum(static_pressure, RUNNING_MIN_STATIC_PRESSURE)


def damper_command(flow_fraction, static_pressure):
    """    static_pressure is the commanded value; the running floor of 200 Pa is
    applied, so the command also stays finite while the fan is off."""
    return flow_fraction / np.sqrt(effective_static_pressure(static_pressure) / DESIGN_STATIC_PRESSURE)


def estimate_supply_air(return_temp, oat, recirculation, sat_setpoint, ahu_on):
    """    Mixed air r * T_return + (1 - r) * T_outside; a running AHU cools it down to
    its setpoint and never heats."""
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
    for fidelity to the simulator equations."""
    total = np.bincount(zone_group, weights=demand, minlength=n_groups)
    return np.minimum(1.0, AHU_MAX_FLOW / np.maximum(total, MIN_FLOW_DEMAND))


def reheat_valve_for(discharge_setpoint, damper, static_pressure, flow_ratio, supply_temp, water_temp,
                     water_dp):
    """    Inverse of estimate_discharge for the same inputs (vav.py:316-373): the
    effectiveness needed is e* = (T_set - T_supply) / (T_water - T_supply) and
    e = C_water / C_air below saturation, so valve = e* * C_air / C_water(valve 1).
    No heat needed (T_set <= T_supply) gives 0; a setpoint beyond the reachable
    effectiveness min(1, headroom / gap), or no water flow, gives 1. The result
    is clipped to [0, 1]."""
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
    """    All inputs are per zone except water_temp (tank temperature, K) and
    water_dp (pump differential pressure, bar). Each box's water flow is
    valve * Kv * sqrt(dp) / 3600 m3/s (header resistance 0,
    hot_water_system.py:294-326; vav.py:308-314)."""
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
