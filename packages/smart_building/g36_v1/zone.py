"""Zone control: PI loops, VAV dual maximum, damper conversion and the discharge
air temperature estimate (WP2 step 2.4: D-R06, D-R07, D-R08, D-R09).

D-R06  Zone heating and cooling loops are PI with k = 0.1 per K; the integral
       time Ti is an A/B switch: 900 s (G36/ThermalZones/ControlLoops.mo:4-15,
       notes/04 C7) or 120 s (G36/TerminalUnits/Reheat/Controller.mo:51-66,
       notes/04 C6). Loop outputs are 0-1 (0-100 %).
D-R07  Dual maximum (DamperValves.mo:862-919, notes/04 C6): cooling airflow
       rises with the cooling loop from minimum to cooling maximum; deadband
       holds the minimum; heating loop 0-50 % raises the discharge air setpoint
       linearly from the supply air temperature to min(heating setpoint + 11 K,
       308.15 K) and a valve PI (k 0.5 per K, Ti 300 s) tracks it (since D-R28
       the valve is set by inverting the D-R09 equations; the PI is the WP3
       ablation); heating loop
       50-100 % raises airflow from heating minimum to heating maximum while the
       discharge air is more than 3 K above the zone.
D-R08  Airflow limits: minimum = the simulator damper floor 0.1, cooling
       maximum = damper 1.0, heating maximum = 50 % of cooling maximum. Read as
       fractions of the design flow Qmax at the design pressure 2000 Pa
       (vav.py:279-286, q = damper * Qmax * sqrt(SP / 2000)): 0.1, 1.0, 0.5.
       Airflow setpoints are absolute so that the static pressure requests
       (D-R10) and the closed-form reset (D-R16 b) can ask for more pressure.
D-R09  Damper command = required airflow / (Qmax * sqrt(SP / 2000)); with the
       airflow as a fraction of Qmax this is fraction / sqrt(SP / 2000). The
       box has no discharge air sensor, so the discharge temperature is
       estimated with the simulator's reheat equations (vav.py:253-373,
       air_handler.py:257-289, hot_water_system.py:294-326) from the observed
       actuator positions; it plays the role of the G36 discharge sensor.
"""
import numpy as np

from smart_building.g36_v1.trim_respond import STEP_SECONDS

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
SECONDS_PER_HOUR = 3600.0            # Kv in m3/h, hot_water_system.py:58, 307-318


class PILoop:
    """Discrete PI loop over a vector of zones, output limited to 0-1.

    D-R06/D-R07 give the gain and integral time. D-R27 (REQ-010 (1)): the
    integral term advances by gain * error * step / Ti each step; when
    gain * error + integral leaves [0, 1] the integral is set to the bound minus
    gain * error, the step-size limit of the CDL back-calculation anti-windup
    (MBL CDL/Reals/PID.mo:32-34, 373-390). The integral itself is not clamped.
    """

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
    """D-R07, D-R08: airflow setpoint (fraction of Qmax) and discharge air setpoint (K).

    heating, cooling: zone loop outputs 0-1; supply_temp: the AHU supply air
    temperature reaching the box; discharge_temp: the D-R09 estimate.
    """
    cooling_flow = min_flow + cooling * (cooling_max_flow - min_flow)
    first_half = np.clip(heating / 0.5, 0.0, 1.0)
    discharge_max = np.minimum(heating_setpoint + discharge_offset, discharge_limit)
    discharge_setpoint = supply_temp + first_half * (discharge_max - supply_temp)
    second_half = np.clip((heating - 0.5) / 0.5, 0.0, 1.0)
    heating_flow = np.where(discharge_temp > zone_temp + zone_margin,
                            min_flow + second_half * (heating_max_flow - min_flow), min_flow)
    # D-R27: both loops can be above 0 for a step while an integral unwinds;
    # the larger branch airflow is kept. With either loop at 0 this is exactly
    # the D-R07 branch of the other loop.
    flow = np.maximum(cooling_flow, heating_flow)
    return flow, discharge_setpoint


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
