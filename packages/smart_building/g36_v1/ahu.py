"""Air handler sequences.

       285.15-291.15 K from SP0 = 291.15 K, +0.1 / -0.2 / -0.6 K per 2 min
, Td 10 min, driven by the cooling
       SAT requests of the group; in occupied and setup modes the
       setpoint is T-max at outside air <= 289.15 K, 285.15 K at >= 294.15 K,
       linear in between; in
       warm-up and setback 305 K (G36 308.15 K cut to the action maximum,
       running minimum) - 2000 Pa, -12 / +15 / +32 Pa per 2 min (-30 / +37.5 /
       +80 Pa per step), Td 10 min, on the static pressure requests of the group;
       (b) closed form SP* = max(200, 2000 * max_i (q_i / Qmax)^2), the lowest
       pressure at which every box reaches its airflow setpoint with damper <= 1
.
       outside air below return - 0.00322 K/Pa * SP, outside air fraction
       f = (T_return - target) / (T_return - T_outside) so the mix lands on the
       target, recirculation 1 - f; the target is the supply air setpoint but
       not below the zones' heating setpoint (target = max(target, low),
       official cap 0.7.
       occupied, warm-up, cool-down, setback and setup; (b) official pid rule
       is 0, i.e. the zone temperature it looks at is inside the band; no
       hysteresis.
       (pid_plus.py:67-75, 143-158, 290, 296-300)."""
import numpy as np

from smart_building.g36_v1.modes import COOL_DOWN, OCCUPIED, SETBACK, SETUP, UNOCCUPIED, WARM_UP
from smart_building.g36_v1.trim_respond import TrimRespond, delay_steps, per_step
from smart_building.g36_v1.zone import DESIGN_STATIC_PRESSURE, RUNNING_MIN_STATIC_PRESSURE


def deadband_error(temperature, low, high):
    """Signed error of pid.py:175-180: low - T below the band, high - T above it, else 0."""
    return np.where(temperature < low, low - temperature,
                    np.where(temperature > high, high - temperature, 0.0))


def group_errors(zone_temp, group_slices, heating_setpoint, cooling_setpoint, aggregate):
    """    'worst': the zone with the largest |error| (first one on ties,
    pid_plus.py:150-158); 'average': error of the group's mean temperature
    (pid_plus.py:296-300)."""
    errors = np.zeros(len(group_slices))
    for g, zones in enumerate(group_slices):
        temps = zone_temp[zones]
        if aggregate == "worst":
            zone_errors = deadband_error(temps, heating_setpoint[g], cooling_setpoint[g])
            errors[g] = zone_errors[int(np.argmax(np.abs(zone_errors)))]
        elif aggregate == "average":
            errors[g] = float(deadband_error(temps.mean(), heating_setpoint[g], cooling_setpoint[g]))
        else:
            raise ValueError(f"unknown aggregate {aggregate!r}")
    return errors


def fan_command(method, *, mode, error):
    """See README.md."""
    if method == "g36_modes":
        return mode != UNOCCUPIED
    if method == "official_pid":
        return error != 0.0
    raise ValueError(f"unknown fan control {method!r}")


class SupplyAirTemperature:
    """See README.md."""

    def __init__(self, ignore, *, minimum=285.15, maximum=291.15, oat_low=289.15, oat_high=294.15,
                 heating_mode=305.0, cooldown=285.15, trim=0.1, respond=-0.2, respond_max=-0.6,
                 period=120.0, delay=600.0):
        self.minimum = float(minimum)
        self.oat_low = float(oat_low)
        self.oat_high = float(oat_high)
        self.heating_mode = float(heating_mode)
        self.cooldown = cooldown
        self.loops = [TrimRespond(initial=maximum, minimum=minimum, maximum=maximum,
                                  trim=per_step(trim, period), respond=per_step(respond, period),
                                  respond_max=per_step(respond_max, period), ignore=i,
                                  delay=delay_steps(delay)) for i in ignore]

    def reset_value(self, t_max, oat):
        """See README.md."""
        return np.interp(oat, [self.oat_low, self.oat_high], [t_max, self.minimum])

    def update(self, requests, fan_on, oat, *, mode, error):
        """See README.md."""
        t_max = np.array([loop.update(int(r), bool(on)) for loop, r, on in zip(self.loops, requests, fan_on)])
        reset = np.array([self.reset_value(t, o) for t, o in zip(t_max, oat)])
        sat = reset.copy()
        for g in range(len(sat)):
            if mode[g] == COOL_DOWN:
                sat[g] = reset[g] if self.cooldown is None else float(self.cooldown)
            elif mode[g] in (WARM_UP, SETBACK):
                sat[g] = self.heating_mode
            elif mode[g] == UNOCCUPIED and fan_on[g] and error[g] > 0.0:
                sat[g] = self.heating_mode
            elif mode[g] not in (OCCUPIED, SETUP, UNOCCUPIED):
                raise ValueError(f"unknown mode {mode[g]}")
        return sat


class StaticPressure:
    """See README.md."""

    def __init__(self, method, ignore, *, initial=120.0, minimum=RUNNING_MIN_STATIC_PRESSURE,
                 maximum=DESIGN_STATIC_PRESSURE, trim=-12.0, respond=15.0, respond_max=32.0,
                 period=120.0, delay=600.0):
        if method not in ("trim_respond", "closed_form"):
            raise ValueError(f"unknown static pressure reset {method!r}")
        self.method = method
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.loops = [TrimRespond(initial=initial, minimum=minimum, maximum=maximum,
                                  trim=per_step(trim, period), respond=per_step(respond, period),
                                  respond_max=per_step(respond_max, period), ignore=i,
                                  delay=delay_steps(delay)) for i in ignore]
        self.setpoint = np.array([loop.setpoint for loop in self.loops])

    def update(self, requests, fan_on, flow_fraction, group_slices):
        """Pressure per group (Pa); flow_fraction is each box's airflow setpoint / Qmax."""
        if self.method == "trim_respond":
            self.setpoint = np.array([loop.update(int(r), bool(on))
                                      for loop, r, on in zip(self.loops, requests, fan_on)])
        else:
            peak = np.array([flow_fraction[zones].max() for zones in group_slices])
            self.setpoint = np.clip(self.maximum * peak ** 2, self.minimum, self.maximum)
        return self.setpoint


def economizer(return_temp, oat, sat_setpoint, static_pressure, heating_setpoint, *, k_per_pa=0.00322,
               max_recirculation=0.7):
    """    The mixing target is the supply air setpoint, not below the group's
    heating setpoint (pid_plus.py:337-339). Cooling demand is return air above
    that target: the two hold branches of pid_plus.py:306-307 (heating) and
    340-341 (inside <= target) both reduce to return <= target here. The fan
    pressure is the commanded one with the 200 Pa running floor
    (pid_plus.py:330-331)."""
    pressure = np.maximum(static_pressure, RUNNING_MIN_STATIC_PRESSURE)
    target = np.maximum(sat_setpoint, heating_setpoint)
    cooling = return_temp > target
    worthwhile = oat < return_temp - k_per_pa * pressure
    span = np.where(return_temp - oat > 0.0, return_temp - oat, 1.0)
    outside_fraction = (return_temp - target) / span
    recirculation = np.clip(1.0 - outside_fraction, 0.0, 1.0)
    return np.where(cooling & worthwhile, recirculation, max_recirculation)
