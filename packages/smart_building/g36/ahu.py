"""Air handler sequences (WP2 step 2.6: D-R15, D-R16, D-R20, D-R21, D-R22).

D-R15  Supply air temperature: T-max is reset by trim & respond within
       285.15-291.15 K from SP0 = 291.15 K, +0.1 / -0.2 / -0.6 K per 2 min
       (+0.25 / -0.5 / -1.5 K per step, D-R14), Td 10 min, driven by the cooling
       SAT requests of the group (I from D-R12); in occupied and setup modes the
       setpoint is T-max at outside air <= 289.15 K, 285.15 K at >= 294.15 K,
       linear in between (notes/04 C3; Addendum s §5.16.2.2.b, notes/04 G); in
       warm-up and setback 305 K (G36 308.15 K cut to the action maximum,
       notes/00 §C); in cool-down TSupCoo_min 285.15 K (D-R26; the "reset value"
       option is withdrawn, audit decision 20).
D-R16  Static pressure, A/B: (a) trim & respond, SP0 120 Pa, range 200 (the
       running minimum) - 2000 Pa, -12 / +15 / +32 Pa per 2 min (-30 / +37.5 /
       +80 Pa per step), Td 10 min, on the static pressure requests of the group;
       (b) closed form SP* = max(200, 2000 * max_i (q_i / Qmax)^2), the lowest
       pressure at which every box reaches its airflow setpoint with damper <= 1
       (notes/04 A6 from vav.py:279-290).
D-R20  Economizer (official pid_plus.py:54-64, 266-350): with cooling demand and
       outside air below return - 0.00322 K/Pa * SP, outside air fraction
       f = (T_return - target) / (T_return - T_outside) so the mix lands on the
       target, recirculation 1 - f; the target is the supply air setpoint but
       not below the zones' heating setpoint (target = max(target, low),
       pid_plus.py:337-339; text corrected 2026-10-16, REQ-011); otherwise the
       official cap 0.7. Since 2026-10-16 (audit decision 7) only while the AHU
       loop output u of the official baseline is below 0 (pid_plus.py:306-307),
       u computed as in pid_plus.py (OfficialAhuLoop).
D-R21  Fan on/off, A/B: (a) G36 (SupplyFan.mo:372-413, notes/04 C2): on in
       occupied, warm-up, cool-down, setback and setup; (b) official pid rule
       (pid.py:241-250, pid_plus.py:212-222): off when the group error of D-R22
       is 0, i.e. the zone temperature it looks at is inside the band; no
       hysteresis.
D-R22  Group error from the worst zone or from the group average, A/B
       (pid_plus.py:67-75, 143-158, 290, 296-300).
"""
import numpy as np

from smart_building.g36.modes import COOL_DOWN, OCCUPIED, SETBACK, SETUP, UNOCCUPIED, WARM_UP
from smart_building.g36.trim_respond import TrimRespond, delay_steps, per_step
from smart_building.g36.zone import DESIGN_STATIC_PRESSURE, RUNNING_MIN_STATIC_PRESSURE


def deadband_error(temperature, low, high):
    """Signed error of pid.py:175-180: low - T below the band, high - T above it, else 0."""
    return np.where(temperature < low, low - temperature,
                    np.where(temperature > high, high - temperature, 0.0))


def group_errors(zone_temp, group_slices, heating_setpoint, cooling_setpoint, aggregate):
    """D-R22: one signed error per zone group.

    'worst': the zone with the largest |error| (first one on ties,
    pid_plus.py:150-158); 'average': error of the group's mean temperature
    (pid_plus.py:296-300).
    """
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
    """D-R21: run command per zone group (mode from modes.operating_mode, D-R26)."""
    if method == "g36_modes":
        return mode != UNOCCUPIED
    if method == "official_pid":
        return error != 0.0
    raise ValueError(f"unknown fan control {method!r}")


class SupplyAirTemperature:
    """D-R15 per zone group."""

    def __init__(self, ignore, *, minimum=285.15, maximum=291.15, oat_low=289.15, oat_high=294.15,
                 heating_mode=305.0, cooldown=285.15, trim=0.1, respond=-0.2, respond_max=-0.6,
                 period=120.0, delay=600.0):
        self.minimum = float(minimum)
        self.oat_low = float(oat_low)
        self.oat_high = float(oat_high)
        self.heating_mode = float(heating_mode)
        self.cooldown = float(cooldown)
        self.loops = [TrimRespond(initial=maximum, minimum=minimum, maximum=maximum,
                                  trim=per_step(trim, period), respond=per_step(respond, period),
                                  respond_max=per_step(respond_max, period), ignore=i,
                                  delay=delay_steps(delay)) for i in ignore]

    def reset_value(self, t_max, oat):
        """D-R15: T-max at OAT <= 289.15 K, the minimum at OAT >= 294.15 K, linear between."""
        return np.interp(oat, [self.oat_low, self.oat_high], [t_max, self.minimum])

    def update(self, requests, fan_on, oat, *, mode, error):
        """Setpoint per group (K); mode is the D-R26 mode of each group."""
        t_max = np.array([loop.update(int(r), bool(on)) for loop, r, on in zip(self.loops, requests, fan_on)])
        reset = np.array([self.reset_value(t, o) for t, o in zip(t_max, oat)])
        sat = reset.copy()                                                 # D-R15 occupied, setup
        for g in range(len(sat)):
            if mode[g] == COOL_DOWN:
                sat[g] = self.cooldown                                     # D-R26 TSupCoo_min
            elif mode[g] in (WARM_UP, SETBACK):
                sat[g] = self.heating_mode                                 # D-R15 warm-up / setback
            elif mode[g] == UNOCCUPIED and fan_on[g] and error[g] > 0.0:
                # Only with D-R21 (b), not in the first version (reviewer, item 8 of
                # the REQ-009-011 decision): fan running in an unoccupied period
                # with a heating error is treated as setback.
                sat[g] = self.heating_mode
            elif mode[g] not in (OCCUPIED, SETUP, UNOCCUPIED):
                raise ValueError(f"unknown mode {mode[g]}")
        return sat


class StaticPressure:
    """D-R16 per zone group."""

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


class OfficialAhuLoop:
    """Audit decision 7: the AHU loop output u of the official baseline, used only as the
    D-R20 economizer gate (pid_plus.py:184-262 get_action, 296-300 mean-zone error of
    PIDEconomizerPolicy, 306-307 the gate; gains pid.py:44-55).

    Band: comfort window on weekdays within the metadata working hours, else the eco
    window. Error: deadband error of the group's mean zone temperature. Error 0: the
    integral halves, the previous error is 0 and there is no u (the economizer holds
    the cap). Otherwise integral = clip(I + e, -10, 10), derivative = e - previous
    error, u = clip(0.5 e + 0.02 I + 0.1 d, -1, 1). The hour is minute_of_day // 60
    of the D-R04 clock (the baseline floors the hod angle; the two agree away from
    float rounding at the hour).
    """

    def __init__(self, n_groups, *, kp=0.5, ki=0.02, kd=0.1, integral_limit=10.0):
        self.kp, self.ki, self.kd, self.limit = float(kp), float(ki), float(kd), float(integral_limit)
        self.integral = np.zeros(n_groups)
        self.previous = np.zeros(n_groups)

    def update(self, zone_temp, group_slices, *, work_hours, comfort_window, eco_window):
        """u per group (NaN where the error is 0); work_hours: weekday and within the hours."""
        low, high = comfort_window if work_hours else eco_window
        u = np.full(len(group_slices), np.nan)
        for g, zones in enumerate(group_slices):
            error = float(deadband_error(float(np.mean(zone_temp[zones])), low, high))
            if error == 0.0:
                self.integral[g] *= 0.5
                self.previous[g] = 0.0
                continue
            self.integral[g] = float(np.clip(self.integral[g] + error, -self.limit, self.limit))
            derivative = error - self.previous[g]
            self.previous[g] = error
            u[g] = float(np.clip(self.kp * error + self.ki * self.integral[g] + self.kd * derivative, -1.0, 1.0))
        return u


def economizer(return_temp, oat, sat_setpoint, static_pressure, heating_setpoint, *, allowed, k_per_pa=0.00322,
               max_recirculation=0.7, heating_floor=True):
    """D-R20: recirculation fraction per group; `allowed` is the official gate u < 0.

    The mixing target is the supply air setpoint, not below the group's
    heating setpoint (pid_plus.py:337-339). Cooling demand is return air above
    that target: the two hold branches of pid_plus.py:306-307 (heating) and
    340-341 (inside <= target) both reduce to return <= target here. The fan
    pressure is the commanded one with the 200 Pa running floor
    (pid_plus.py:330-331). heating_floor=False (baseline v2 A7, G36 5.16.2) mixes
    down to the supply air setpoint itself.
    """
    pressure = np.maximum(static_pressure, RUNNING_MIN_STATIC_PRESSURE)
    target = np.maximum(sat_setpoint, heating_setpoint) if heating_floor else np.asarray(sat_setpoint, dtype=float)
    cooling = return_temp > target
    worthwhile = oat < return_temp - k_per_pa * pressure
    span = np.where(return_temp - oat > 0.0, return_temp - oat, 1.0)
    outside_fraction = (return_temp - target) / span
    recirculation = np.clip(1.0 - outside_fraction, 0.0, 1.0)
    return np.where(allowed & cooling & worthwhile, recirculation, max_recirculation)
