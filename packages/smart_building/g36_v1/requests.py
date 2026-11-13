"""Zone requests computed by the controller (WP2 step 2.3: D-R10, D-R13).

The simulator's own request-count fields are constants (notes/04 A2, notes/00
§H1), so every request is computed here from zone temperatures, setpoints and
the controller's own commands (notes/04 §0 point 1).

D-R10  Thresholds of MBL SystemRequests (notes/04 C6; G36/TerminalUnits/Reheat/
       Subsequences/SystemRequests.mo, parameters 7-69, description 591-672):
       cooling SAT 3 requests above the cooling setpoint by > 3 K, 2 by > 2 K,
       else 1 while the cooling loop is > 95 % until it is < 85 %; static
       pressure 3 when the damper is > 95 % open and the airflow < 50 % of its
       setpoint, 2 below 70 %, else 1 while the damper is > 95 % until < 85 %;
       hot-water reset 3 when the discharge air is > 17 K below its setpoint,
       2 when > 8 K, else 1 while the valve is > 95 % until < 85 %; heating
       plant 1 while the valve is > 95 % until it is < 10 %. The persistence
       times of the source (1, 2 and 5 minutes) are at most one 5-minute step,
       so each condition is judged on the current step.
D-R13  After the active setpoint changes by dT, requests are suppressed for
       min(540 s/K * |dT|, 1800 s) (notes/04 C9 TimeSuppression.mo:5-17, G);
       on/off is an A/B switch.
"""
import numpy as np


def update_latch(latch, value, on_above, off_below):
    """D-R10: a request held from value > on_above until value < off_below."""
    return (value > on_above) | (latch & (value >= off_below))


def cooling_sat_requests(excess, loop, latch, suppressed, *, three_above=3.0, two_above=2.0,
                         loop_on=0.95, loop_off=0.85):
    """Cooling supply-air-temperature requests per zone (D-R10).

    excess: zone temperature minus active cooling setpoint (K); loop: cooling
    loop output 0-1; suppressed: D-R13 mask for the temperature conditions.
    Returns (requests, new latch).
    """
    latch = update_latch(latch, loop, loop_on, loop_off)
    requests = np.where(latch, 1, 0)
    requests = np.where((excess > two_above) & ~suppressed, 2, requests)
    requests = np.where((excess > three_above) & ~suppressed, 3, requests)
    return requests, latch


def static_pressure_requests(demanded_damper, latch, *, damper_on=0.95, damper_off=0.85,
                             three_below=0.5, two_below=0.7):
    """Static pressure requests per zone (D-R10).

    demanded_damper is the damper position the airflow setpoint needs at the
    present static pressure (D-R09); above 1 the damper stays fully open and the
    airflow falls to 1 / demanded of its setpoint (notes/04 §1 "Static pressure
    requests": 需求 damper > 1 即有缺口).
    """
    damper = np.minimum(demanded_damper, 1.0)
    flow_ratio = np.where(demanded_damper > 1.0, 1.0 / np.maximum(demanded_damper, 1.0), 1.0)
    latch = update_latch(latch, damper, damper_on, damper_off)
    wide_open = damper > damper_on
    requests = np.where(latch, 1, 0)
    requests = np.where(wide_open & (flow_ratio < two_below), 2, requests)
    requests = np.where(wide_open & (flow_ratio < three_below), 3, requests)
    return requests, latch


def hot_water_reset_requests(deficit, valve, latch, *, three_above=17.0, two_above=8.0,
                             valve_on=0.95, valve_off=0.85):
    """Hot-water reset requests per zone (D-R10).

    deficit: discharge air temperature setpoint minus the estimated discharge
    air temperature (D-R09), K.
    """
    latch = update_latch(latch, valve, valve_on, valve_off)
    requests = np.where(latch, 1, 0)
    requests = np.where(deficit > two_above, 2, requests)
    requests = np.where(deficit > three_above, 3, requests)
    return requests, latch


def heating_plant_requests(valve, latch, *, valve_on=0.95, valve_off=0.10):
    """Heating plant requests per zone (D-R10): 1 while the valve is > 95 % until < 10 %."""
    latch = update_latch(latch, valve, valve_on, valve_off)
    return latch.astype(int), latch


class SetpointChangeSuppression:
    """D-R13: suppression timer per zone group after an active setpoint change."""

    def __init__(self, n_groups, *, enabled=True, seconds_per_kelvin=540.0, max_seconds=1800.0,
                 step_seconds=300):
        self.enabled = bool(enabled)
        self.seconds_per_kelvin = float(seconds_per_kelvin)
        self.max_seconds = float(max_seconds)
        self.step_seconds = float(step_seconds)
        self.previous = None
        self.remaining = np.zeros(n_groups)

    def update(self, setpoints):
        """Feed this step's active setpoints; True where requests are suppressed now."""
        setpoints = np.asarray(setpoints, dtype=float)
        if self.previous is not None:
            change = np.abs(setpoints - self.previous)
            self.remaining = np.where(change > 0.0,
                                      np.minimum(self.seconds_per_kelvin * change, self.max_seconds),
                                      self.remaining)
        self.previous = setpoints.copy()
        suppressed = self.remaining > 0.0
        self.remaining = np.maximum(self.remaining - self.step_seconds, 0.0)
        if not self.enabled:
            return np.zeros(setpoints.shape, dtype=bool)
        return suppressed
