"""Hot-water plant sequences of b1_r_v2.

       -1.1 / +1.7 / +3.9 K, Td 10 min, I = 2, SP0 = 350 K, range 310-350 K
, on the
       hot-water reset requests of all zones. The commanded setpoint rises by at
       most 2.5 K per step: the tank heats at 0.5 K/min (config.gin:146) and
       gas is billed on the setpoint.
       (baselines/pid.py:281-286, pid_plus.py:252-257): maximum while any AHU
       Templates/Plants/Controls/Setpoints/PlantReset.mo:4-87, 235-315): one
       reset variable R by trim & respond (R0 = 1, range 0-1, -0.02 / +0.03 /
       +0.07, T 5 min, Td 15 min, I = 2); R 0-0.5 moves the pressure from its
       minimum to its maximum, R 0.5-1 the supply temperature from its lower
       limit to its design value.
       enable when the plant has been off for at least 3 steps (15 min), there
       are more than I = 0 plant requests and the outside air is below HW-LOT;
       disable when it has been on for at least 3 steps and either this step has
       at most I requests or the outside air is above HW-LOT + 1 F. HW-LOT is the
       the same informative note says to set HW-LOT above the maximum outside
       temperature when the plant must run at any outside air, which this
       simulator needs, lockout=None); schedule 24/7; no warm-up start. The timer
       starts at the episode start with the plant off, as in MBL (no enable in
       the first 3 steps). The 3-minute persistence of the request condition is
       judged on the single step."""
from smart_building.g36.trim_respond import TrimRespond, delay_steps, per_step

HW_LOCKOUT_K = (75.0 - 32.0) * 5.0 / 9.0 + 273.15
HW_LOCKOUT_HYSTERESIS_K = 5.0 / 9.0


class SupplyWaterTemperature:
    """See README.md."""

    def __init__(self, *, initial=350.0, minimum=310.0, maximum=350.0, trim=-1.1, respond=1.7,
                 respond_max=3.9, period=300.0, delay=600.0, ignore=2, max_rise=2.5):
        self.loop = TrimRespond(initial=initial, minimum=minimum, maximum=maximum,
                                trim=per_step(trim, period), respond=per_step(respond, period),
                                respond_max=per_step(respond_max, period), ignore=ignore,
                                delay=delay_steps(delay))
        self.max_rise = float(max_rise)

    def target(self, requests, plant_on):
        return self.loop.update(int(requests), bool(plant_on))

    def limit_rise(self, target, previous_setpoint):
        """See README.md."""
        return min(float(target), float(previous_setpoint) + self.max_rise)


class PlantReset:
    """    310-350 K. The 900 s hold of PlantReset.mo has no defined trigger here and"""

    def __init__(self, *, dp_min=0.0, dp_max=20.0, temp_low=310.0, temp_high=350.0, initial=1.0,
                 trim=-0.02, respond=0.03, respond_max=0.07, period=300.0, delay=900.0, ignore=2):
        self.dp_min, self.dp_max = float(dp_min), float(dp_max)
        self.temp_low, self.temp_high = float(temp_low), float(temp_high)
        self.loop = TrimRespond(initial=initial, minimum=0.0, maximum=1.0,
                                trim=per_step(trim, period), respond=per_step(respond, period),
                                respond_max=per_step(respond_max, period), ignore=ignore,
                                delay=delay_steps(delay))

    def _pressure(self, r):
        return self.dp_min + min(r, 0.5) / 0.5 * (self.dp_max - self.dp_min)

    def pressure_in_force(self):
        """Differential pressure of the reset value in force (before this step's update)."""
        return self._pressure(self.loop.setpoint)

    def update(self, requests, plant_on):
        """(differential pressure bar, supply temperature target K)."""
        r = self.loop.update(int(requests), bool(plant_on))
        temp = self.temp_low + max(r - 0.5, 0.0) / 0.5 * (self.temp_high - self.temp_low)
        return self._pressure(r), temp


def official_pump_pressure(any_fan_on, *, dp_min=0.0, dp_max=20.0):
    """See README.md."""
    return dp_max if any_fan_on else dp_min


class PlantEnable:
    """    outside air lockout; lockout None = no
    outside air lockout."""

    def __init__(self, *, min_off_steps=3, min_on_steps=3, ignore=0, lockout=HW_LOCKOUT_K,
                 lockout_hysteresis=HW_LOCKOUT_HYSTERESIS_K):
        self.min_off_steps = int(min_off_steps)
        self.min_on_steps = int(min_on_steps)
        self.ignore = int(ignore)
        self.lockout = None if lockout is None else float(lockout)
        self.lockout_hysteresis = float(lockout_hysteresis)
        self.enabled = False
        self.steps_in_state = 0

    def update(self, plant_requests, oat):
        """Plant command for this step from this step's plant requests and outside air (K)."""
        locked = self.lockout is not None
        if self.enabled:
            if self.steps_in_state >= self.min_on_steps and (
                    plant_requests <= self.ignore or (locked and oat > self.lockout + self.lockout_hysteresis)):
                self.enabled, self.steps_in_state = False, 0
        elif (self.steps_in_state >= self.min_off_steps and plant_requests > self.ignore
              and (not locked or oat < self.lockout)):
            self.enabled, self.steps_in_state = True, 0
        self.steps_in_state += 1
        return self.enabled
