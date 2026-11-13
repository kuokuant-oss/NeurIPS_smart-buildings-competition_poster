"""Hot-water plant sequences (WP2 step 2.7: D-R17, D-R18, D-R19).

D-R17  Supply water temperature: trim & respond every 5 minutes (one step),
       -1.1 / +1.7 / +3.9 K, Td 10 min, I = 2, SP0 = 350 K, range 310-350 K
       (Addendum s table 5.21.4.1, notes/04 G; action range notes/00 §C), on the
       hot-water reset requests of all zones. The commanded setpoint rises by at
       most 2.5 K per step: the tank heats at 0.5 K/min (config.gin:146) and
       gas is billed on the setpoint (boiler.py:213-225, notes/04 A5).
D-R18  Pump differential pressure. Main case = official pid / pid_water
       (baselines/pid.py:281-286, pid_plus.py:252-257): maximum while any AHU
       runs, else minimum. Ablation = MBL PlantReset (notes/04 E,
       Templates/Plants/Controls/Setpoints/PlantReset.mo:4-87, 235-315): one
       reset variable R by trim & respond (R0 = 1, range 0-1, -0.02 / +0.03 /
       +0.07, T 5 min, Td 15 min, I = 2); R 0-0.5 moves the pressure from its
       minimum to its maximum, R 0.5-1 the supply temperature from its lower
       limit to its design value.
D-R19  The plant runs when there is at least one heating plant request, and
       always during warm-up (notes/04 C6, §2 table).
"""
from smart_building.g36_v1.trim_respond import TrimRespond, delay_steps, per_step


class SupplyWaterTemperature:
    """D-R17."""

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
        """D-R17: at most max_rise above the setpoint in force (observed, raw K)."""
        return min(float(target), float(previous_setpoint) + self.max_rise)


class PlantReset:
    """D-R18 ablation: shared reset variable R for pressure and temperature.

    The ranges are not in notes/04 E; until REQ-011 (3) is answered they are the
    official action ranges: 0-20 bar (config.gin:328-331) and the D-R17 range
    310-350 K. The 900 s hold of PlantReset.mo has no defined trigger here and
    is not implemented (REQ-011 (3)).
    """

    def __init__(self, *, dp_min=0.0, dp_max=20.0, temp_low=310.0, temp_high=350.0, initial=1.0,
                 trim=-0.02, respond=0.03, respond_max=0.07, period=300.0, delay=900.0, ignore=2):
        self.dp_min, self.dp_max = float(dp_min), float(dp_max)
        self.temp_low, self.temp_high = float(temp_low), float(temp_high)
        self.loop = TrimRespond(initial=initial, minimum=0.0, maximum=1.0,
                                trim=per_step(trim, period), respond=per_step(respond, period),
                                respond_max=per_step(respond_max, period), ignore=ignore,
                                delay=delay_steps(delay))

    def update(self, requests, plant_on):
        """(differential pressure bar, supply temperature target K)."""
        r = self.loop.update(int(requests), bool(plant_on))
        dp = self.dp_min + min(r, 0.5) / 0.5 * (self.dp_max - self.dp_min)
        temp = self.temp_low + max(r - 0.5, 0.0) / 0.5 * (self.temp_high - self.temp_low)
        return dp, temp


def official_pump_pressure(any_fan_on, *, dp_min=0.0, dp_max=20.0):
    """D-R18 main case: maximum while any AHU runs, else minimum (pid.py:283-284)."""
    return dp_max if any_fan_on else dp_min


def plant_enable(plant_requests, any_warm_up):
    """D-R19."""
    return bool(plant_requests >= 1 or any_warm_up)
