"""Trim & respond setpoint reset (WP2 step 2.2: D-R11, D-R12, D-R14).

D-R11  Addendum b to Guideline 36-2018 (notes/04 F) semantics: at each update,
       if the request count R is at most the ignore count I the setpoint is only
       trimmed; if R > I it is only responded to, by (R - I) * respond, at most
       respond_max. Trimming and responding in the same update (the MBL wiring,
       notes/04 C1) is not used. The setpoint is held in [minimum, maximum].
       While the device is off the output is the initial setpoint, and updates
       start once the device has run for the delay Td (notes/04 C1,
       TrimAndRespond.mo:393-439).
D-R12  Default ignore count I = round(0.1 * downstream zones), rounded half up
       (Addendum s, notes/04 G: "四捨五入"); the ablation overrides it with 0, 1 or 2.
D-R14  Guideline 36 air-side amounts are per 2-minute period: per 5-minute step
       they are multiplied by 300 / 120 = 2.5; Td of 10 minutes is 2 steps; the
       hot-water side has a 5-minute period and is used as given.
"""

STEP_SECONDS = 300  # time_step_sec, config.gin:250


def per_step(amount, period_seconds, step_seconds=STEP_SECONDS):
    """D-R14: an amount per G36 period expressed per simulator step.

    Derivation: same rate of change, amount * step / period; for the air side
    period 120 s this is amount * 2.5 (notes/04 §1 "時間換算").
    """
    return amount * step_seconds / period_seconds


def delay_steps(delay_seconds, step_seconds=STEP_SECONDS):
    """D-R14: Td in whole steps (600 s -> 2, 900 s -> 3)."""
    steps = delay_seconds / step_seconds
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError(f"delay {delay_seconds} s is not a whole number of {step_seconds} s steps")
    return int(round(steps))


def ignore_count(downstream_zones):
    """D-R12: round(0.1 * n) rounded half up, in exact integer arithmetic."""
    return (int(downstream_zones) + 5) // 10


class TrimRespond:
    """One trim & respond loop, updated once per simulator step.

    Signs follow the quantity: a static pressure loop trims down (trim < 0) and
    responds up (respond > 0); a supply air temperature loop trims up and
    responds down. respond_max has the sign of respond and bounds its size.
    """

    def __init__(self, initial, minimum, maximum, trim, respond, respond_max, ignore, delay):
        if minimum > maximum:
            raise ValueError("minimum above maximum")
        if respond * respond_max < 0:
            raise ValueError("respond and respond_max must have the same sign")
        self.initial = float(initial)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.trim = float(trim)
        self.respond = float(respond)
        self.respond_max = float(respond_max)
        self.ignore = int(ignore)
        self.delay = int(delay)
        self.reset()

    def reset(self):
        self.setpoint = self.initial
        self.steps_on = None  # None while the device is off

    def update(self, requests, device_on=True):
        """Advance one step and return the setpoint to command for it."""
        if not device_on:
            self.reset()
            return self.setpoint
        self.steps_on = 0 if self.steps_on is None else self.steps_on + 1
        if self.steps_on < self.delay:
            self.setpoint = self.initial
            return self.setpoint
        # D-R11: trim or respond, never both.
        if requests <= self.ignore:
            change = self.trim
        else:
            change = (requests - self.ignore) * self.respond
            if abs(change) > abs(self.respond_max):
                change = self.respond_max
        self.setpoint = min(self.maximum, max(self.minimum, self.setpoint + change))
        return self.setpoint
