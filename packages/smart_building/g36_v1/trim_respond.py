"""Trim & respond setpoint reset.

       if the request count R is at most the ignore count I the setpoint is only
       trimmed; if R > I it is only responded to, by (R - I) * respond, at most
       respond_max. Trimming and responding in the same update (the MBL wiring,
       While the device is off the output is the initial setpoint, and updates
       TrimAndRespond.mo:393-439).
; the ablation overrides it with 0, 1 or 2.
       they are multiplied by 300 / 120 = 2.5; Td of 10 minutes is 2 steps; the
       hot-water side has a 5-minute period and is used as given."""

STEP_SECONDS = 300


def per_step(amount, period_seconds, step_seconds=STEP_SECONDS):
    """    Derivation: same rate of change, amount * step / period; for the air side
    period 120 s this is amount * 2.5."""
    return amount * step_seconds / period_seconds


def delay_steps(delay_seconds, step_seconds=STEP_SECONDS):
    """See README.md."""
    steps = delay_seconds / step_seconds
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError(f"delay {delay_seconds} s is not a whole number of {step_seconds} s steps")
    return int(round(steps))


def ignore_count(downstream_zones):
    """See README.md."""
    return (int(downstream_zones) + 5) // 10


class TrimRespond:
    """One trim & respond loop, updated once per simulator step.

    Signs follow the quantity: a static pressure loop trims down (trim < 0) and
    responds up (respond > 0); a supply air temperature loop trims up and
    responds down. respond_max has the sign of respond and bounds its size."""

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
        self.steps_on = None

    def update(self, requests, device_on=True):
        """Advance one step and return the setpoint to command for it."""
        if not device_on:
            self.reset()
            return self.setpoint
        self.steps_on = 0 if self.steps_on is None else self.steps_on + 1
        if self.steps_on < self.delay:
            self.setpoint = self.initial
            return self.setpoint
        if requests <= self.ignore:
            change = self.trim
        else:
            change = (requests - self.ignore) * self.respond
            if abs(change) > abs(self.respond_max):
                change = self.respond_max
        self.setpoint = min(self.maximum, max(self.minimum, self.setpoint + change))
        return self.setpoint
