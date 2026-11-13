"""Optimal start: warm-up and cool-down timing (WP2 step 2.5: D-R05).

D-R05  MBL Buildings/Controls/OBC/Utilities/OptimalStart.mo with
       BaseClasses/OptimalStartCalculation.mo (notes/04 D): tOptMax 10800 s,
       nDay 3, uLow 0, uHigh 0.5 K, thrOptOn 60 s, slope floor 1/3600 K/s,
       moving average refreshed every midnight. Algorithm (OptimalStart.mo:
       189-290 as summarised in notes/04 D):
         1. slope = |T_set - T_sam1| / dt, dt = time from system start until
            the zone reaches its setpoint, at most tOptMax;
         2. the slope used is the average of the last nDay daily values;
         3. at tOptMax before occupancy the zone temperature T_sam2 is sampled
            and tOpt = |T_set - T_sam2| / average slope; no early start when
            the deviation is at most uHigh.
       Every AHU takes its slowest zone; the next occupancy comes from the
       hod/dow clock and morning_start_hour (modes.seconds_to_next_occupied).

Implementation reading of the steps above (REQ-010 (5), approved 2026-10-16):
  * T_set is the occupied (comfort) setpoint: heating for warm-up, cooling for
    cool-down; deviations are T_set_heat - T and T - T_set_cool.
  * A zone is "reached" when its deviation is at most uLow; it learns a new
    slope only when it was not reached at the group's warm-up (cool-down)
    start. dt is counted in 300 s steps, so tReaMin = 120 s never binds.
  * Moving average (audit decision 5, OptimalStartCalculation.mo:59-70, 160-163,
    244-268; CDL Discrete/TriggeredMovingMean.mo:36-53): at each midnight, only
    in zones whose deviation sampled at the last calculation start (dTCalOn,
    tOptMax before occupancy) was above 0 (reqStaUp, t = 0), the latest learned
    slope (temSlo; 0 until a first slope is learned, the samplers start at 0)
    enters a ring of nDay samples; the mean is over the samples received so far,
    counting the one taken at initialisation (`when {initial(), trigger}`), and
    the slope used is max(mean, 1/3600 K/s): 1/3600 K/s is a floor
    (temSloDef), not a starting value. dTCalOn keeps its value until the next
    calculation start, so after a Friday that needed a start-up the weekend
    midnights also push (the source does this; noted in REQ-016).
  * tOpt is capped at tOptMax; the mode is on while 0 < time to occupancy <=
    tOpt and tOpt > thrOptOn, and ends when occupancy starts. The last tOpt is
    kept until the next sample replaces it (it cannot switch a mode on earlier,
    because the time to occupancy exceeds tOptMax until that sample).
"""
import numpy as np

from smart_building.g36.modes import is_occupied, seconds_to_next_occupied
from smart_building.g36.trim_respond import STEP_SECONDS


class _Direction:
    """Optimal start for one direction (heating -> warm-up, or cooling -> cool-down)."""

    def __init__(self, n_zones, n_groups, *, n_day, slope_floor):
        self.n_day = int(n_day)
        self.slope_floor = float(slope_floor)
        self.held = np.zeros(n_zones)                      # temSlo (samplers start at 0)
        self.samples = np.zeros((n_zones, self.n_day))     # TriggeredMovingMean ySample
        self.samples[:, 0] = self.held                     # sample taken at initial()
        self.count = np.ones(n_zones, dtype=int)
        self.next_index = np.full(n_zones, 1 % self.n_day, dtype=int)
        self.sampled_deviation = np.zeros(n_zones)         # dTCalOn
        self.average = np.maximum(self.samples.sum(axis=1) / self.count, self.slope_floor)
        self.updates = 0                                   # diagnostics: zone-midnights pushed
        self.zone_t_opt = np.zeros(n_zones)
        self.group_t_opt = np.zeros(n_groups)
        self.active = np.zeros(n_groups, dtype=bool)
        self.tracking = np.zeros(n_zones, dtype=bool)
        self.start_deviation = np.zeros(n_zones)
        self.elapsed = np.zeros(n_zones)
        # pipeline 5 O504 (fit "affine"): the last n_day (start deviation, time to reach) pairs of each zone
        self.pairs = np.full((n_zones, self.n_day, 2), np.nan)
        self.pair_index = np.zeros(n_zones, dtype=int)
        self.group_sampled = np.zeros(n_groups, dtype=bool)  # option audit O506: per-direction samples

    def remember(self, done, deviation, dt):
        """O504: keep the finished warm-ups (cool-downs) of the zones flagged in done."""
        for z in np.flatnonzero(done):
            self.pairs[z, self.pair_index[z]] = (deviation[z], dt[z])
            self.pair_index[z] = (self.pair_index[z] + 1) % self.n_day

    def midnight(self):
        trigger = self.sampled_deviation > 0.0             # reqStaUp (GreaterThreshold t = 0, h = 0)
        if trigger.any():
            zones = np.flatnonzero(trigger)
            self.samples[zones, self.next_index[zones]] = self.held[zones]
            self.next_index[zones] = (self.next_index[zones] + 1) % self.n_day
            self.count[zones] = np.minimum(self.count[zones] + 1, self.n_day)
            self.updates += int(zones.size)
        self.average = np.maximum(self.samples.sum(axis=1) / self.count, self.slope_floor)


class OptimalStart:
    def __init__(self, zone_group, group_slices, *, heating_setpoint, cooling_setpoint,
                 morning_hour, evening_hour, t_opt_max=10800.0, n_day=3, u_low=0.0, u_high=0.5,
                 slope_floor=1.0 / 3600.0, thr_opt_on=60.0, step_seconds=STEP_SECONDS, continuous=False,
                 fit="slope"):
        self.zone_group = np.asarray(zone_group)
        self.group_slices = list(group_slices)
        n_zones, n_groups = len(self.zone_group), len(self.group_slices)
        self.heating_setpoint = float(heating_setpoint)
        self.cooling_setpoint = float(cooling_setpoint)
        self.morning_hour = int(morning_hour)
        self.evening_hour = int(evening_hour)
        self.t_opt_max = float(t_opt_max)
        self.u_low = float(u_low)
        self.u_high = float(u_high)
        self.thr_opt_on = float(thr_opt_on)
        self.step_seconds = float(step_seconds)
        # Option catalogue O1 (2026-11-03): with continuous, the zone deviations are sampled again at every step
        # within tOptMax before occupancy for the groups whose mode has not started yet (latest-feasible start from
        # the present state instead of the single sample at tOptMax); False = MBL's single sample (D-R05).
        self.continuous = bool(continuous)
        # Pipeline 5 O504 (2026-11-04): fit "affine" = each zone's start time from a least-squares line through its
        # last n_day (start deviation, time to reach) pairs, tOpt = a + b x deviation (b >= 0), once it has two pairs
        # with different deviations; else G36's deviation / average slope. "slope" = G36 only (default).
        if fit not in ("slope", "affine"):
            raise ValueError(f"unknown optimal start fit {fit!r}")
        self.fit = fit
        # Pipeline 5 O506 (2026-11-04): group_extra (s per group, None = off, set by the controller) starts the groups
        # with a short capacity margin that much earlier (O505's fixed candidates: every group, also later with a
        # negative extra); each group then samples its deviations at its own tOptMax + max(extra, 0) before
        # occupancy (group_sampled).
        self.group_extra = None
        self.group_sampled = np.zeros(len(self.group_slices), dtype=bool)
        # Option audit fix O506 (2026-11-05): with group_extra_cool set (not None) the directions take their own extra
        # (warm-up group_extra, gated per step by heat_extra_gate when that is set; cool-down group_extra_cool) and
        # sample on their own schedule; None = the shared extra above (unchanged).
        self.group_extra_cool = None
        self.heat_extra_gate = None
        self.heating = _Direction(n_zones, n_groups, n_day=n_day, slope_floor=slope_floor)
        self.cooling = _Direction(n_zones, n_groups, n_day=n_day, slope_floor=slope_floor)
        self.sampled = False
        self.previous_minute = None

    def update(self, zone_temp, minute_of_day, weekday):
        """Advance one step; returns (warm_up, cool_down) flags per zone group."""
        if self.previous_minute is not None and minute_of_day < self.previous_minute:
            self.heating.midnight()
            self.cooling.midnight()
        self.previous_minute = minute_of_day
        occupied = is_occupied(minute_of_day, weekday, self.morning_hour, self.evening_hour)
        to_occupancy = seconds_to_next_occupied(minute_of_day, weekday, self.morning_hour,
                                                self.evening_hour)
        deviations = (self.heating_setpoint - zone_temp, zone_temp - self.cooling_setpoint)
        for direction, deviation in zip((self.heating, self.cooling), deviations):
            self._learn(direction, deviation)
        if occupied:
            self.sampled = False
            self.group_sampled[:] = False
            for direction in (self.heating, self.cooling):
                direction.active[:] = False
                direction.group_sampled[:] = False
            return self.heating.active.copy(), self.cooling.active.copy()
        sample_now = not self.sampled and to_occupancy <= self.t_opt_max
        resample = self.continuous and self.sampled and to_occupancy <= self.t_opt_max
        for direction, deviation in zip((self.heating, self.cooling), deviations):
            if resample:
                # O1: groups not yet started take the present deviations (started groups keep theirs)
                waiting = ~direction.active[self.zone_group]
                direction.sampled_deviation = np.where(waiting, deviation, direction.sampled_deviation)
                zone_t_opt = self._zone_t_opt(direction, deviation)
                direction.zone_t_opt = np.where(waiting, zone_t_opt, direction.zone_t_opt)
                group_t_opt = np.array([zone_t_opt[s].max() for s in self.group_slices])
                direction.group_t_opt = np.where(direction.active, direction.group_t_opt, group_t_opt)
            if self.group_extra is not None and self.group_extra_cool is not None:
                # option audit fix O506: per-direction extra and samples
                extra = self.group_extra if direction is self.heating else self.group_extra_cool
                if direction is self.heating and self.heat_extra_gate is not None:
                    extra = np.where(self.heat_extra_gate, extra, 0.0)
                due = ~direction.group_sampled & (to_occupancy <= self.t_opt_max + np.maximum(extra, 0.0))
                if due.any():
                    zones_due = due[self.zone_group]
                    direction.sampled_deviation = np.where(zones_due, deviation, direction.sampled_deviation)
                    direction.zone_t_opt = np.where(zones_due, self._zone_t_opt(direction, deviation),
                                                    direction.zone_t_opt)
                    group_t_opt = np.array([direction.zone_t_opt[s].max() for s in self.group_slices])
                    direction.group_t_opt = np.where(due, group_t_opt, direction.group_t_opt)
                direction.group_sampled |= due
                limit = direction.group_t_opt + np.where(direction.group_t_opt > self.thr_opt_on, extra, 0.0)
            elif self.group_extra is not None:
                # O506: each group samples at its own tOptMax + extra and starts extra earlier than G36's tOpt
                due = ~self.group_sampled & (to_occupancy <= self.t_opt_max + np.maximum(self.group_extra, 0.0))
                if due.any():
                    zones_due = due[self.zone_group]
                    direction.sampled_deviation = np.where(zones_due, deviation, direction.sampled_deviation)
                    direction.zone_t_opt = np.where(zones_due, self._zone_t_opt(direction, deviation),
                                                    direction.zone_t_opt)
                    group_t_opt = np.array([direction.zone_t_opt[s].max() for s in self.group_slices])
                    direction.group_t_opt = np.where(due, group_t_opt, direction.group_t_opt)
                limit = direction.group_t_opt + np.where(direction.group_t_opt > self.thr_opt_on,
                                                         self.group_extra, 0.0)
            else:
                if sample_now:
                    # Step 3: sample at tOptMax before occupancy (also dTCalOn for reqStaUp).
                    direction.sampled_deviation = np.array(deviation, dtype=float, copy=True)
                    direction.zone_t_opt = self._zone_t_opt(direction, deviation)
                    direction.group_t_opt = np.array([direction.zone_t_opt[s].max()
                                                      for s in self.group_slices])
                limit = direction.group_t_opt
            on = ((direction.group_t_opt > self.thr_opt_on) & (to_occupancy > 0)
                  & (to_occupancy <= limit))
            starting = on & ~direction.active
            if starting.any():
                zones = starting[self.zone_group] & (deviation > self.u_low)
                direction.tracking |= zones
                direction.start_deviation = np.where(zones, deviation, direction.start_deviation)
                direction.elapsed = np.where(zones, 0.0, direction.elapsed)
            direction.active = on
        if self.group_extra is not None and self.group_extra_cool is not None:
            self.sampled = bool(self.heating.group_sampled.all() and self.cooling.group_sampled.all())
        elif self.group_extra is not None:
            self.group_sampled |= to_occupancy <= self.t_opt_max + np.maximum(self.group_extra, 0.0)
            self.sampled = bool(self.group_sampled.all())
        elif sample_now:
            self.sampled = True
        return self.heating.active.copy(), self.cooling.active.copy()

    def _learn(self, direction, deviation):
        """Step 1: time to reach the setpoint from the system start, at most tOptMax."""
        if not direction.tracking.any():
            return
        direction.elapsed = np.where(direction.tracking, direction.elapsed + self.step_seconds,
                                     direction.elapsed)
        done = direction.tracking & ((deviation <= self.u_low) | (direction.elapsed >= self.t_opt_max))
        if done.any():
            dt = np.minimum(direction.elapsed, self.t_opt_max)
            slope = direction.start_deviation / np.where(dt > 0, dt, 1.0)
            direction.held = np.where(done, slope, direction.held)
            if self.fit == "affine":
                direction.remember(done, direction.start_deviation, dt)
            direction.tracking &= ~done

    def _zone_t_opt(self, direction, deviation):
        """Each zone's optimal start time for its deviation (step 3), capped at tOptMax; 0 within uHigh."""
        if self.fit != "affine":
            return np.where(deviation > self.u_high, np.minimum(deviation / direction.average, self.t_opt_max), 0.0)
        t_opt = np.array(deviation / direction.average, dtype=float)
        for z in range(len(t_opt)):
            pairs = direction.pairs[z]
            pairs = pairs[~np.isnan(pairs[:, 0])]
            if len(pairs) >= 2 and np.ptp(pairs[:, 0]) > 1e-6:
                b, a = np.polyfit(pairs[:, 0], pairs[:, 1], 1)
                if b >= 0.0:
                    t_opt[z] = max(a + b * float(deviation[z]), 0.0)
        return np.where(deviation > self.u_high, np.minimum(t_opt, self.t_opt_max), 0.0)
