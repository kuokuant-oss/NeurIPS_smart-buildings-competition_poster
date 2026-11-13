"""Optimal start: warm-up and cool-down timing (WP2 step 2.5: D-R05).

D-R05  MBL Buildings/Controls/OBC/Utilities/OptimalStart.mo with
       BaseClasses/OptimalStartCalculation.mo (notes/04 D): tOptMax 10800 s,
       nDay 3, uLow 0, uHigh 0.5 K, thrOptOn 60 s, initial slope 1/3600 K/s,
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
  * Each midnight the latest learned slope (initially 1/3600 K/s) enters a
    first-in-first-out window of nDay values that starts filled with the
    initial slope; the window mean is the slope of step 3.
  * tOpt is capped at tOptMax; the mode is on while 0 < time to occupancy <=
    tOpt and tOpt > thrOptOn, and ends when occupancy starts. The last tOpt is
    kept until the next sample replaces it (it cannot switch a mode on earlier,
    because the time to occupancy exceeds tOptMax until that sample).
"""
import numpy as np

from smart_building.g36_v1.modes import is_occupied, seconds_to_next_occupied
from smart_building.g36_v1.trim_respond import STEP_SECONDS


class _Direction:
    """Optimal start for one direction (heating -> warm-up, or cooling -> cool-down)."""

    def __init__(self, n_zones, n_groups, *, n_day, initial_slope):
        self.held = np.full(n_zones, float(initial_slope))
        self.window = np.full((n_zones, int(n_day)), float(initial_slope))
        self.average = self.window.mean(axis=1)
        self.zone_t_opt = np.zeros(n_zones)
        self.group_t_opt = np.zeros(n_groups)
        self.active = np.zeros(n_groups, dtype=bool)
        self.tracking = np.zeros(n_zones, dtype=bool)
        self.start_deviation = np.zeros(n_zones)
        self.elapsed = np.zeros(n_zones)

    def midnight(self):
        self.window = np.roll(self.window, -1, axis=1)
        self.window[:, -1] = self.held
        self.average = self.window.mean(axis=1)


class OptimalStart:
    def __init__(self, zone_group, group_slices, *, heating_setpoint, cooling_setpoint,
                 morning_hour, evening_hour, t_opt_max=10800.0, n_day=3, u_low=0.0, u_high=0.5,
                 initial_slope=1.0 / 3600.0, thr_opt_on=60.0, step_seconds=STEP_SECONDS):
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
        self.heating = _Direction(n_zones, n_groups, n_day=n_day, initial_slope=initial_slope)
        self.cooling = _Direction(n_zones, n_groups, n_day=n_day, initial_slope=initial_slope)
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
            for direction in (self.heating, self.cooling):
                direction.active[:] = False
            return self.heating.active.copy(), self.cooling.active.copy()
        sample_now = not self.sampled and to_occupancy <= self.t_opt_max
        for direction, deviation in zip((self.heating, self.cooling), deviations):
            if sample_now:
                # Step 3: sample at tOptMax before occupancy.
                direction.zone_t_opt = np.where(deviation > self.u_high,
                                                np.minimum(deviation / direction.average, self.t_opt_max),
                                                0.0)
                direction.group_t_opt = np.array([direction.zone_t_opt[s].max()
                                                  for s in self.group_slices])
            on = ((direction.group_t_opt > self.thr_opt_on) & (to_occupancy > 0)
                  & (to_occupancy <= direction.group_t_opt))
            starting = on & ~direction.active
            if starting.any():
                zones = starting[self.zone_group] & (deviation > self.u_low)
                direction.tracking |= zones
                direction.start_deviation = np.where(zones, deviation, direction.start_deviation)
                direction.elapsed = np.where(zones, 0.0, direction.elapsed)
            direction.active = on
        if sample_now:
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
            direction.tracking &= ~done
