"""Zone groups, schedule and operating modes (WP2 step 2.5: D-R04; clock for D-R05).

D-R04  Each AHU is one zone group. Occupied = weekday (Monday-Friday) between
       the metadata morning_start_hour and evening_start_hour; warm-up and
       cool-down are triggered by optimal start (D-R05, optimal_start.py), at
       most 3 h before occupancy; setback / setup start when 5 zones (all zones
       if the group has fewer than 5) are outside the eco band, or when the
       group average is, and end when every zone is back inside the band by
       1 K (MBL G36/ZoneGroups/OperationMode.mo documentation 770-878 and
       ZoneStatus.mo bouLim 1 K, notes/04 C8). Setback / setup are unoccupied
       modes, so they are cleared when the occupied period starts.
       REQ-022 (C3, reviewer 2026-10-24): SetbackSetupZoneHysteresis starts and
       ends setback as MBL @11e34d2 does (per-zone hysteresis, edge-triggered
       latch); SetbackSetup (every zone 1 K above in the same step) stays the
       default and keeps the setup latch of both.
D-R26  One mode per zone group, by priority (MBL OperationMode.mo:252-286):
       occupied > cool-down, warm-up > setup, setback, freeze protection (D-R32:
       a freeze-protection group is in setback, REQ-016 point 4); within a tier
       the cooling mode wins. Zone
       setpoints follow the mode (MBL ThermalZones/Setpoints.mo:1286-1289):
       occupied, warm-up and cool-down use the occupied (comfort) heating and
       cooling setpoints; unoccupied, setback and setup the unoccupied (eco) ones.
"""
import numpy as np

MINUTES_PER_DAY = 1440

# D-R26 priority order.
OCCUPIED, COOL_DOWN, WARM_UP, SETUP, SETBACK, UNOCCUPIED = range(6)
MODE_NAMES = ("occupied", "cool_down", "warm_up", "setup", "setback", "unoccupied")
COMFORT_SETPOINT_MODES = (OCCUPIED, WARM_UP, COOL_DOWN)


def operating_mode(occupied, cool_down, warm_up, setup, setback):
    """D-R26: the mode of each zone group, first true in priority order."""
    occupied = np.broadcast_to(np.asarray(occupied, dtype=bool), np.shape(warm_up))
    return np.select([occupied, cool_down, warm_up, setup, setback],
                     [OCCUPIED, COOL_DOWN, WARM_UP, SETUP, SETBACK], UNOCCUPIED)


def zone_group_setpoints(mode, comfort_window, eco_window):
    """D-R26: (heating, cooling) setpoints per group from its mode."""
    comfort = np.isin(mode, COMFORT_SETPOINT_MODES)
    heating = np.where(comfort, comfort_window[0], eco_window[0])
    cooling = np.where(comfort, comfort_window[1], eco_window[1])
    return heating, cooling


def is_occupied(minute_of_day, weekday, morning_hour, evening_hour):
    """D-R04: occupied schedule (weekday, between the metadata hours)."""
    return weekday < 5 and morning_hour * 60 <= minute_of_day < evening_hour * 60


def seconds_to_next_occupied(minute_of_day, weekday, morning_hour, evening_hour):
    """D-R05: seconds until the next occupied start (0 while occupied).

    The next start is the next weekday at morning_start_hour, from the hod/dow
    clock and the metadata (notes/04 D: tNexOcc).
    """
    if is_occupied(minute_of_day, weekday, morning_hour, evening_hour):
        return 0
    start = morning_hour * 60
    for days in range(8):
        day = (weekday + days) % 7
        if day < 5 and (days > 0 or minute_of_day < start):
            return (days * MINUTES_PER_DAY + start - minute_of_day) * 60
    raise ValueError("no weekday within a week")


def _zone_counts(min_zones, n_groups):
    """D-R04 zone count: one int for every group (the default), or one per group (option catalogue O5)."""
    if np.ndim(min_zones) == 0:
        return int(min_zones)
    counts = [int(v) for v in min_zones]
    if len(counts) != n_groups:
        raise ValueError(f"{len(counts)} zone counts for {n_groups} groups")
    return counts


def _count(min_zones, g):
    return min_zones if isinstance(min_zones, int) else min_zones[g]


class SetbackSetup:
    """D-R04: setback and setup latches per zone group."""

    def __init__(self, group_slices, *, min_zones=5, release_margin=1.0):
        self.group_slices = list(group_slices)
        self.min_zones = _zone_counts(min_zones, len(self.group_slices))
        self.release_margin = float(release_margin)
        self.setback = np.zeros(len(self.group_slices), dtype=bool)
        self.setup = np.zeros(len(self.group_slices), dtype=bool)

    def update(self, zone_temp, occupied, heating_setpoint, cooling_setpoint):
        """heating/cooling_setpoint: the eco (unoccupied) band of this step."""
        if occupied:
            self.setback[:] = False
            self.setup[:] = False
            return self.setback.copy(), self.setup.copy()
        for g, zones in enumerate(self.group_slices):
            temps = zone_temp[zones]
            needed = min(_count(self.min_zones, g), temps.size)
            if self.setback[g]:
                self.setback[g] = not bool(np.all(temps >= heating_setpoint + self.release_margin))
            else:
                self.setback[g] = (int(np.sum(temps < heating_setpoint)) >= needed
                                   or float(temps.mean()) < heating_setpoint)
            if self.setup[g]:
                self.setup[g] = not bool(np.all(temps <= cooling_setpoint - self.release_margin))
            else:
                self.setup[g] = (int(np.sum(temps > cooling_setpoint)) >= needed
                                 or float(temps.mean()) > cooling_setpoint)
        return self.setback.copy(), self.setup.copy()


def hysteresis(u, previous, u_low, u_high):
    """CDL Reals.Hysteresis: (not pre(y) and u > uHigh) or (pre(y) and u >= uLow)."""
    return np.where(previous, u >= u_low, u > u_high)


class SetbackSetupZoneHysteresis:
    """D-R04 setback start and end as MBL @11e34d2 (REQ-022 C3, reviewer 2026-10-24); setup as SetbackSetup.

    MBL G36/ZoneGroups/ZoneStatus.mo: a zone is cold by Hysteresis(TUnoHeaSet - TZon, uLow -0.1, uHigh 0.1) (hys2,
    lines 14-23, 134-137, 222-224) and may end setback by Hysteresis(TZon - TUnoHeaSet, uLow 0, uHigh bouLim 1 K)
    (hys3, lines 4-10, 158-161, 272-274, 234, 248): once more than 1 K above, it stays so until it falls below the
    setpoint. GroupStatus.mo: the group ends setback when every zone may end (MultiAnd, lines 199-201); the group
    average is cold by Hysteresis(-0.1, 0.1) (lines 15-24, 240-243, 397-399; here the average over the group's
    zones, as the G36 text says, OperationMode.mo:842-849). OperationMode.mo: setback starts when more than 4 zones
    (intGreThr t = 4, lines 103-105), all zones or the average are cold, and its latch (lat, lines 106-108, 535-539;
    CDL Logical/Latch.mo:14-19) is set on the rising edge of that start condition and cleared on the rising edge of
    the group end (endSetBac Edge, lines 331-332, 571-573). As SetbackSetup, the latch is cleared while the building
    is occupied; then the start edge memory is cleared too, so a group still cold after occupancy starts again.
    setback_status_band_k = 0.1 is the band; min_zones and release_margin are D-R04's. The setup latch is
    SetbackSetup's, unchanged (`setup` holds its value after each update).
    """

    def __init__(self, group_slices, n_zones, *, min_zones=5, release_margin=1.0, band=0.1):
        self.group_slices = list(group_slices)
        self.min_zones = _zone_counts(min_zones, len(self.group_slices))
        self.release_margin = float(release_margin)
        self.band = float(band)
        groups = len(self.group_slices)
        self.setback = np.zeros(groups, dtype=bool)           # OperationMode.mo lat
        self.setup = np.zeros(groups, dtype=bool)             # SetbackSetup's setup latch, copied after each update
        self.zone_cold = np.zeros(n_zones, dtype=bool)        # ZoneStatus.mo hys2 pre(y)
        self.zone_may_end = np.zeros(n_zones, dtype=bool)     # ZoneStatus.mo hys3 pre(y)
        self.average_cold = np.zeros(groups, dtype=bool)      # GroupStatus.mo hys pre(y)
        self.start_previous = np.zeros(groups, dtype=bool)    # pre(u) of the latch input (set on its rising edge)
        self.end_previous = np.zeros(groups, dtype=bool)      # pre(u) of the Edge before lat.clr
        self.first = True                                     # Latch.mo initial equation: y = not clr and u
        self.setup_latches = SetbackSetup(self.group_slices, min_zones=min_zones, release_margin=release_margin)

    def update(self, zone_temp, occupied, heating_setpoint, cooling_setpoint):
        """heating/cooling_setpoint: the eco (unoccupied) band of this step. Returns (setback, setup) copies."""
        t = np.asarray(zone_temp, dtype=float)
        self.zone_cold = hysteresis(heating_setpoint - t, self.zone_cold, -self.band, self.band)
        self.zone_may_end = hysteresis(t - heating_setpoint, self.zone_may_end, 0.0, self.release_margin)
        groups = len(self.group_slices)
        start, end = np.zeros(groups, dtype=bool), np.zeros(groups, dtype=bool)
        for g, zones in enumerate(self.group_slices):
            self.average_cold[g] = hysteresis(np.array(heating_setpoint - t[zones].mean()),
                                              np.array(self.average_cold[g]), -self.band, self.band)
            cold = int(self.zone_cold[zones].sum())
            start[g] = cold > _count(self.min_zones, g) - 1 or cold == zones.stop - zones.start or bool(self.average_cold[g])
            end[g] = bool(self.zone_may_end[zones].all())
        if occupied:
            self.setback[:] = False
            self.start_previous[:] = False
        else:
            clear = end & ~self.end_previous
            if self.first:
                self.setback = ~clear & start
            else:
                self.setback = np.where(clear, False, np.where(start & ~self.start_previous, True, self.setback))
            self.start_previous = start
        self.end_previous = end
        _, self.setup = self.setup_latches.update(zone_temp, occupied, heating_setpoint, cooling_setpoint)
        self.first = False
        return self.setback.copy(), self.setup.copy()


class FreezeProtection:
    """D-R32 (audit decision 13): freeze-protection setback detection per zone group.

    MBL ZoneGroups/OperationMode.mo (11e34d2): TZonFreProOn 277.15 K, TZonFreProOff
    280.15 K (lines 10-21); Hysteresis(277.15 - TZonMin, uLow -0.1, uHigh 0.1) sets
    and Hysteresis(TZonMin - 280.15, uLow -0.1, uHigh 0.1) clears the latch lat1
    (lines 145-155, 305-317, 412-414, 527-533); not activated in occupied, warm-up
    or cool-down (line 259); documentation 855-859 ("shall enter setback mode until
    all zones are above 7 C"). The MBL blocks downstream of the mode (Setpoints,
    ActiveAirFlow, DamperValves, SupplyFan, SupplyTemperature) have no branch for
    this mode. REQ-016 point 4, option (A): the controller treats an active group
    as setback (G36 text), in the setback tier of the D-R26 priority.
    """

    def __init__(self, group_slices, *, on=277.15, off=280.15, band=0.1):
        self.group_slices = list(group_slices)
        self.on, self.off, self.band = float(on), float(off), float(band)
        n = len(self.group_slices)
        self.cold = np.zeros(n, dtype=bool)      # hys9 pre(y)
        self.warm = np.zeros(n, dtype=bool)      # hys10 pre(y)
        self.latched = np.zeros(n, dtype=bool)   # lat1
        self.active_group_steps = 0              # diagnostics

    def update(self, zone_temp, blocked):
        """blocked: groups in occupied, warm-up or cool-down mode. Returns the active flags."""
        t_min = np.array([zone_temp[s].min() for s in self.group_slices])
        cold = hysteresis(self.on - t_min, self.cold, -self.band, self.band)
        warm = hysteresis(t_min - self.off, self.warm, -self.band, self.band)
        clear, set_ = warm & ~self.warm, cold & ~self.cold
        self.latched = np.where(clear, False, np.where(set_, True, self.latched))
        self.cold, self.warm = cold, warm
        active = self.latched & ~np.asarray(blocked, dtype=bool)
        self.active_group_steps += int(active.sum())
        return active
