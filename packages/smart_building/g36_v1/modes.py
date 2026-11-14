"""Zone groups, schedule and operating modes.

       the metadata morning_start_hour and evening_start_hour; warm-up and
       cool-down are triggered by optimal start, at
       most 3 h before occupancy; setback / setup start when 5 zones (all zones
       if the group has fewer than 5) are outside the eco band, or when the
       group average is, and end when every zone is back inside the band by
       1 K (MBL G36/ZoneGroups/OperationMode.mo documentation 770-878 and
       modes, so they are cleared when the occupied period starts.
       occupied > cool-down, warm-up > setup, setback (freeze protection does
       setpoints follow the mode (MBL ThermalZones/Setpoints.mo:1286-1289):
       occupied, warm-up and cool-down use the occupied (comfort) heating and
       cooling setpoints; unoccupied, setback and setup the unoccupied (eco) ones."""
import numpy as np

MINUTES_PER_DAY = 1440

OCCUPIED, COOL_DOWN, WARM_UP, SETUP, SETBACK, UNOCCUPIED = range(6)
MODE_NAMES = ("occupied", "cool_down", "warm_up", "setup", "setback", "unoccupied")
COMFORT_SETPOINT_MODES = (OCCUPIED, WARM_UP, COOL_DOWN)


def operating_mode(occupied, cool_down, warm_up, setup, setback):
    """See README.md."""
    occupied = np.broadcast_to(np.asarray(occupied, dtype=bool), np.shape(warm_up))
    return np.select([occupied, cool_down, warm_up, setup, setback],
                     [OCCUPIED, COOL_DOWN, WARM_UP, SETUP, SETBACK], UNOCCUPIED)


def zone_group_setpoints(mode, comfort_window, eco_window):
    """See README.md."""
    comfort = np.isin(mode, COMFORT_SETPOINT_MODES)
    heating = np.where(comfort, comfort_window[0], eco_window[0])
    cooling = np.where(comfort, comfort_window[1], eco_window[1])
    return heating, cooling


def is_occupied(minute_of_day, weekday, morning_hour, evening_hour):
    """See README.md."""
    return weekday < 5 and morning_hour * 60 <= minute_of_day < evening_hour * 60


def seconds_to_next_occupied(minute_of_day, weekday, morning_hour, evening_hour):
    """    The next start is the next weekday at morning_start_hour, from the hod/dow
    clock and the metadata."""
    if is_occupied(minute_of_day, weekday, morning_hour, evening_hour):
        return 0
    start = morning_hour * 60
    for days in range(8):
        day = (weekday + days) % 7
        if day < 5 and (days > 0 or minute_of_day < start):
            return (days * MINUTES_PER_DAY + start - minute_of_day) * 60
    raise ValueError("no weekday within a week")


class SetbackSetup:
    """See README.md."""

    def __init__(self, group_slices, *, min_zones=5, release_margin=1.0):
        self.group_slices = list(group_slices)
        self.min_zones = int(min_zones)
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
            needed = min(self.min_zones, temps.size)
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
