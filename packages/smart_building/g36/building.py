"""Metadata indices and observation parsing (WP2 step 2.1: D-R01, D-R02, D-R03).

D-R01  Every index is built from the metadata in __init__; the discrete order
       AHU 1..K then the hot-water system is verified against the metadata and
       the action spec; nothing is kept at module level, so no state can cross
       scenarios (each scenario builds a new policy object).
D-R02  Zone temperatures are de-normalised with the metadata constants. Every
       other field is read raw unless its measurement name has an exact key in
       the normalisation table, which is the official normaliser's own rule
       (smart_control/utils/observation_normalizer.py:48-55, 74-86): the zone
       setpoint sensors and all ahs_ahu_N_* fields are raw, while the hot-water
       fields differential_pressure_setpoint, supervisor_run_command and
       run_status are normalised (notes/01 §F). The request-count fields and the
       zone supply_air_flowrate_setpoint fields are never read.
D-R03  Presence is num_occupants (a raw head count, notes/01 §O); the schedule
       flag comfort_mode_now only selects the comfort or eco band.
"""
from dataclasses import dataclass
import math
import re

import numpy as np

ZONE_TEMPERATURE = "zone_air_temperature_sensor"
DAMPER = "supply_air_damper_percentage_command"
VALVE = "reheat_valve_percentage_command"


class MetadataError(ValueError):
    """The metadata does not have the layout the controller was built for."""


def numeric_suffix(name):
    """Number in an id such as 'ahu_3' (same parsing as baselines/pid.py:149-152)."""
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


@dataclass
class Observation:
    """One parsed observation in physical units (K, Pa, bar, m3/s, fractions)."""
    zone_temp: np.ndarray        # (V,) K, de-normalised (D-R02)
    damper: np.ndarray           # (V,) damper position the box took, 0-1
    valve: np.ndarray            # (V,) reheat valve position, 0-1
    oat: np.ndarray              # (K,) outside air temperature, K
    sat_setpoint: np.ndarray     # (K,) last supply air temperature setpoint, K
    static_pressure: np.ndarray  # (K,) static pressure sensor, Pa (0 while off)
    recirculation: np.ndarray    # (K,) recirculation fraction the AHU took
    ahu_on: np.ndarray           # (K,) bool, supervisor run command
    ahu_flow: np.ndarray         # (K,) supply air flow, m3/s
    water_temp: float            # hot-water tank (supply) temperature, K
    water_setpoint: float        # last hot-water supply temperature setpoint, K
    water_dp: float              # differential pressure sensor, bar (0 while off)
    hws_on: bool
    minute_of_day: int
    weekday: int                 # Monday = 0 (conversion_utils.get_radian_time)
    comfort_now: bool
    comfort_soon: bool
    occupants: float             # raw head count (D-R03)


class BuildingIndex:
    """All observation and action indices of one building, from its metadata."""

    def __init__(self, metadata, action_spec):
        fields = list(metadata["observation_field_names"])
        self.field_index = {name: i for i, name in enumerate(fields)}
        self._norm = metadata["observation_normalization_metadata"]
        self.morning_start_hour = int(metadata["morning_start_hour"])
        self.evening_start_hour = int(metadata["evening_start_hour"])
        self.comfort_window = tuple(float(v) for v in metadata["comfort_temp_window"])
        self.eco_window = tuple(float(v) for v in metadata["eco_temp_window"])

        # D-R01: AHUs in numeric order of their ids; the k-th id must own the
        # k-th positional field prefix ahs_ahu_k_ (config.gin:166-221 names the
        # device of position k 'ahu_k').
        assignments = metadata["ahu_zone_assignments"]
        self.ahu_ids = sorted(assignments, key=numeric_suffix)
        self.n_ahu = len(self.ahu_ids)
        if self.n_ahu < 1:
            raise MetadataError("no air handlers in ahu_zone_assignments")
        for position, ahu_id in enumerate(self.ahu_ids, start=1):
            if numeric_suffix(ahu_id) != position:
                raise MetadataError(f"air handler ids are not ahu_1..ahu_K: {self.ahu_ids}")

        # D-R01: discrete order AHU 1..K, hot-water system last.
        n_discrete = int(metadata["n_discrete_action"])
        if n_discrete != self.n_ahu + 1:
            raise MetadataError(f"{n_discrete} discrete actions for {self.n_ahu} AHUs + 1 hot-water system")
        if tuple(action_spec["discrete_action"].shape) != (n_discrete,):
            raise MetadataError("discrete action spec does not match n_discrete_action")
        for k in range(1, self.n_ahu + 1):
            self._field("ahs", f"ahu_{k}_supervisor_run_command")
        self._field("hws", "supervisor_run_command")
        self.ahu_run_action = np.arange(self.n_ahu)
        self.hws_run_action = self.n_ahu

        # D-R01: continuous indices from the metadata. Per AHU the action tuples
        # run (run command, supply air temperature, static pressure,
        # recirculation) and the hot-water system (run, supply water
        # temperature, differential pressure) comes last (config.gin:517-536),
        # which is what makes the last pressure index the pump's.
        self.n_continuous = int(metadata["n_continuous_action"])
        if tuple(action_spec["continuous_action"].shape) != (self.n_continuous,):
            raise MetadataError("continuous action spec does not match n_continuous_action")
        temp = list(metadata["temp_indices"])
        pressure = list(metadata["pressure_indices"])
        water = list(metadata["water_indices"])
        recirculation = list(metadata["recirculation_indices"])
        if (len(temp), len(pressure), len(water), len(recirculation)) != (
                self.n_ahu, self.n_ahu + 1, 1, self.n_ahu):
            raise MetadataError("plant action index lists do not match K AHUs + 1 hot-water system")
        for k in range(self.n_ahu):
            if pressure[k] != temp[k] + 1 or recirculation[k] != temp[k] + 2:
                raise MetadataError(f"AHU {k + 1} action indices are not adjacent")
        if pressure[self.n_ahu] != water[0] + 1:
            raise MetadataError("hot-water pressure index does not follow its temperature index")
        self.sat_action = np.array(temp)
        self.sp_action = np.array(pressure[:self.n_ahu])
        self.recirculation_action = np.array(recirculation)
        self.water_temp_action = int(water[0])
        self.water_dp_action = int(pressure[self.n_ahu])

        # D-R01: zones grouped by AHU, room_N served by VAV_N (baselines/pid.py:118-120).
        zone_actions = metadata["zone_action_indices"]
        vav_ids, groups, starts = [], [], []
        for g, ahu_id in enumerate(self.ahu_ids):
            rooms = list(assignments[ahu_id])
            if not rooms:
                raise MetadataError(f"{ahu_id} serves no zone")
            starts.append(len(vav_ids))
            for room in rooms:
                vav_ids.append(room.replace("room_", "VAV_"))
                groups.append(g)
        if len(set(vav_ids)) != len(vav_ids):
            raise MetadataError("a zone is assigned to more than one AHU")
        if set(vav_ids) != set(zone_actions):
            raise MetadataError("zones in ahu_zone_assignments differ from zone_action_indices")
        self.vav_ids = vav_ids
        self.n_zones = len(vav_ids)
        self.zone_group = np.array(groups)
        self.group_slices = [slice(start, stop) for start, stop in
                             zip(starts, starts[1:] + [self.n_zones])]
        self.group_sizes = np.array([s.stop - s.start for s in self.group_slices])
        try:
            self.damper_action = np.array([zone_actions[v][DAMPER] for v in vav_ids])
            self.valve_action = np.array([zone_actions[v][VALVE] for v in vav_ids])
        except KeyError as error:
            raise MetadataError(f"zone action index missing: {error}") from error
        covered = np.concatenate([self.sat_action, self.sp_action, self.recirculation_action,
                                  [self.water_temp_action, self.water_dp_action],
                                  self.damper_action, self.valve_action])
        if sorted(covered.tolist()) != list(range(self.n_continuous)):
            raise MetadataError("continuous indices are not covered exactly once")

        # Observation readers: (index, scale, offset) with value = obs * scale + offset (D-R02).
        self._zone_temp = self._vector([(v, ZONE_TEMPERATURE) for v in vav_ids])
        if sorted(self._zone_temp[0].tolist()) != sorted(int(i) for i in metadata["zone_temp_indices"]):
            raise MetadataError("zone temperature fields differ from metadata zone_temp_indices")
        self._damper = self._vector([(v, DAMPER) for v in vav_ids])
        self._valve = self._vector([(v, VALVE) for v in vav_ids])

        def ahu_fields(name):
            return self._vector([("ahs", f"ahu_{k}_{name}") for k in range(1, self.n_ahu + 1)])

        self._oat = ahu_fields("outside_air_temperature_sensor")
        self._sat_setpoint = ahu_fields("supply_air_temperature_setpoint")
        self._static_pressure = ahu_fields("supply_air_static_pressure_sensor")
        self._recirculation = ahu_fields("recirculation_percentage_command")
        self._ahu_run = ahu_fields("supervisor_run_command")
        self._ahu_flow = ahu_fields("supply_air_flowrate_sensor")
        self._water_temp = self._scalar("hws", "supply_water_temperature_sensor")
        self._water_setpoint = self._scalar("hws", "supply_water_temperature_setpoint")
        self._water_dp = self._scalar("hws", "differential_pressure_sensor")
        self._hws_run = self._scalar("hws", "supervisor_run_command")
        self._hod = (self._index("hod_cos_000"), self._index("hod_sin_000"))
        self._dow = (self._index("dow_cos_000"), self._index("dow_sin_000"))
        self._comfort_now = self._index("comfort_mode_now")
        self._comfort_soon = self._index("comfort_mode_soon")
        self._occupants = self._index("num_occupants")

        # Native ranges of every continuous action, for native -> normalised.
        ranges = metadata["action_normalization_metadata"]
        names = {}
        for k in range(1, self.n_ahu + 1):
            names[int(self.sat_action[k - 1])] = f"ahu_{k}_supply_air_temperature_setpoint"
            names[int(self.sp_action[k - 1])] = f"ahu_{k}_supply_air_static_pressure_setpoint"
            names[int(self.recirculation_action[k - 1])] = f"ahu_{k}_recirculation_percentage_command"
        names[self.water_temp_action] = "supply_water_temperature_setpoint"
        names[self.water_dp_action] = "differential_pressure_setpoint"
        for i in self.damper_action:
            names[int(i)] = DAMPER
        for i in self.valve_action:
            names[int(i)] = VALVE
        try:
            table = [ranges[names[i]] for i in range(self.n_continuous)]
        except KeyError as error:
            raise MetadataError(f"no action normalisation for {error}") from error
        self.native_min = np.array([r["min_native_value"] for r in table], dtype=float)
        self.native_max = np.array([r["max_native_value"] for r in table], dtype=float)
        self.normalized_min = np.array([r["min_normalized_value"] for r in table], dtype=float)
        self.normalized_max = np.array([r["max_normalized_value"] for r in table], dtype=float)

    # -- readers ------------------------------------------------------------

    def _index(self, field):
        try:
            return self.field_index[field]
        except KeyError as error:
            raise MetadataError(f"observation field missing: {field}") from error

    def _field(self, device, measurement):
        """(index, scale, offset): de-normalise only on an exact measurement key (D-R02)."""
        index = self._index(f"{device}_{measurement}")
        constants = self._norm.get(measurement)
        if constants is None:
            return index, 1.0, 0.0
        return index, float(constants["sample_std"]), float(constants["sample_mean"])

    def _vector(self, pairs):
        triples = [self._field(device, measurement) for device, measurement in pairs]
        return (np.array([t[0] for t in triples]), np.array([t[1] for t in triples]),
                np.array([t[2] for t in triples]))

    def _scalar(self, device, measurement):
        return self._field(device, measurement)

    @staticmethod
    def _read(obs, reader):
        index, scale, offset = reader
        return obs[index].astype(float) * scale + offset

    @staticmethod
    def _read_scalar(obs, reader):
        index, scale, offset = reader
        return float(obs[index]) * scale + offset

    @staticmethod
    def decode_cycle(cos_value, sin_value, period):
        """Position on a cycle of `period` units from one cos/sin feature pair.

        The pair is (cos, sin) of 2*pi*fraction (regression_building_utils.py:
        expand_time_features, phase 0; conversion_utils.get_radian_time).
        Rounded to the nearest unit: float32 features are exact to ~1e-5 units.
        """
        angle = math.atan2(float(sin_value), float(cos_value)) % (2.0 * math.pi)
        return int(round(angle / (2.0 * math.pi) * period)) % period

    def parse(self, obs):
        obs = np.asarray(obs)
        return Observation(
            zone_temp=self._read(obs, self._zone_temp),
            damper=self._read(obs, self._damper),
            valve=self._read(obs, self._valve),
            oat=self._read(obs, self._oat),
            sat_setpoint=self._read(obs, self._sat_setpoint),
            static_pressure=self._read(obs, self._static_pressure),
            recirculation=self._read(obs, self._recirculation),
            ahu_on=self._read(obs, self._ahu_run) > 0.5,
            ahu_flow=self._read(obs, self._ahu_flow),
            water_temp=self._read_scalar(obs, self._water_temp),
            water_setpoint=self._read_scalar(obs, self._water_setpoint),
            water_dp=self._read_scalar(obs, self._water_dp),
            hws_on=self._read_scalar(obs, self._hws_run) > 0.5,
            # D-R04/D-R05: clock from the hod/dow features (minutes; Monday = 0).
            minute_of_day=self.decode_cycle(obs[self._hod[0]], obs[self._hod[1]], 1440),
            weekday=self.decode_cycle(obs[self._dow[0]], obs[self._dow[1]], 7),
            # D-R03: the flag selects the band; presence is the raw head count.
            comfort_now=float(obs[self._comfort_now]) > 0.5,
            comfort_soon=float(obs[self._comfort_soon]) > 0.5,
            occupants=float(obs[self._occupants]),
        )

    def band(self, comfort_now):
        """D-R03: (heating, cooling) setpoints of the schedule band the flag selects."""
        return self.comfort_window if comfort_now else self.eco_window

    def normalize(self, native):
        """Native continuous values -> normalised action (bounded_action_normalizer.py agent_value)."""
        fraction = (np.asarray(native, dtype=float) - self.native_min) / (self.native_max - self.native_min)
        return fraction * (self.normalized_max - self.normalized_min) + self.normalized_min
