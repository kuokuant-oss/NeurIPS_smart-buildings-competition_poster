"""Entity tokens of the student B1-L.

       global token. Each token's input is its entity's observation fields
       plus static features computed in __init__ from the metadata floor_plan
       and zone_map; tokens of one type
       share a linear tokenizer (model.py).
       feature vectors.

  (1) dynamic features = the entity's observation fields as delivered, minus
      supply_air_flowrate_setpoint, each AHU's cooling_request_count and the
      hot-water heating_request_count (CONSTANT_FIELDS). Scaling is the
      model's input normalisation (model.py), not done here;
  (2) static features as defined in static_features, normalised like the
      dynamic ones (model.py);
  (3) the history is padded with the oldest observation at the episode start
      (approved 2026-10-16).
Zone, AHU and action indices come from smart_building.g36.building
, so the student and B1-R read the metadata the same way."""
import numpy as np

from smart_building.g36.building import BuildingIndex

ZONE, AHU, HWS, GLOBAL = "zone", "ahu", "hws", "global"
TOKEN_TYPES = (ZONE, AHU, HWS, GLOBAL)
HISTORY_LENGTHS = (12, 64, 288)
OUTPUT_DIMS = {ZONE: 2, AHU: 3, HWS: 2, GLOBAL: 0}
GLOBAL_FIELDS = ("comfort_mode_now", "comfort_mode_soon", "num_occupants")
CV_AREA_M2 = 0.2 * 0.2
CONSTANT_FIELDS = {ZONE: ("supply_air_flowrate_setpoint",), AHU: ("cooling_request_count",),
                   HWS: ("heating_request_count",)}


def _entity_fields(fields, prefix, dropped=()):
    return [i for i, name in enumerate(fields)
            if name.startswith(prefix) and name[len(prefix):] not in dropped]


def static_features(metadata, index):
    """    Rooms and walls come from the official building_utils.construct_building_data_types
    applied to the metadata floor_plan and zone_map, as the simulator itself
    builds them (smart_control/simulator/building.py:836-856).
    Zone columns: area m2, exterior exposure (room cells 4-adjacent to the
    exterior wall shell), floor (building-outline component), neighbours
    (rooms sharing an interior-wall cell), AHU size (zones of its AHU).
    AHU column: its zone count. Hot-water and global tokens have none."""
    from scipy import ndimage
    from smart_buildings.smart_control.simulator import building_utils, constants
    floor_plan = np.asarray(metadata["floor_plan"])
    zone_map = np.asarray(metadata["zone_map"])
    rooms, exterior_walls, interior_walls, exterior_space, _ = building_utils.construct_building_data_types(
        floor_plan=floor_plan, zone_map=zone_map)
    room_id = np.full(np.shape(exterior_walls), -1, dtype=np.int64)
    for z, vav in enumerate(index.vav_ids):
        cells = np.asarray(rooms[vav.replace("VAV_", "room_")])
        room_id[cells[:, 0], cells[:, 1]] = z
    n = index.n_zones
    area = np.bincount(room_id[room_id >= 0], minlength=n) * CV_AREA_M2

    def neighbours_of(mask):
        """Room ids 4-adjacent to the cells of `mask`, as (cell, room) pairs."""
        pairs = []
        for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            moved = np.roll(room_id, shift, axis=(0, 1))
            hit = mask & (moved >= 0)
            pairs.append(np.stack([np.flatnonzero(hit), moved[hit]], axis=1))
        return np.concatenate(pairs)

    shell = exterior_walls == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
    exposed = np.zeros(room_id.shape, dtype=bool)
    for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        exposed |= np.roll(shell, shift, axis=(0, 1))
    exposure = np.bincount(room_id[exposed & (room_id >= 0)], minlength=n).astype(float)

    walls = interior_walls == constants.INTERIOR_WALL_VALUE_IN_FUNCTION
    neighbours = [set() for _ in range(n)]
    by_cell = {}
    for cell, room in neighbours_of(walls):
        by_cell.setdefault(int(cell), set()).add(int(room))
    for touching in by_cell.values():
        for room in touching:
            neighbours[room] |= touching - {room}
    neighbour_count = np.array([len(s) for s in neighbours], dtype=float)

    outline, _ = ndimage.label(exterior_space != constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION)
    floor = np.zeros(n)
    for z in range(n):
        labels = outline[room_id == z]
        floor[z] = float(np.bincount(labels).argmax()) if labels.size else 0.0

    ahu_size = index.group_sizes[index.zone_group].astype(float)
    zone_static = np.stack([area, exposure, floor, neighbour_count, ahu_size], axis=1)
    ahu_static = index.group_sizes.astype(float)[:, None]
    return {ZONE: zone_static, AHU: ahu_static, HWS: np.zeros((1, 0)), GLOBAL: np.zeros((1, 0))}


class TokenLayout:
    """Index tables of one building, built once in __init__."""

    def __init__(self, metadata, action_spec, history, *, with_static=True):
        if history not in HISTORY_LENGTHS:
            raise ValueError(f"history {history} not in {HISTORY_LENGTHS}")
        self.history = int(history)
        self.index = index = BuildingIndex(metadata, action_spec)
        fields = list(metadata["observation_field_names"])
        self.n_fields = len(fields)
        dynamic = {
            ZONE: [_entity_fields(fields, f"{vav}_", CONSTANT_FIELDS[ZONE]) for vav in index.vav_ids],
            AHU: [_entity_fields(fields, f"ahs_ahu_{k}_", CONSTANT_FIELDS[AHU]) for k in range(1, index.n_ahu + 1)],
            HWS: [_entity_fields(fields, "hws_", CONSTANT_FIELDS[HWS])],
            GLOBAL: [[i for i, name in enumerate(fields)
                      if name.startswith(("hod_", "dow_")) or name in GLOBAL_FIELDS]],
        }
        dropped = ([i for vav in index.vav_ids for i, name in enumerate(fields)
                    if name in {f"{vav}_{c}" for c in CONSTANT_FIELDS[ZONE]}]
                   + [i for k in range(1, index.n_ahu + 1) for i, name in enumerate(fields)
                      if name in {f"ahs_ahu_{k}_{c}" for c in CONSTANT_FIELDS[AHU]}]
                   + [i for i, name in enumerate(fields) if name in {f"hws_{c}" for c in CONSTANT_FIELDS[HWS]}])
        covered = sorted([i for rows in dynamic.values() for row in rows for i in row] + dropped)
        if covered != list(range(self.n_fields)):
            raise ValueError("observation fields are not covered exactly once by the tokens and the constant fields")
        self.dropped = np.asarray(sorted(dropped))
        self.dynamic = {}
        for kind, rows in dynamic.items():
            if len({len(row) for row in rows}) != 1 or not rows[0]:
                raise ValueError(f"{kind} tokens do not have the same fields")
            self.dynamic[kind] = np.asarray(rows)
        if with_static:
            self.static = {k: v.astype(np.float32) for k, v in static_features(metadata, index).items()}
        else:
            self.static = {ZONE: np.zeros((index.n_zones, 5), np.float32),
                           AHU: np.zeros((index.n_ahu, 1), np.float32),
                           HWS: np.zeros((1, 0), np.float32), GLOBAL: np.zeros((1, 0), np.float32)}
        self.counts = {kind: self.dynamic[kind].shape[0] for kind in TOKEN_TYPES}
        self.input_dims = {kind: self.history * self.dynamic[kind].shape[1] + self.static[kind].shape[1]
                           for kind in TOKEN_TYPES}

    def inputs(self, window):
        """Token inputs from an (H, n_fields) window, oldest first: {type: (count, input_dim)}."""
        window = np.asarray(window, dtype=np.float32)
        out = {}
        for kind in TOKEN_TYPES:
            idx = self.dynamic[kind]
            stacked = window[:, idx]
            dyn = np.transpose(stacked, (1, 0, 2)).reshape(idx.shape[0], -1)
            out[kind] = np.concatenate([dyn, self.static[kind]], axis=1)
        return out

    def spec(self):
        """Index tables a stored rollout needs to rebuild token inputs and targets without the metadata."""
        ix = self.index
        out = {f"dynamic_{k}": self.dynamic[k].astype(np.int32) for k in TOKEN_TYPES}
        out.update({f"static_{k}": self.static[k] for k in TOKEN_TYPES})
        out.update(damper=np.asarray(ix.damper_action), valve=np.asarray(ix.valve_action),
                   sat=np.asarray(ix.sat_action), sp=np.asarray(ix.sp_action),
                   recirculation=np.asarray(ix.recirculation_action),
                   water_temp=np.asarray([ix.water_temp_action]).reshape(-1),
                   water_dp=np.asarray([ix.water_dp_action]).reshape(-1))
        return out

    def scatter(self, outputs):
        """Continuous action vector from decoder outputs {type: (count, OUTPUT_DIMS[type])}."""
        ix = self.index
        action = np.zeros(ix.n_continuous, dtype=np.float32)
        action[ix.damper_action] = outputs[ZONE][:, 0]
        action[ix.valve_action] = outputs[ZONE][:, 1]
        action[ix.sat_action] = outputs[AHU][:, 0]
        action[ix.sp_action] = outputs[AHU][:, 1]
        action[ix.recirculation_action] = outputs[AHU][:, 2]
        action[ix.water_temp_action] = outputs[HWS][0, 0]
        action[ix.water_dp_action] = outputs[HWS][0, 1]
        return action


def gather_targets(spec, actions):
    """Per-type targets from continuous action vectors (..., n_continuous), the inverse of scatter."""
    actions = np.asarray(actions)
    zone = np.stack([actions[..., spec["damper"]], actions[..., spec["valve"]]], axis=-1)
    ahu = np.stack([actions[..., spec["sat"]], actions[..., spec["sp"]], actions[..., spec["recirculation"]]], axis=-1)
    hws = np.stack([actions[..., spec["water_temp"]], actions[..., spec["water_dp"]]], axis=-1)
    return {ZONE: zone, AHU: ahu, HWS: hws}


def token_inputs(spec, rows, step, history):
    """    exactly as HistoryBuffer + TokenLayout.inputs give them at that step of the episode."""
    window = np.asarray(rows)[np.maximum(np.arange(step - history + 1, step + 1), 0)]
    out = {}
    for kind in TOKEN_TYPES:
        idx = spec[f"dynamic_{kind}"]
        stacked = window[:, idx]
        dyn = np.transpose(stacked, (1, 0, 2)).reshape(idx.shape[0], -1)
        out[kind] = np.concatenate([dyn, spec[f"static_{kind}"]], axis=1).astype(np.float32)
    return out


class HistoryBuffer:
    """See README.md."""

    def __init__(self, history, n_fields):
        self.history = int(history)
        self.buffer = np.zeros((self.history, n_fields), dtype=np.float32)
        self.filled = False

    def push(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if not self.filled:
            self.buffer[:] = obs
            self.filled = True
        else:
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = obs
        return self.buffer
