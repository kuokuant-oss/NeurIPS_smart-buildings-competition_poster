"""B1-R internal state as student inputs, and the student that reads it (j7b-v5; docs/plan/plan-direction-20261020.md
§10 item 3; the list of the WP4 diagnosis probe, docs/plan/reviews/WP4-student-diagnosis.md §1.2 and §6).

The features are attributes of the B1-R object (policies.g36 G36PolicyV2 and its subclasses, here the j7b-v5
teacher G36PolicyC3LaaOccrowDampfloorReadback): its G36Controller's modes, timers, request latches, integrators
and trim & respond setpoints. They are read *before* the object's get_action for the step, i.e. the state the
previous steps left, which the student would also have at inference: the submission runs the same B1-R object
for the on/off switches and the D-T02 fallback (D-L07), so the state is a deterministic function of the same
observation history in training and at deployment.

Per token (appended after the token's existing inputs, tokens.TokenLayout order: zones in BuildingIndex.vav_ids
order, AHUs 1..K, one hot-water token; the global token gets none):
  zone (20)  PI integrals (heating, cooling); zone state cooling / heating; cooling / heating hold steps; raw
             cooling / heating flags; discharge gate, warm supply and low discharge hysteresis states; cooling,
             static pressure, hot-water reset and heating plant request latches; suppression time left / 1800 s;
             optimal start heating / cooling slopes (K per hour) and tracking flags;
  ahu (13)   SAT trim & respond setpoint - 288 K and its running steps; static pressure trim & respond setpoint
             (kPa) and its running steps; the official AHU loop integral and last error; setback; setup; freeze
             latch; optimal start heating / cooling group times (h) and active flags;
  hws (4)    supply water trim & respond setpoint - 330 K and its running steps; plant enabled; plant steps in
             its state.
Running steps are min(steps, 50), -1 while the device is off. The scaling only keeps raw values near unit size;
the student normalises every input with frozen first-iteration statistics (model.py), these included.

StudentNetworkV5 / StudentPolicyV5 are model.StudentNetwork / StudentPolicy with these inputs; the model file is
model.save_student's format with an extra "privileged" entry (load_student_v5). D-T02: if the state cannot be
read or the network fails, the step uses the B1-R action; D-T03: finite check and clipping as in model.py.
"""
import numpy as np

from smart_building.learned.tokens import TOKEN_TYPES

PRIVILEGED_FORMAT = "b1-l-privileged/1"
ZONE_FEATURES = ("heat_integral", "cool_integral", "state_cool", "state_heat", "cool_hold", "heat_hold", "cool_raw",
                 "heat_raw", "discharge_gate", "warm_supply", "low_discharge", "cooling_latch", "pressure_latch",
                 "hot_water_latch", "plant_latch", "suppression_s", "os_heat_slope", "os_cool_slope",
                 "os_heat_tracking", "os_cool_tracking")
AHU_FEATURES = ("sat_tr_setpoint", "sat_tr_steps", "sp_tr_setpoint", "sp_tr_steps", "official_integral",
                "official_previous", "setback", "setup", "freeze", "os_heat_t_opt", "os_cool_t_opt", "os_heat_active",
                "os_cool_active")
HWS_FEATURES = ("water_tr_setpoint", "water_tr_steps", "plant_enabled", "plant_steps")
FEATURES = {"zone": ZONE_FEATURES, "ahu": AHU_FEATURES, "hws": HWS_FEATURES, "global": ()}
COUNTS = {k: len(v) for k, v in FEATURES.items()}
PRIVILEGED_TYPES = ("zone", "ahu", "hws")


def _steps(loop):
    return -1.0 if loop.steps_on is None else float(min(loop.steps_on, 50))


def controller_state(c):
    """{type: {name: float64 array}} of a G36Controller (the probe's snapshot, same names and scaling)."""
    zl, zs, os_ = c.zone_loops, c.zone_states, c.optimal_start
    zone = {
        "heat_integral": zl.heating.integral, "cool_integral": zl.cooling.integral,
        "state_cool": zs.state == 1, "state_heat": zs.state == 2, "cool_hold": zs.cool_hold, "heat_hold": zs.heat_hold,
        "cool_raw": zs.cool_raw, "heat_raw": zs.heat_raw,
        "discharge_gate": c.discharge_gate, "warm_supply": c.warm_supply, "low_discharge": c.low_discharge,
        "cooling_latch": c.cooling_latch, "pressure_latch": c.pressure_latch, "hot_water_latch": c.hot_water_latch,
        "plant_latch": c.plant_latch, "suppression_s": c.suppression.remaining / 1800.0,
        "os_heat_slope": os_.heating.average * 3600.0, "os_cool_slope": os_.cooling.average * 3600.0,
        "os_heat_tracking": os_.heating.tracking, "os_cool_tracking": os_.cooling.tracking,
    }
    ahu = {
        "sat_tr_setpoint": np.array([loop.setpoint for loop in c.supply_air.loops]) - 288.0,
        "sat_tr_steps": np.array([_steps(loop) for loop in c.supply_air.loops]),
        "sp_tr_setpoint": np.array([loop.setpoint for loop in c.static_pressure.loops]) / 1000.0,
        "sp_tr_steps": np.array([_steps(loop) for loop in c.static_pressure.loops]),
        "official_integral": c.official_loop.integral, "official_previous": c.official_loop.previous,
        "setback": c.setback_setup.setback, "setup": c.setback_setup.setup, "freeze": c.freeze_protection.latched,
        "os_heat_t_opt": os_.heating.group_t_opt / 3600.0, "os_cool_t_opt": os_.cooling.group_t_opt / 3600.0,
        "os_heat_active": os_.heating.active, "os_cool_active": os_.cooling.active,
    }
    hws = {
        "water_tr_setpoint": np.array([c.water.loop.setpoint - 330.0]), "water_tr_steps": np.array([_steps(c.water.loop)]),
        "plant_enabled": np.array([c.plant_enable.enabled]), "plant_steps": np.array([min(c.plant_enable.steps_in_state, 50)]),
    }
    return {kind: {k: np.asarray(v, dtype=np.float64) for k, v in part.items()}
            for kind, part in (("zone", zone), ("ahu", ahu), ("hws", hws))}


def state_features(controller):
    """{type: float32 (count, features)} in FEATURES order for the zone, AHU and hot-water tokens."""
    state = controller_state(controller)
    return {k: np.stack([state[k][name] for name in FEATURES[k]], axis=-1).astype(np.float32) for k in PRIVILEGED_TYPES}


def policy_features(policy):
    """state_features of a B1-R policy object (policies.g36 G36PolicyV2 family); raises if it has no controller."""
    controller = getattr(policy, "_controller", None)
    if controller is None:
        raise RuntimeError("the B1-R object has no controller (construction failed)")
    return state_features(controller)


def append_features(inputs, features):
    """Token inputs {type: (count, dim)} or {type: (samples, count, dim)} with the features appended."""
    out = dict(inputs)
    for k in PRIVILEGED_TYPES:
        out[k] = np.concatenate([inputs[k], np.asarray(features[k], dtype=np.float32)], axis=-1)
    return out


def augmented_input_dims(layout):
    return {k: layout.input_dims[k] + COUNTS[k] for k in TOKEN_TYPES}


def save_student_v5(path, model, *, teacher=None, extra=None):
    from smart_building.learned.model import save_student
    save_student(path, model, teacher=teacher,
                 extra={**(extra or {}), "privileged": {"format": PRIVILEGED_FORMAT,
                                                         "features": {k: list(v) for k, v in FEATURES.items()}}})


def load_student_v5(path):
    """(StudentConfig, state dict, input dims, teacher spec) of a j7b-v5 student file."""
    import torch
    from smart_building.learned.model import MODEL_FORMAT, StudentConfig
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != MODEL_FORMAT:
        raise ValueError(f"{path}: not a {MODEL_FORMAT} file")
    privileged = payload.get("privileged") or {}
    if privileged.get("format") != PRIVILEGED_FORMAT or {k: tuple(v) for k, v in privileged["features"].items()} != FEATURES:
        raise ValueError(f"{path}: not a j7b-v5 student (privileged features {privileged.get('format')})")
    return StudentConfig(**payload["config"]), payload["state_dict"], dict(payload["input_dims"]), payload.get("teacher")


class StudentNetworkV5:
    """model.StudentNetwork with the B1-R state appended to the zone, AHU and hot-water tokens."""

    def __init__(self, building_meta_data, action_spec, config, state_dict, input_dims=None):
        import torch
        from smart_building.learned.model import EntityTransformer, keep_from_state
        from smart_building.learned.tokens import HistoryBuffer, TokenLayout
        self.config = config
        if config.torch_threads:
            torch.set_num_threads(config.torch_threads)                         # REQ-014 (7)
        self.layout = TokenLayout(building_meta_data, action_spec, config.history)
        dims = augmented_input_dims(self.layout)
        if input_dims is not None and dict(input_dims) != dims:
            raise ValueError(f"saved input dims {dict(input_dims)} != {dims}")
        self.model = EntityTransformer(dims, config, keep_from_state(state_dict))
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.model.eval()
        self.history = HistoryBuffer(config.history, self.layout.n_fields)

    def mean_action(self, obs, features):
        """Continuous action (normalised, unclipped); pushes obs into the history first (as StudentNetwork)."""
        import torch
        window = self.history.push(obs)
        if features is None:
            raise RuntimeError("B1-R state unavailable")
        inputs = append_features(self.layout.inputs(window), features)
        with torch.no_grad():
            outputs = self.model({k: torch.from_numpy(v).unsqueeze(0) for k, v in inputs.items()})
        return self.layout.scatter({k: v[0].numpy() for k, v in outputs.items()})


class StudentPolicyV5:
    """The competition interface: B1-R switches and fallback (D-L07, D-T02), the B1-R state read before its call."""

    name = "b1_l_v5"

    def __init__(self, building_meta_data, action_spec, config, state_dict, switches_class, input_dims=None):
        self.network = StudentNetworkV5(building_meta_data, action_spec, config, state_dict, input_dims)
        self.switches = switches_class(building_meta_data, action_spec)        # D-L07
        continuous = action_spec["continuous_action"]
        self._low = np.asarray(continuous.low, dtype=np.float32)
        self._high = np.asarray(continuous.high, dtype=np.float32)
        self.fallback_steps = 0

    def get_action(self, obs):
        try:
            features = policy_features(self.switches)                          # before this step's B1-R call
        except Exception:  # noqa: BLE001 - D-T02 below
            features = None
        rule = self.switches.get_action(obs)
        try:
            continuous = self.network.mean_action(obs, features)
            if not np.all(np.isfinite(continuous)):
                raise FloatingPointError("non-finite student action")          # D-T03
        except Exception:  # noqa: BLE001 - D-T02: the B1-R action for this step
            self.fallback_steps += 1
            return rule
        return {"continuous_action": np.clip(continuous, self._low, self._high).astype(np.float32),
                "discrete_action": rule["discrete_action"]}


def student_v5_policy_class(path, switches_class):
    """A competition policy class (building_meta_data, action_spec) bound to a saved j7b-v5 student file."""
    config, state_dict, input_dims, _ = load_student_v5(path)

    class BoundStudentV5(StudentPolicyV5):
        def __init__(self, building_meta_data, action_spec):
            super().__init__(building_meta_data, action_spec, config, state_dict, switches_class, input_dims)

    return BoundStudentV5
