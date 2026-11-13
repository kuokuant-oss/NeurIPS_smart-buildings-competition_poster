"""Baseline v2 B2-RL competition policy: the final rule plus an hourly choice of its five parameters
(handoff/RUNBOOK-windows-b2rl.md §4.1, §4.3; docs/plan/baseline-v2.md §2.2).

The rule G36PolicyB2R2 acts on every 5-minute step. On the first step and then every DECISION_STEPS (12 = one hour)
steps, a small network maps a feature vector to an action a in [-1, 1]^5 and the rule's parameters become
clip(d + a * h, low, high) (RULE_PARAMETER_RANGES; d = the value the rule was built with, h = half the range), set
through set_rule_parameters before the rule acts on that step. The zero action is the rule itself, so an actor whose
output layer starts near zero starts from the rule (no student, no imitation).

Shared with the trainer (learned/b2rl_env.py): the features, the action mapping and the decision schedule live here
(HourlyRule), so training and the submission compute them the same way. numpy only, no torch (the Kaggle notebook has
no torch): the actor is a JSON of weights and normalisation constants (scripts/b2rl.py export).

Features use the observation only. The competition policy is given obs (smart_buildings_inference_server.Policy), not
the reward or env.get_reward_info_and_response(), so the runbook's "last hour: reward, comfort, energy, carbon" items
are replaced by observation proxies over the last hour: the comfort deviation of the zones while people are in, the
mean reheat valve position and the AHU supply flow relative to the largest flow seen in the episode.

Layouts (the model's "layout"; none = v1): v1 is the five parameters above with the mapping above. v2 (2026-10-28,
docs/plan/reviews/WIN-b2rl-rl-alignment-20261028.md) adds the static pressure cap sp_max_pa and maps each side of the
default on its own, d + a * (high - d) for a >= 0 and d + a * (d - low) for a < 0: four of the five defaults are not
in the middle of their range, so under v1 part of each range cannot be reached and part of the actions clip to the
default. a = 0 is the rule in both.

Anchor (the model's "anchor", optional; 2026-10-28): a dict of parameter values set on the rule before the slow layer
reads its defaults, so d and a = 0 are those values instead of the rule's own. The fixed-parameter rule (the v2 actor's
mean parameters at update 100, held all month) passed both gates on the official simulator (Kaggle j15) and became
the new base (user 2026-10-28: "如果驗證結果不錯就可以提交然後換新底"); an anchored actor starts from it.

Rule configuration (the model's "rule_config", optional; 2026-10-29, rule-side search): whitelisted G36 switches
(policies/g36.py C3_CONFIGURABLE) on top of G36PolicyB2R2's own combination, for switches that are not slow
parameters (zone loop gains, freeze protection). None or {} = G36PolicyB2R2 itself.

Room policy (the model's "room_policy", optional; 2026-10-31, docs/plan/geo-room-stages.md stage M): each zone gets one
of several option dicts of the per-room switches (g36/rooms.py option_index / settings): "fixed", "latin" (data
generation), "tree" (a tree on the zone's physical quantities and the 24 h outside air, re-evaluated at every hourly
decision) or "mix". The controller then uses those per-zone arrays (G36Controller.room_settings). Any failure leaves
the rule with its own switches. With "oat_mode": "episode" (2026-11-01, docs/plan/geo-room-stages-3.md) the tree's
outside air is the running mean since the episode start instead of the 24 h mean (the climate a scenario-level rule
was trained on: its month mean). With "periods" (2026-11-02, docs/plan/geo-room-stages-4.md) a tree may split on the
period of the day (g36/rooms.py period_of: outside the schedule / warm-up / in between / late, from the policy's own
clock and occupancy schedule). The options may also set room_heat_max_ratio and room_discharge_offset_k (occupied
rooms; absent = the rule's own).
"""
from collections import deque
import json
import math
from pathlib import Path

import numpy as np

from smart_building.g36.modes import seconds_to_next_occupied
from smart_building.policies.g36 import RULE_PARAMETER_RANGES, SLOW_PARAMETER_RANGES, G36PolicyB2R2

DECISION_STEPS = 12          # one decision per hour of 5-minute steps
HOUR_STEPS = 12              # the "last hour" windows
DAY_STEPS = 288              # the 24-hour outside air mean
PARAMETERS = tuple(RULE_PARAMETER_RANGES)
OBSERVATION_FEATURES = ("hour_sin", "hour_cos", "weekday", "monday", "to_occupied_start",
                        "outside_air", "outside_air_24h", "occupants",
                        "zone_min_below_low", "zone_mean_from_middle", "zone_max_above_high",
                        "hour_zones_out", "hour_occupied_deviation", "hour_ahu_on", "hour_hot_water_on",
                        "hour_reheat_valve", "hour_supply_flow")
LAYOUTS = {"v1": {"parameters": PARAMETERS, "mapping": "symmetric"},
           "v2": {"parameters": PARAMETERS + ("sp_max_pa",), "mapping": "two_sided"},
           # v3 (2026-10-29): v2 plus the A12 comfort guard width (policies/g36.py SLOW_PARAMETER_RANGES), for the
           # outside-air schedules of the rule-side search; v1 and v2 models are unchanged
           "v3": {"parameters": PARAMETERS + ("sp_max_pa", "sat_cost_guard_k"), "mapping": "two_sided"}}


def layout_features(layout="v1"):
    return OBSERVATION_FEATURES + tuple(f"parameter_{name}" for name in LAYOUTS[layout]["parameters"])


FEATURES = layout_features("v1")
MODEL_FORMAT = "b2rl-actor-v1"
# scripts/build_submission.py writes the exported model (or the extracted rule's tree) here as a JSON string (the
# notebook reads no outside file)
B2RL_MODEL_JSON = None
B2RX_TREE_JSON = None


class ParameterMap:
    """Action a in [-1, 1]^n <-> rule parameters; a = 0 gives the rule's own values exactly. mapping "symmetric"
    (v1): clip(d + a * half range); "two_sided" (v2): each side of the default scaled to its own end of the range."""

    def __init__(self, defaults, ranges=RULE_PARAMETER_RANGES, mapping="symmetric"):
        if mapping not in ("symmetric", "two_sided"):
            raise ValueError(f"unknown parameter mapping {mapping!r}")
        self.names, self.mapping = tuple(ranges), mapping
        self.low = np.array([float(ranges[k][0]) for k in self.names])
        self.high = np.array([float(ranges[k][1]) for k in self.names])
        self.half = (self.high - self.low) / 2.0
        self.middle = (self.high + self.low) / 2.0
        self.default = np.array([float(defaults[k]) for k in self.names])
        if np.any(self.default < self.low) or np.any(self.default > self.high):
            raise ValueError(f"rule defaults {dict(zip(self.names, self.default))} outside {dict(ranges)}")

    def values(self, action):
        a = np.clip(np.asarray(action, dtype=float).reshape(len(self.names)), -1.0, 1.0)
        if self.mapping == "two_sided":
            p = np.where(a >= 0.0, self.default + a * (self.high - self.default),
                         self.default + a * (self.default - self.low))
            p = np.clip(p, self.low, self.high)
        else:
            p = np.clip(self.default + a * self.half, self.low, self.high)
        out = dict(zip(self.names, (float(x) for x in p)))
        if "unoccupied_lead_min" in out:
            out["unoccupied_lead_min"] = int(round(out["unoccupied_lead_min"]))
        return out

    def encode(self, values):
        """Parameters -> [-1, 1] over their whole range (the features' view of the current parameters)."""
        return (np.array([float(values[k]) for k in self.names]) - self.middle) / self.half


class FeatureTracker:
    """The observation history the features need; observe() once per step, features() at a decision."""

    def __init__(self, index):
        self.index = index
        self.low, self.high = (float(v) for v in index.comfort_window)
        self.middle = (self.low + self.high) / 2.0
        self.current = None
        self.hour = deque(maxlen=HOUR_STEPS)       # per-step (zones out, occupied deviation, ahu, hws, valve, flow)
        self.outside = deque(maxlen=DAY_STEPS)     # outside air, deg C
        self.outside_sum, self.outside_n = 0.0, 0  # since the episode start (room policy "oat_mode": "episode")
        self.max_occupants = 0.0
        self.max_flow = 0.0

    def observe(self, obs):
        o = self.index.parse(obs)
        t = np.asarray(o.zone_temp, dtype=float)
        below, above = np.maximum(self.low - t, 0.0), np.maximum(t - self.high, 0.0)
        out = float(np.mean((below > 0.0) | (above > 0.0)))
        deviation = float(np.mean(below + above)) if o.occupants > 0 else 0.0
        flow = float(np.sum(o.ahu_flow))
        self.max_occupants = max(self.max_occupants, float(o.occupants))
        self.max_flow = max(self.max_flow, flow)
        self.hour.append((out, deviation, float(np.mean(o.ahu_on)), float(o.hws_on), float(np.mean(o.valve)), flow))
        self.outside.append(float(np.mean(o.oat)) - 273.15)
        self.outside_sum += self.outside[-1]
        self.outside_n += 1
        self.current = o

    def features(self, parameters):
        o, ix = self.current, self.index
        angle = 2.0 * math.pi * o.minute_of_day / 1440.0
        to_start = seconds_to_next_occupied(o.minute_of_day, o.weekday, ix.morning_start_hour, ix.evening_start_hour)
        t = np.asarray(o.zone_temp, dtype=float)
        hour = np.array(self.hour)
        flow = hour[:, 5] / self.max_flow if self.max_flow > 0.0 else np.zeros(len(hour))
        values = [math.sin(angle), math.cos(angle), float(o.weekday < 5), float(o.weekday == 0), to_start / 36000.0,
                  self.outside[-1] / 20.0, sum(self.outside) / len(self.outside) / 20.0,
                  float(o.occupants) / self.max_occupants if self.max_occupants > 0.0 else 0.0,
                  (float(t.min()) - self.low) / 3.0, (float(t.mean()) - self.middle) / 3.0,
                  (float(t.max()) - self.high) / 3.0,
                  float(hour[:, 0].mean()), float(hour[:, 1].mean()) / 3.0, float(hour[:, 2].mean()),
                  float(hour[:, 3].mean()), float(hour[:, 4].mean()), float(flow.mean())]
        return np.concatenate([np.array(values), np.asarray(parameters, dtype=float)])


class HourlyRule:
    """A rule policy with the hourly slow layer. Per step: due = observe(obs); at a due step features() and
    apply(action); then the rule's get_action(obs). The trainer and G36PolicyB2RL both run exactly this order."""

    def __init__(self, rule_policy, interval=DECISION_STEPS, layout="v1", anchor=None):
        controller = getattr(rule_policy, "_controller", None)
        if controller is None:
            raise RuntimeError("the rule controller could not be built; there are no rule parameters")
        self.rule, self.interval, self.layout = rule_policy, int(interval), layout
        names = LAYOUTS[layout]["parameters"]
        if anchor:
            check_anchor(anchor, layout)
            rule_policy.set_rule_parameters(**anchor)               # the defaults d (a = 0) become the anchor
        self.parameters = {k: getattr(controller.cfg, k) for k in names}
        self.map = ParameterMap(self.parameters, {k: SLOW_PARAMETER_RANGES[k] for k in names},
                                LAYOUTS[layout]["mapping"])
        self.tracker = FeatureTracker(controller.index)
        self.steps = 0

    def observe(self, obs):
        self.tracker.observe(obs)
        due = self.steps % self.interval == 0
        self.steps += 1
        return due

    def features(self):
        return self.tracker.features(self.map.encode(self.parameters))

    def apply(self, action):
        values = self.map.values(action)
        self.rule.set_rule_parameters(**values)
        self.parameters = values
        return values


def check_anchor(anchor, layout="v1"):
    """An anchor is a dict of the layout's parameters, each inside SLOW_PARAMETER_RANGES."""
    names = LAYOUTS[layout]["parameters"]
    for key, value in anchor.items():
        if key not in names:
            raise ValueError(f"anchor parameter {key!r} is not one of layout {layout}'s {names}")
        low, high = SLOW_PARAMETER_RANGES[key]
        if not low <= float(value) <= high:
            raise ValueError(f"anchor {key} = {value!r} outside [{low}, {high}]")
    return anchor


def load_model(source):
    """An exported actor (dict) from a dict, a JSON text or a JSON file path; checks its format."""
    if isinstance(source, dict):
        model = source
    elif isinstance(source, Path) or (isinstance(source, str) and not source.lstrip().startswith("{")):
        model = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        model = json.loads(source)
    if model.get("format") != MODEL_FORMAT:
        raise ValueError(f"not a {MODEL_FORMAT} model: {model.get('format')!r}")
    layout = model.get("layout", "v1")
    if layout not in LAYOUTS:
        raise ValueError(f"unknown model layout {layout!r}")
    if tuple(model["features"]) != layout_features(layout) or tuple(model["parameters"]) != \
            LAYOUTS[layout]["parameters"]:
        raise ValueError("the model's features or parameters differ from this policies/b2rl.py")
    if int(model["interval"]) != DECISION_STEPS:
        raise ValueError(f"model decision interval {model['interval']} != {DECISION_STEPS}")
    if model.get("anchor"):
        check_anchor(model["anchor"], layout)
    if model.get("rule_config"):
        configured_rule_class(model["rule_config"])
    return model


def gated_model(model, building_meta_data, index):
    """The model with its rule gates that are on for this building merged in (pipeline 5 stage 戊, 2026-11-05). A gate
    {"id", "tree" or "trees" (majority vote), "rule_config", optional "room_policy"} is decided once, before the
    episode, from the building's static features (rooms.building_features); trees are rooms.tree_choice dicts with
    leaves 0 (off) / 1 (on). Returns (ids of the gates on, model)."""
    from smart_building.g36 import rooms
    x = rooms.building_features(building_meta_data, index)
    on, out, rc = [], dict(model), dict(model.get("rule_config") or {})
    for gate in model["rule_gates"]:
        votes = [rooms.tree_choice(t, x) for t in (gate.get("trees") or [gate["tree"]])]
        if 2 * sum(votes) > len(votes):
            on.append(gate["id"])
            for key in gate.get("rule_config_remove") or ():    # back to the rule's default (whitelists pin values)
                rc.pop(key, None)
            rc.update(gate.get("rule_config") or {})
            if gate.get("room_policy"):
                out["room_policy"] = gate["room_policy"]
    out["rule_config"] = rc
    return on, out


def configured_rule_class(rule_config):
    """G36PolicyB2R2's combination plus whitelisted switches (C3_CONFIGURABLE; raises on anything else)."""
    from smart_building.policies.g36 import C3_COMBINATIONS, c3_configured_class
    return c3_configured_class({**C3_COMBINATIONS[G36PolicyB2R2.combination], **rule_config})


class Actor:
    """The exported actor's deterministic forward pass (mean action, numpy float64)."""

    def __init__(self, model):
        self.mean = np.asarray(model["obs_mean"], dtype=float)
        self.scale = 1.0 / np.sqrt(np.asarray(model["obs_var"], dtype=float) + float(model["obs_epsilon"]))
        self.clip = float(model["obs_clip"])
        self.layers = [(np.asarray(layer["weight"], dtype=float), np.asarray(layer["bias"], dtype=float))
                       for layer in model["layers"]]

    def __call__(self, features):
        x = np.clip((np.asarray(features, dtype=float) - self.mean) * self.scale, -self.clip, self.clip)
        for i, (weight, bias) in enumerate(self.layers):
            x = weight @ x + bias
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
        return np.clip(x, -1.0, 1.0)


class G36PolicyB2RL:
    """Competition policy: G36PolicyB2R2 with hourly parameters from the exported actor (`model`, else the embedded
    B2RL_MODEL_JSON). Any failure of the slow layer leaves the rule running with its last parameters."""

    name = "b2rl"
    model = None
    rule_class = G36PolicyB2R2

    def __init__(self, building_meta_data, action_spec):
        self.building_meta_data, self.action_spec = building_meta_data, action_spec
        self.diagnostics = {"decisions": 0, "slow_layer_error": None}
        self.slow, model, rule_class = None, None, self.rule_class
        try:
            model = self.model if self.model is not None else load_model(B2RL_MODEL_JSON)
            if model.get("rule_config") and self.rule_class is G36PolicyB2R2:
                rule_class = configured_rule_class(model["rule_config"])
        except Exception as error:  # noqa: BLE001 - the rule alone still controls the building
            self.diagnostics["slow_layer_error"] = repr(error)
        self.rule = rule_class(building_meta_data, action_spec)
        self.gates_on = []
        try:
            if model is not None and model.get("rule_gates") and self.rule_class is G36PolicyB2R2:
                self.gates_on, model = gated_model(model, building_meta_data, self.rule._controller.index)
                if self.gates_on:
                    self.rule = configured_rule_class(model["rule_config"])(building_meta_data, action_spec)
        except Exception as error:  # noqa: BLE001 - the ungated rule keeps control
            self.diagnostics["rule_gate_error"] = repr(error)
        try:
            if model is None:
                raise RuntimeError(self.diagnostics["slow_layer_error"])
            self.actor = Actor(model)
            self.slow = HourlyRule(self.rule, layout=model.get("layout", "v1"), anchor=model.get("anchor"))
        except Exception as error:  # noqa: BLE001 - the rule alone still controls the building
            self.diagnostics["slow_layer_error"] = repr(error)
        self.room_policy = None
        try:
            if model is not None and model.get("room_policy"):
                from smart_building.g36 import rooms
                index = self.rule._controller.index
                self._rooms = rooms
                self.room_features = rooms.room_features(building_meta_data, index)
                self.room_numbers = [int(v.split("_")[-1]) for v in index.vav_ids]
                self.room_policy = model["room_policy"]
                self._room_apply(None)
        except Exception as error:  # noqa: BLE001 - the rule keeps its own switches
            self.diagnostics["room_policy_error"] = repr(error)
            self.room_policy = None
        self.oat_gates = None
        self.oat_gate_min_steps = 0
        try:
            if model is not None and model.get("rule_oat_gates"):
                self.oat_gates = [dict(g, state=None) for g in model["rule_oat_gates"]]
                self.oat_gate_min_steps = int(model.get("rule_oat_gate_min_steps", 0))
                self._gate_apply(None)
        except Exception as error:  # noqa: BLE001 - the rule keeps the configuration it was built with
            self.diagnostics["rule_oat_gate_error"] = repr(error)
            self.oat_gates = None

    def _gate_apply(self, oat):
        """Stage 戊 (2026-11-06, docs/plan/tree-stage-wu-20261106.md): building-wide rule switches by climate. A gate
        {"id", "on": {rule_config values}, "off": {the rule's own values}, "lo", "hi"} is on while the running mean
        outside air since the episode start (deg C) is in (lo, hi] (None = unbounded); unknown (None) = off. The
        rule is built with the "on" values (model rule_config), so the option's machinery exists; a change of state
        replaces those configuration keys (read at every act())."""
        from dataclasses import replace
        ctrl = self.rule._controller
        for g in self.oat_gates:
            on = oat is not None and (g.get("lo") is None or oat > g["lo"]) and (g.get("hi") is None
                                                                                  or oat <= g["hi"])
            if on != g["state"]:
                ctrl.cfg = replace(ctrl.cfg, **(g["on"] if on else g["off"]))
                g["state"] = on
        self.gates_state = {g["id"]: g["state"] for g in self.oat_gates}

    def _room_apply(self, oat, hour=None, period=None):
        """Each zone's option under the room policy, given the 24 h outside air (deg C, None before the first hour),
        the hour of day (for a policy with time blocks) and the period of the day (a policy with "periods")."""
        chosen = self._rooms.option_index(self.room_policy, self.room_features, self.room_numbers, oat, hour, period)
        self.rule._controller.room_settings = self._rooms.settings(self.room_policy["options"], chosen)
        self.room_choice = chosen

    def get_action(self, obs):
        if self.slow is not None:
            try:
                if self.slow.observe(obs):
                    features = self.slow.features()
                    self.slow.apply(self.actor(features))
                    self.diagnostics["decisions"] += 1
                    if self.oat_gates:
                        try:
                            tracker = self.slow.tracker
                            # 2026-11-11: the gate thresholds were learned on the whole-episode mean outside air;
                            # the running mean of the first hours (from 06:00, daytime only) is biased high, so a gate
                            # stays "unknown" (= off) until rule_oat_gate_min_steps 5-min steps (0 = as before).
                            need = max(1, self.oat_gate_min_steps)
                            self._gate_apply(tracker.outside_sum / tracker.outside_n
                                             if tracker.outside_n >= need else None)
                        except Exception as error:  # noqa: BLE001 - keep the last gate states
                            self.diagnostics["rule_oat_gate_error"] = repr(error)
                            self.oat_gates = None
                    if self.room_policy is not None and (self.room_policy["mode"] in ("tree", "mix")
                                                         or self.room_policy.get("blocks")
                                                         or self.room_policy.get("periods")):
                        try:
                            oat = 20.0 * float(features[layout_features(self.slow.layout).index("outside_air_24h")])
                            tracker = self.slow.tracker
                            if self.room_policy.get("oat_mode") == "episode" and tracker.outside_n:
                                oat = tracker.outside_sum / tracker.outside_n
                            index = self.rule._controller.index
                            parsed = index.parse(obs)
                            hour = int(parsed.minute_of_day) // 60
                            period = None
                            if self.room_policy.get("periods"):          # policy's own clock + occupancy schedule
                                period = self._rooms.period_of(self.room_policy["periods"], parsed.minute_of_day,
                                                               parsed.weekday, index.morning_start_hour,
                                                               index.evening_start_hour)
                            self._room_apply(oat, hour, period)
                        except Exception as error:  # noqa: BLE001 - keep the last room settings
                            self.diagnostics["room_policy_error"] = repr(error)
                            self.room_policy = None
            except Exception as error:  # noqa: BLE001 - keep the rule with its last parameters
                self.diagnostics["slow_layer_error"] = repr(error)
                self.slow = None
        return self.rule.get_action(obs)


# -- hourly rule variants: the lookahead search's candidates (scripts/oracle_rollout.py) and the rule extracted from
#    its choices (G36PolicyB2RX); runbook §2.6. Each applies for one hour, then the rule's own configuration returns.
CANDIDATES = (
    ("rule", {}),
    ("margin05", {"params": {"occupied_margin_k": 0.5}}),
    ("econ", {"switches": {"economizer_mode": "supply_air"}}),
    ("sp200", {"sp_max": 200.0}),
    ("ahu_off", {"ahu_off": True}),
    ("precool", {"switches": {"precool_until_hour": 12}}),
    ("evening", {"switches": {"evening_charge_min": 120}}),
    ("opt6", {"params": {"t_opt_max_s": 21600.0}}),
)
CANDIDATE_NAMES = tuple(name for name, _ in CANDIDATES)


def same_as_rule(controller, spec):
    """A candidate whose values the rule already has (nothing to try)."""
    values = {**spec.get("params", {}), **spec.get("switches", {})}
    if spec.get("sp_max") is not None:
        return all(loop.maximum <= spec["sp_max"] for loop in controller.static_pressure.loops)
    return bool(values) and all(getattr(controller.cfg, k) == v for k, v in values.items())


def candidate_eligible(controller, tracker, spec):
    """Whether a candidate can be tried at this decision point: not the rule's own values; AHUs off only during
    occupied hours with every zone more than 0.5 K inside the comfort band."""
    if not spec.get("ahu_off"):
        return not same_as_rule(controller, spec)
    from smart_building.g36.modes import is_occupied
    o, ix = tracker.current, tracker.index
    t = np.asarray(o.zone_temp, dtype=float)
    low, high = ix.comfort_window
    return bool(is_occupied(o.minute_of_day, o.weekday, ix.morning_start_hour, ix.evening_start_hour)
                and np.all(t > low + 0.5) and np.all(t < high - 0.5))


def apply_candidate(controller, spec):
    """Apply a candidate to the rule controller; returns the undo function (the rule's own configuration back).
    Parameters go through set_parameters; switches replace the configuration (read at every act()); the static
    pressure cap is the reset loops' maximum (G36Controller builds them from sp_max_pa at construction)."""
    from dataclasses import replace
    saved_cfg = controller.cfg
    saved_max = [loop.maximum for loop in controller.static_pressure.loops]
    if spec.get("params"):
        controller.set_parameters(**spec["params"])
    if spec.get("switches"):
        controller.cfg = replace(controller.cfg, **spec["switches"])
        controller.set_parameters()                       # the derived targets of the new configuration
    if spec.get("sp_max") is not None:
        for loop in controller.static_pressure.loops:
            loop.maximum = max(loop.minimum, float(spec["sp_max"]))

    def undo():
        controller.cfg = saved_cfg
        controller.set_parameters()
        for loop, maximum in zip(controller.static_pressure.loops, saved_max):
            loop.maximum = maximum

    return undo


def ahu_off(action, n_ahu):
    """The rule's action with every AHU's discrete run command off (the hot-water system as the rule chose)."""
    discrete = np.array(action["discrete_action"], dtype=np.int32)
    discrete[:n_ahu] = 0
    return {"continuous_action": action["continuous_action"], "discrete_action": discrete}


class CandidateTree:
    """A fitted sklearn DecisionTreeClassifier as plain arrays: which candidate, from the features."""

    def __init__(self, tree):
        self.feature, self.threshold = tree["feature"], tree["threshold"]
        self.left, self.right, self.label = tree["left"], tree["right"], tree["label"]

    def __call__(self, features):
        node = 0
        while self.left[node] != -1:
            node = self.left[node] if features[self.feature[node]] <= self.threshold[node] else self.right[node]
        return self.label[node]


class G36PolicyB2RX:
    """Extracted rule: G36PolicyB2R2 plus, every hour, the candidate a small decision tree picks from the features
    (fitted on the lookahead search's choices). Any failure of the hourly layer leaves the plain rule running."""

    name = "b2rx"
    tree = None
    rule_class = G36PolicyB2R2

    def __init__(self, building_meta_data, action_spec):
        self.rule = self.rule_class(building_meta_data, action_spec)
        self.n_ahu = len(building_meta_data["ahu_zone_assignments"])
        self.diagnostics = {"chosen": {}, "slow_layer_error": None}
        self.undo, self.off, self.slow = None, False, None
        try:
            self.choose = CandidateTree(self.tree if self.tree is not None else json.loads(B2RX_TREE_JSON)["tree"])
            self.slow = HourlyRule(self.rule)
        except Exception as error:  # noqa: BLE001
            self.diagnostics["slow_layer_error"] = repr(error)

    def get_action(self, obs):
        if self.slow is not None:
            try:
                if self.slow.observe(obs):
                    if self.undo is not None:
                        self.undo()
                    self.undo, self.off = None, False
                    name = self.choose(self.slow.features())
                    spec = dict(CANDIDATES)[name]
                    controller = self.rule._controller
                    if name != "rule" and candidate_eligible(controller, self.slow.tracker, spec):
                        self.undo, self.off = apply_candidate(controller, spec), bool(spec.get("ahu_off"))
                        self.diagnostics["chosen"][name] = self.diagnostics["chosen"].get(name, 0) + 1
            except Exception as error:  # noqa: BLE001
                self.diagnostics["slow_layer_error"] = repr(error)
                self.slow, self.off = None, False
                try:
                    if self.undo is not None:
                        self.undo()
                except Exception:  # noqa: BLE001
                    pass
        action = self.rule.get_action(obs)
        return ahu_off(action, self.n_ahu) if self.off else action


def b2rx_policy_class(tree, rule_class=None):
    """A competition policy class bound to an exported candidate tree (dict, or a JSON file path)."""
    bound = json.loads(Path(tree).read_text(encoding="utf-8")) if not isinstance(tree, dict) else tree
    base = rule_class or G36PolicyB2R2

    class BoundB2RX(G36PolicyB2RX):
        tree = bound["tree"] if "tree" in bound else bound
        rule_class = base

    return BoundB2RX


def b2rl_policy_class(model, rule_class=None):
    """A competition policy class (building_meta_data, action_spec) bound to an exported actor (dict, JSON text or
    path); rule_class defaults to G36PolicyB2R2 (another rule only for pilot runs)."""
    bound = load_model(model)
    base = rule_class or G36PolicyB2R2

    class BoundB2RL(G36PolicyB2RL):
        model = bound
        rule_class = base

    return BoundB2RL
