"""The condition-based rule as a competition-style policy (2026-10-31; the dashboard's "strongest candidate" and the
first step toward submitting it). G36PolicyB2RL bound to a model (the hourly outside-air schedule) plus a rule JSON of
pipeline/setup/fit_condition_rule.py, following exactly what pipeline/setup/headroom.py does in rule mode: on weekdays a daily
choice at midnight (held 288 steps) and a block choice every 240 minutes from the morning start until the evening start
(held to the next block or the evening), each from the observable features the hourly layer computes before it observes
the step, applied as overrides that the hourly schedule cannot undo; a held choice ends after its steps.

A rule fitted with "geo" needs the building descriptors; here they come from data/runs/stage2/descriptors.json by the
scenario's pool/building (`spec`), which only a local run knows (a submission would compute them from the metadata).

cond_policy_class(model_json, rule_json, spec=None) -> a policy class."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def cond_policy_class(model_json, rule_json, spec=None):
    import headroom
    from smart_building.policies.b2rl import b2rl_policy_class, layout_features
    base = b2rl_policy_class(Path(model_json))
    rule = json.loads(Path(rule_json).read_text(encoding="utf-8"))
    options = {"daily": dict(headroom.DAILY), "block": dict(headroom.BLOCK)}
    geometry = {}
    for kind in ("daily", "block"):
        names = rule[kind].get("geometry_features")
        if names:
            if spec is None:
                raise ValueError("a rule with geometry features needs the scenario spec")
            pool, building, _ = spec.split("/")
            table = json.loads((ROOT / "data/runs/stage2/descriptors.json").read_text(encoding="utf-8"))["buildings"]
            geometry[kind] = {f"geo_{k}": float(table[f"{pool}/{building}"][k]) for k in names}

    class CondPolicy(base):
        def __init__(self, building_meta_data, action_spec):
            super().__init__(building_meta_data, action_spec)
            ix = self.rule._controller.index
            self.morning, self.evening = ix.morning_start_hour * 60, ix.evening_start_hour * 60
            self.held = {"daily": {}, "block": {}}
            self.ends = {"daily": None, "block": None}
            self.steps = 0
            self.choices = {}

        def _features(self):
            values = self.slow.tracker.features(self.slow.map.encode(self.slow.parameters))
            return dict(zip(layout_features(self.slow.layout), map(float, values)))

        def _decide(self, kind, hold):
            features = {**self._features(), **geometry.get(kind, {})}
            best, _ = headroom.choose(rule[kind], features)
            self.held[kind], self.ends[kind] = dict(options[kind][best]), self.steps + hold
            self.choices[best] = self.choices.get(best, 0) + 1
            headroom.apply_overrides(self, self.held["daily"], self.held["block"])

        def get_action(self, obs):
            o = self.rule._controller.index.parse(obs)
            minute, weekday = int(o.minute_of_day), int(o.weekday)
            if weekday < 5 and self.slow.tracker.current is not None:
                if minute == 0:
                    self._decide("daily", headroom.DAY_STEPS)
                if minute in range(self.morning, self.evening, 240):
                    self._decide("block", min(headroom.BLOCK_STEPS, (self.evening - minute) // 5))
            if self.slow.observe(obs):
                self.slow.apply(self.actor(self.slow.features()))
                if self.held["daily"] or self.held["block"]:
                    headroom.apply_overrides(self, self.held["daily"], self.held["block"])
            action = self.rule.get_action(obs)
            self.steps += 1
            for kind in ("daily", "block"):
                if self.ends[kind] is not None and self.steps >= self.ends[kind]:
                    self.held[kind], self.ends[kind] = {}, None
                    headroom.apply_overrides(self, self.held["daily"], self.held["block"])
            return action

    CondPolicy.__name__ = "CondPolicy_" + Path(rule_json).stem
    return CondPolicy
