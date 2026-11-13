"""The extracted rule with a per-building online guard (2026-10-28, user: "可以作甲"; docs/plan/reviews/WIN-b2rl-20261027.md
16:1x): G36PolicyB2RX, whose hourly candidate came from a perfect-foresight lookahead on five buildings, can hurt a
building it has not seen. On INNER the static pressure cap (sp200) chosen while heating with cold zones left the
zones further outside the comfort band than the rule did (one Denver building: +0.27 K over 230 hours). The guard
learns that from the building itself, with observations only (the policy sees no reward):

after an hour in which a guarded candidate ran, the occupied comfort deviation of that hour (policies/b2rl.py
FeatureTracker: mean distance of the zones outside the comfort band while people are in, K) is compared with the
hour before it; a rise of more than rise_k is a strike against that candidate in this building, and after `strikes`
strikes the candidate is not used again in this episode (the rule runs instead). A building where the candidate keeps
the zones as they were keeps it. The guard is the H3C idea of an admission check per rule change, with the building's
own recent past as the evidence. Any failure of the hourly layer leaves the plain rule running (as G36PolicyB2RX).
"""
import json
from pathlib import Path

import numpy as np

from smart_building.policies.b2rl import CANDIDATES, G36PolicyB2RX, ahu_off, apply_candidate, candidate_eligible
from smart_building.policies.g36 import G36PolicyB2R2

DEFAULT_GUARD = {"candidates": ["sp200"], "strikes": 2, "rise_k": 0.0}


class G36PolicyB2RXGuarded(G36PolicyB2RX):
    """G36PolicyB2RX plus the per-building guard (`guard`: candidates, strikes, rise_k)."""

    name = "b2rx-guard"
    guard = DEFAULT_GUARD

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.strikes, self.disabled, self.pending = {}, set(), None
        self.diagnostics["guard"] = {"strikes": self.strikes, "disabled": [], "vetoed": 0}

    def hour_deviation(self):
        return float(np.mean([h[1] for h in self.slow.tracker.hour]))

    def get_action(self, obs):
        if self.slow is not None:
            try:
                if self.slow.observe(obs):
                    deviation = self.hour_deviation()        # the hour that just ended
                    if self.pending is not None:
                        name, before = self.pending
                        if deviation > before + float(self.guard["rise_k"]):
                            self.strikes[name] = self.strikes.get(name, 0) + 1
                            if self.strikes[name] >= int(self.guard["strikes"]) and name not in self.disabled:
                                self.disabled.add(name)
                                self.diagnostics["guard"]["disabled"].append(name)
                        self.pending = None
                    if self.undo is not None:
                        self.undo()
                    self.undo, self.off = None, False
                    name = self.choose(self.slow.features())
                    if name in self.disabled:
                        self.diagnostics["guard"]["vetoed"] += 1
                        name = "rule"
                    spec = dict(CANDIDATES)[name]
                    controller = self.rule._controller
                    if name != "rule" and candidate_eligible(controller, self.slow.tracker, spec):
                        self.undo, self.off = apply_candidate(controller, spec), bool(spec.get("ahu_off"))
                        self.diagnostics["chosen"][name] = self.diagnostics["chosen"].get(name, 0) + 1
                        if name in self.guard["candidates"]:
                            self.pending = (name, deviation)
            except Exception as error:  # noqa: BLE001 - the rule alone still controls the building
                self.diagnostics["slow_layer_error"] = repr(error)
                self.slow, self.off = None, False
                try:
                    if self.undo is not None:
                        self.undo()
                except Exception:  # noqa: BLE001
                    pass
        action = self.rule.get_action(obs)
        return ahu_off(action, self.n_ahu) if self.off else action


def guarded_policy_class(source, rule_class=None):
    """A competition policy class bound to a candidate tree with a guard: {"tree": ..., "guard": {...}} (dict or a
    JSON file path); a missing guard key takes DEFAULT_GUARD."""
    bound = json.loads(Path(source).read_text(encoding="utf-8")) if not isinstance(source, dict) else source
    base = rule_class or G36PolicyB2R2

    class BoundB2RXGuarded(G36PolicyB2RXGuarded):
        tree = bound["tree"]
        guard = {**DEFAULT_GUARD, **(bound.get("guard") or {})}
        rule_class = base

    return BoundB2RXGuarded
