"""B1-L ensemble of the j7b-v5 fold students (plan-direction-20261020 §10): policies/b1l_ensemble.py for students that
read the B1-R internal state (learned/privileged.py).

The switches object is the one B1-R instance (D-L07, D-T02), as in b1l_ensemble.py. Its state is read once per
step, before its get_action, and the same features are given to every student (each keeps its own observation
history, identical since all see the same observations). The continuous action is the mean of the students'
outputs; D-T02: if the state cannot be read, a student raises or the mean is not finite, the step uses the B1-R
action; D-T03: the action is clipped to the action spec.
"""
import numpy as np

from smart_building.learned.privileged import StudentNetworkV5, load_student_v5, policy_features


class EnsembleStudentPolicyV5:
    """The competition interface over N saved j7b-v5 students and one B1-R class."""

    name = "b1_l_ensemble_v5"

    def __init__(self, building_meta_data, action_spec, students, switches_class):
        if not students:
            raise ValueError("the ensemble needs at least one student")
        self.networks = [StudentNetworkV5(building_meta_data, action_spec, config, state_dict, input_dims)
                         for config, state_dict, input_dims in students]
        self.switches = switches_class(building_meta_data, action_spec)      # D-L07
        continuous = action_spec["continuous_action"]
        self._low = np.asarray(continuous.low, dtype=np.float32)
        self._high = np.asarray(continuous.high, dtype=np.float32)
        self.fallback_steps = 0

    def get_action(self, obs):
        try:
            features = policy_features(self.switches)                      # before this step's B1-R call
        except Exception:  # noqa: BLE001 - D-T02 below
            features = None
        rule = self.switches.get_action(obs)
        try:
            outputs = [np.asarray(network.mean_action(obs, features), dtype=np.float64) for network in self.networks]
            continuous = np.mean(outputs, axis=0)
            if not np.all(np.isfinite(continuous)):
                raise FloatingPointError("non-finite ensemble action")      # D-T03
        except Exception:  # noqa: BLE001 - D-T02: the B1-R action for this step
            self.fallback_steps += 1
            return rule
        return {"continuous_action": np.clip(continuous, self._low, self._high).astype(np.float32),
                "discrete_action": rule["discrete_action"]}


def ensemble_policy_class(paths, switches_class):
    """A competition policy class (building_meta_data, action_spec) over saved j7b-v5 student files."""
    students = []
    for path in paths:
        config, state_dict, input_dims, _ = load_student_v5(path)
        students.append((config, state_dict, input_dims))

    class BoundEnsembleV5(EnsembleStudentPolicyV5):
        def __init__(self, building_meta_data, action_spec):
            super().__init__(building_meta_data, action_spec, students, switches_class)

    return BoundEnsembleV5
