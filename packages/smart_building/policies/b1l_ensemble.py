"""B1-L ensemble of the fold students, for the stage-4 leaderboard check (D-K04; reviewer 2026-10-18).

Source of the combination: notes/07 §R, Roberts et al. 2017 Step 5 ("或集成各 fold 模型"). The
continuous action is the mean of the students' continuous outputs (each student's
learned.model.StudentNetwork.mean_action for the same observation); nothing else is combined.

The students output only continuous actions. The on/off (discrete) actions come from the teacher
(D-L07), one instance of the same teacher class for every student, so no rule is needed to combine
them. D-T02: when any student raises, or the mean is not finite, the step uses the teacher's action;
D-T03: the action is clipped to the action spec, as learned.model.StudentPolicy does for one student.
Each student keeps its own observation history (identical, since all see the same observations).
"""
import numpy as np

from smart_building.learned.model import StudentNetwork, load_student


class EnsembleStudentPolicy:
    """The competition interface over N saved students and one teacher class (D-L07)."""

    name = "b1_l_ensemble"

    def __init__(self, building_meta_data, action_spec, students, switches_class):
        if not students:
            raise ValueError("the ensemble needs at least one student")
        self.networks = [StudentNetwork(building_meta_data, action_spec, config, state_dict)
                         for config, state_dict in students]
        self.switches = switches_class(building_meta_data, action_spec)      # D-L07
        continuous = action_spec["continuous_action"]
        self._low = np.asarray(continuous.low, dtype=np.float32)
        self._high = np.asarray(continuous.high, dtype=np.float32)
        self.fallback_steps = 0

    def continuous_action(self, obs):
        outputs = [np.asarray(network.mean_action(obs), dtype=np.float64) for network in self.networks]
        return np.mean(outputs, axis=0)

    def get_action(self, obs):
        rule = self.switches.get_action(obs)
        try:
            continuous = self.continuous_action(obs)
            if not np.all(np.isfinite(continuous)):
                raise FloatingPointError("non-finite ensemble action")      # D-T03
        except Exception:  # noqa: BLE001 - D-T02: the teacher's action for this step
            self.fallback_steps += 1
            return rule
        return {"continuous_action": np.clip(continuous, self._low, self._high).astype(np.float32),
                "discrete_action": rule["discrete_action"]}


def ensemble_policy_class(paths, switches_class):
    """A competition policy class (building_meta_data, action_spec) over the saved student files."""
    students = []
    for path in paths:
        config, state_dict, _ = load_student(path)
        students.append((config, state_dict))

    class BoundEnsemble(EnsembleStudentPolicy):
        def __init__(self, building_meta_data, action_spec):
            super().__init__(building_meta_data, action_spec, students, switches_class)

    return BoundEnsemble
