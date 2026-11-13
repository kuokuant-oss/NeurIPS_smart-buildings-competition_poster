"""B0 pipeline-check policy: the official example notebook's constant action.

Every continuous setpoint sits at the middle of its normalised range (0.0) and
every air handler and the hot-water pump are commanded on (1), both clipped into
the action space bounds, exactly as the organizers' TransferPolicy Notebook does
(reference/official/submission-example/transferpolicy-notebook.ipynb). It exists
to prove that a generated submission notebook runs end to end; it is not a
controller and its score carries no information about control quality.
"""
import numpy as np


class ConstantPolicy:
    name = "b0_constant"

    def __init__(self, building_meta_data, action_spec):
        continuous = action_spec["continuous_action"]
        discrete = action_spec["discrete_action"]
        self.action = {
            "continuous_action": np.clip(np.zeros(continuous.shape), continuous.low,
                                         continuous.high).astype(np.float32),
            "discrete_action": np.clip(np.ones(discrete.shape), discrete.low,
                                       discrete.high).astype(np.int32),
        }

    def get_action(self, obs):
        return self.action
