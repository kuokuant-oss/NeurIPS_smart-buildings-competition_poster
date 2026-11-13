"""Test-only policy factory for the cmaes task kind (DR1/DR2 plumbing), not a B1 policy.

levels(params) returns a policy class whose continuous action i is params[f"level{i % 3}"] (clipped
to the action space) with every unit on, so different candidates score differently and the CMA-ES
loop has three parameters to optimize, as many as D-P02 tunes (pycma 4.5.0 fails on one-parameter
problems). Never used for any decision.
"""
import numpy as np


def levels(params):
    values = [float(params[f"level{i}"]) for i in range(3)]

    class LevelPolicy:
        name = "cmaes_test_levels"

        def __init__(self, building_meta_data, action_spec):
            continuous = action_spec["continuous_action"]
            discrete = action_spec["discrete_action"]
            pattern = np.array([values[i % 3] for i in range(int(np.prod(continuous.shape)))])
            self.action = {
                "continuous_action": np.clip(pattern.reshape(continuous.shape), continuous.low,
                                             continuous.high).astype(np.float32),
                "discrete_action": np.clip(np.ones(discrete.shape), discrete.low, discrete.high).astype(np.int32),
            }

        def get_action(self, obs):
            return self.action

    return LevelPolicy
