"""Policies with the competition interface: __init__(building_meta_data, action_spec), get_action(obs)."""
from smart_building.policies.constant import ConstantPolicy
from smart_building.policies.g36 import (C3_COMBINATION_CLASSES, C3_VARIANT_CLASSES, V3_CANDIDATE_CLASSES,
                                         G36PolicyV2, G36PolicyV3)
from smart_building.policies.g36_v1 import G36Policy as G36PolicyV1

POLICIES = {
    "b0_constant": ConstantPolicy,
    "b1_r": G36PolicyV1,
    "b1_r_v2": G36PolicyV2,
    "b1_r_v3": G36PolicyV3,
    **V3_CANDIDATE_CLASSES,
    **C3_VARIANT_CLASSES,
    **C3_COMBINATION_CLASSES,
}

from smart_building.g36 import variants as _g36_variants
