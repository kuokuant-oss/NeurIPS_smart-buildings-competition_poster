"""Policies with the competition interface: __init__(building_meta_data, action_spec), get_action(obs)."""
from smart_building.policies.constant import ConstantPolicy
from smart_building.policies.g36 import (C3_COMBINATION_CLASSES, C3_VARIANT_CLASSES, V3_CANDIDATE_CLASSES,
                                         G36PolicyV2, G36PolicyV3)
from smart_building.policies.g36_v1 import G36Policy as G36PolicyV1

# Keys usable wherever the official POLICY_MAP is accepted (scripts/batch_evaluate.py).
POLICIES = {
    "b0_constant": ConstantPolicy,
    "b1_r": G36PolicyV1,         # first submission (ref 57242800), frozen copy in policies/g36_v1.py
    "b1_r_v2": G36PolicyV2,      # REQ-015, REQ-013 (C) and source-audit corrections (2026-10-16)
    "b1_r_v3": G36PolicyV3,      # b1_r_v2 without the hot-water plant outside air lockout (REQ-017, 2026-10-17)
    # WP3 candidates on b1_r_v3 (D-R31 row, 2026-10-17): b1_r_v3_loops_always_active, b1_r_v3_same_step_supply,
    # b1_r_v3_ti900; fixed-name classes in policies/g36.py next to G36PolicyV3.
    **V3_CANDIDATE_CLASSES,
    # WP3 discrete variants on C3 = b1_r_v3_ti900 (G1 champion, 2026-10-17): b1_r_c3_<variant>, classes in
    # policies/g36.py (C3_VARIANTS).
    **C3_VARIANT_CLASSES,
    # WP3 combination candidates on C3 (REQ-019, 2026-10-18): b1_r_c3_<combination>, classes in policies/g36.py
    # (C3_COMBINATIONS), e.g. b1_r_c3_laa_occrow_dampfloor.
    **C3_COMBINATION_CLASSES,
}

# WP3 variants of B1-R (rebased on b1_r_v2), registered as b1_r_v2_<variant name>. The fixed-name classes
# live in smart_building/g36/variants.py, which imports this package; it adds its
# classes to POLICIES when it finishes loading, so the registry is complete
# whichever of the two modules is imported first.
from smart_building.g36 import variants as _g36_variants  # noqa: E402,F401
