"""Building blocks of the rule-based backbone B1-R, second version b1_r_v2 (ASHRAE Guideline 36 style).

Every module implements rows of the decision register in
docs/plan/baseline-v1-build.md §3.3; each decision site carries its register ID
in a comment. The policy that composes them is
smart_building.policies.g36.G36PolicyV2. The first version (b1_r) is frozen in
smart_building.g36_v1 and smart_building.policies.g36_v1.
"""
