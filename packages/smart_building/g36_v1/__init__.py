"""Building blocks of the rule-based backbone B1-R (ASHRAE Guideline 36 style).

Every module implements rows of the decision register in
docs/plan/baseline-v1-build.md §3.3 (D-R01–D-R23); each decision site carries
its register ID in a comment. The policy that composes them is
smart_building.policies.g36.G36Policy.
"""
