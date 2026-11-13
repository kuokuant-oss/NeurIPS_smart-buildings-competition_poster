"""B1-R variant settings for WP3 (D-P01 discrete screening and ablations), on b1_r_v2.

Each variant is a set of G36Config overrides on top of the second version
b1_r_v2 (G36Config defaults). Only switches listed in the register are used:

  D-P01 rows   D-R06 Ti 900 s; D-R12 ignore count 0, 2; D-R16 (b) closed-form
               static pressure; D-R18 PlantReset; D-R21 (b) official pid on/off
               with D-R22 average or worst zone.
  ablations    D-R28: the D-R07 valve PI instead of the valve inversion;
               D-R28 variant REQ-013 (B): while the plant is off, invert the
               valve with the commanded pump pressure (audit decision 19);
               D-R28 / D-R35 variant same_step_supply: the inversion and the
               D-R35 test use this step's supply air estimate, from this step's
               recirculation and SAT commands (reviewer 2026-10-16 15:47).

The WP3 candidates on b1_r_v3 (b1_r_v3_loops_always_active, b1_r_v3_same_step_supply,
b1_r_v3_ti900; reviewer, D-R31 row, 2026-10-17) are not in this file: they are
fixed-name G36PolicyV3 subclasses in policies/g36.py (V3_CANDIDATES), so that a
remote job or a submission notebook needs only the files of b1_r_v3.

Withdrawn on 2026-10-16 (audit decisions 18, 20): D-R12 I = 1, D-R13 "off" and
the cool-down "reset value" option, which had no source.

Pending, not for WP3 screening until approved (audit decision 21): D-R21 (b)
depends on two readings of REQ-009 items 4 and 5 (U8); D-R18 PlantReset on the
ranges and hold of REQ-011 (3) (U9).

Every variant except the base has a module-level policy class with a fixed
name (a G36PolicyV2 subclass that builds itself with the variant's G36Config),
registered in smart_building.policies.POLICIES as b1_r_v2_<variant name>; the
base is b1_r_v2 itself (G36PolicyV2) and the first version stays b1_r
(G36Policy in policies/g36_v1.py). A runner can select a variant by registry key, or by this file and
the class name (remote job policy_sources), without code changes. The earlier
first-version variant classes (b1_r_<variant>, B1RDr06Ti900 ...) were removed on
2026-10-16: no job, result or script referred to them.
"""
from dataclasses import dataclass, field, fields, replace

from smart_building.policies.g36 import G36Config, G36PolicyV2

PENDING_REQ_009 = "待核准 REQ-009 第 4–5 點"
PENDING_REQ_011 = "待核准 REQ-011 第 3 點"


@dataclass(frozen=True)
class Variant:
    name: str
    registry: str                 # register row(s) the overrides come from
    overrides: dict = field(default_factory=dict)
    pending: str = ""             # open review items the variant depends on


VARIANTS = (
    Variant("base", "b1_r_v2 (D-R24 with the REQ-015, REQ-013 (C) and audit corrections)"),
    Variant("dr06_ti_900", "D-R06", {"zone_ti_s": 900.0}),
    Variant("dr12_ignore_0", "D-R12", {"ignore_override": 0}),
    Variant("dr12_ignore_2", "D-R12", {"ignore_override": 2}),
    Variant("dr16_closed_form", "D-R16 (b)", {"static_pressure_reset": "closed_form"}),
    Variant("dr18_plant_reset", "D-R18 ablation", {"pump_pressure": "plant_reset"}, PENDING_REQ_011),
    Variant("dr21b_dr22_average", "D-R21 (b), D-R22 average",
            {"fan_control": "official_pid", "ahu_zone_aggregate": "average"}, PENDING_REQ_009),
    Variant("dr21b_dr22_worst", "D-R21 (b), D-R22 worst",
            {"fan_control": "official_pid", "ahu_zone_aggregate": "worst"}, PENDING_REQ_009),
    Variant("dr28_valve_pi", "D-R28 ablation (D-R07 valve PI)", {"valve_control": "pi"}),
    Variant("req013_b_commanded_pressure", "D-R28 variant, REQ-013 (B)", {"plant_off_valve_inputs": "commanded"}),
    Variant("same_step_supply", "D-R28 / D-R35 variant (reviewer 2026-10-16 15:47)",
            {"inversion_supply_air": "same_step"}),
)

BY_NAME = {variant.name: variant for variant in VARIANTS}
_FIELDS = {f.name for f in fields(G36Config)}


def make_config(name):
    """G36Config of a variant: b1_r_v2 with the variant's overrides."""
    variant = BY_NAME[name]
    unknown = set(variant.overrides) - _FIELDS
    if unknown:
        raise KeyError(f"{name}: unknown G36Config fields {sorted(unknown)}")
    return replace(G36Config(), **variant.overrides)


class _VariantPolicy(G36PolicyV2):
    """G36PolicyV2 built with one variant's G36Config (competition constructor signature)."""

    variant = None

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec, make_config(self.variant))


class B1RV2Dr06Ti900(_VariantPolicy):
    variant = "dr06_ti_900"
    name = "b1_r_v2_dr06_ti_900"


class B1RV2Dr12Ignore0(_VariantPolicy):
    variant = "dr12_ignore_0"
    name = "b1_r_v2_dr12_ignore_0"


class B1RV2Dr12Ignore2(_VariantPolicy):
    variant = "dr12_ignore_2"
    name = "b1_r_v2_dr12_ignore_2"


class B1RV2Dr16ClosedForm(_VariantPolicy):
    variant = "dr16_closed_form"
    name = "b1_r_v2_dr16_closed_form"


class B1RV2Dr18PlantReset(_VariantPolicy):
    """待核准 REQ-011 第 3 點 (D-R18 PlantReset ranges and hold)."""
    variant = "dr18_plant_reset"
    name = "b1_r_v2_dr18_plant_reset"


class B1RV2Dr21bDr22Average(_VariantPolicy):
    """待核准 REQ-009 第 4–5 點 (D-R21 b readings)."""
    variant = "dr21b_dr22_average"
    name = "b1_r_v2_dr21b_dr22_average"


class B1RV2Dr21bDr22Worst(_VariantPolicy):
    """待核准 REQ-009 第 4–5 點 (D-R21 b readings)."""
    variant = "dr21b_dr22_worst"
    name = "b1_r_v2_dr21b_dr22_worst"


class B1RV2Dr28ValvePi(_VariantPolicy):
    variant = "dr28_valve_pi"
    name = "b1_r_v2_dr28_valve_pi"


class B1RV2Req013BCommandedPressure(_VariantPolicy):
    variant = "req013_b_commanded_pressure"
    name = "b1_r_v2_req013_b_commanded_pressure"


class B1RV2SameStepSupply(_VariantPolicy):
    variant = "same_step_supply"
    name = "b1_r_v2_same_step_supply"


# Registry keys b1_r_v2_<variant name> -> fixed-name classes (the base is b1_r_v2).
POLICY_CLASSES = {cls.name: cls for cls in (
    B1RV2Dr06Ti900, B1RV2Dr12Ignore0, B1RV2Dr12Ignore2, B1RV2Dr16ClosedForm, B1RV2Dr18PlantReset, B1RV2Dr21bDr22Average, B1RV2Dr21bDr22Worst,
    B1RV2Dr28ValvePi, B1RV2Req013BCommandedPressure, B1RV2SameStepSupply)}


def _register():
    """Add the variants to smart_building.policies.POLICIES (see its comment on import order)."""
    from smart_building import policies
    registry = getattr(policies, "POLICIES", None)
    if registry is not None and __name__ == "smart_building.g36.variants":
        registry.update(POLICY_CLASSES)


_register()
