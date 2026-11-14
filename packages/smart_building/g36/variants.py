"""B1-R variant settings for WP3, on b1_r_v2.

Each variant is a set of G36Config overrides on top of the second version
b1_r_v2 (G36Config defaults). Only switches listed in the register are used:

               valve with the commanded pump pressure (audit decision 19);
               recirculation and SAT commands.

The WP3 candidates on b1_r_v3 (b1_r_v3_loops_always_active, b1_r_v3_same_step_supply,
fixed-name G36PolicyV3 subclasses in policies/g36.py (V3_CANDIDATES), so that a
remote job or a submission notebook needs only the files of b1_r_v3.

the cool-down "reset value" option, which had no source.


Every variant except the base has a module-level policy class with a fixed
name (a G36PolicyV2 subclass that builds itself with the variant's G36Config),
registered in smart_building.policies.POLICIES as b1_r_v2_<variant name>; the
base is b1_r_v2 itself (G36PolicyV2) and the first version stays b1_r
(G36Policy in policies/g36_v1.py). A runner can select a variant by registry key, or by this file and
the class name (remote job policy_sources), without code changes. The earlier
first-version variant classes (b1_r_<variant>, B1RDr06Ti900 ...) were removed on
2026-10-16: no job, result or script referred to them."""
from dataclasses import dataclass, field, fields, replace

from smart_building.policies.g36 import G36Config, G36PolicyV2

PENDING_POINTS_4_5 = "pending approval: points 4-5"
PENDING_POINT_3 = "pending approval: point 3"


@dataclass(frozen=True)
class Variant:
    name: str
    registry: str
    overrides: dict = field(default_factory=dict)
    pending: str = ""


VARIANTS = (
    Variant("base", "b1_r_v2 ( with the (C) and audit corrections)"),
    Variant("dr06_ti_900", "", {"zone_ti_s": 900.0}),
    Variant("dr12_ignore_0", "", {"ignore_override": 0}),
    Variant("dr12_ignore_2", "", {"ignore_override": 2}),
    Variant("dr16_closed_form", " (b)", {"static_pressure_reset": "closed_form"}),
    Variant("dr18_plant_reset", " ablation", {"pump_pressure": "plant_reset"}, PENDING_POINT_3),
    Variant("dr21b_dr22_average", " (b), average",
            {"fan_control": "official_pid", "ahu_zone_aggregate": "average"}, PENDING_POINTS_4_5),
    Variant("dr21b_dr22_worst", " (b), worst",
            {"fan_control": "official_pid", "ahu_zone_aggregate": "worst"}, PENDING_POINTS_4_5),
    Variant("dr28_valve_pi", " ablation", {"valve_control": "pi"}),
    Variant("req013_b_commanded_pressure", " variant, (B)", {"plant_off_valve_inputs": "commanded"}),
    Variant("same_step_supply", " variant",
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
    """See README.md."""
    variant = "dr18_plant_reset"
    name = "b1_r_v2_dr18_plant_reset"


class B1RV2Dr21bDr22Average(_VariantPolicy):
    """See README.md."""
    variant = "dr21b_dr22_average"
    name = "b1_r_v2_dr21b_dr22_average"


class B1RV2Dr21bDr22Worst(_VariantPolicy):
    """See README.md."""
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
