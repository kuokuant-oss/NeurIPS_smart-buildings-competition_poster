"""B1-R teachers for the B1-L student made of registered C3 switch combinations (WP4).

learned/rollouts.py builds the teacher from {"file", "class"} with the competition constructor
(building_meta_data, action_spec), so a combination of C3 switches needs a class of its own. Each
class here is G36PolicyC3Configured bound to its overrides through policies/g36.py
c3_configured_class (whitelisted switches only); nothing else is added.

G36PolicyC3LoopsReadback: C3 + loops_always_active (D-R31) + dr25_observed_readback (D-R25 (b)),
the J7b teacher of the trial run j7b-r1 (user decision 2026-10-18 02:21; REQ-018 item 9 applied to
that teacher; docs/plan/reviews/WP3-K1.md §4).

G36PolicyC3LaaOccrowDampfloorReadback: the WP3 champion b1_r_c3_laa_occrow_dampfloor (C3 +
loops_always_active + D-R30 (A) occupied_row + D-R30 (B) damper_floor, REQ-019) + dr25_observed_readback
(REQ-018 item 9), the teacher of the J7b main run j7b-v4 (user decision 2026-10-18 19:00).

G36PolicyC3LaaOccrowDampfloorCold3Readback: the REQ-022 candidate b1_r_c3_laa_occrow_dampfloor_cold3 (v4 + setback
discharge floor + forced air at 308.15 K + MBL setback release) + dr25_observed_readback: the student's teacher and
the RL rule controller if the candidate replaces v4 (user decision 2026-10-24; REQ-018 item 9 on INNER decides
between it and the candidate itself).
"""
from smart_building.policies.g36 import C3_COMBINATIONS, c3_configured_class

LOOPS_READBACK = {"loop_activation": "always_active", "command_readback": "observed"}
LAA_OCCROW_DAMPFLOOR_READBACK = {**C3_COMBINATIONS["laa_occrow_dampfloor"], "command_readback": "observed"}
LAA_OCCROW_DAMPFLOOR_COLD3_READBACK = {**C3_COMBINATIONS["laa_occrow_dampfloor_cold3"], "command_readback": "observed"}

G36PolicyC3LoopsReadback = c3_configured_class(LOOPS_READBACK)
G36PolicyC3LoopsReadback.__name__ = G36PolicyC3LoopsReadback.__qualname__ = "G36PolicyC3LoopsReadback"
G36PolicyC3LoopsReadback.name = "b1_r_c3_loops_readback"

G36PolicyC3LaaOccrowDampfloorReadback = c3_configured_class(LAA_OCCROW_DAMPFLOOR_READBACK)
G36PolicyC3LaaOccrowDampfloorReadback.__name__ = "G36PolicyC3LaaOccrowDampfloorReadback"
G36PolicyC3LaaOccrowDampfloorReadback.__qualname__ = "G36PolicyC3LaaOccrowDampfloorReadback"
G36PolicyC3LaaOccrowDampfloorReadback.name = "b1_r_c3_laa_occrow_dampfloor_readback"

G36PolicyC3LaaOccrowDampfloorCold3Readback = c3_configured_class(LAA_OCCROW_DAMPFLOOR_COLD3_READBACK)
G36PolicyC3LaaOccrowDampfloorCold3Readback.__name__ = "G36PolicyC3LaaOccrowDampfloorCold3Readback"
G36PolicyC3LaaOccrowDampfloorCold3Readback.__qualname__ = "G36PolicyC3LaaOccrowDampfloorCold3Readback"
G36PolicyC3LaaOccrowDampfloorCold3Readback.name = "b1_r_c3_laa_occrow_dampfloor_cold3_readback"
