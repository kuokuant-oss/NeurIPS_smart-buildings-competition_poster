"""Build today's final candidates into the reviewer's scratch folder (nothing in the workspace is written).
  gt_x1: merged one (gt_m1) with the gates o538_2, o522_1, o545_3, o13s removed (same removal rule as
         pipeline/setup/gate_variants_20261111.py): two-lever rule base + 84-option room tree + night-setback gate only.
  gt_x2: gt_x1 with the room policy of the rule version gt_b2 (the 15-option tree): two-lever base + night-setback gate only.
"""
import copy
import json
import sys
from pathlib import Path

MODELS = (__import__("pathlib").Path(__file__).resolve().parents[2] / "data/runs/b2rl/search/models")
OUT = Path(sys.argv[1])
load = lambda n: json.loads((MODELS / f"model-{n}.json").read_text(encoding="utf-8"))
b2, m1, a2 = load("gt_b2"), load("gt_m1"), load("gt_final_a2")
DROP = ("o538_2", "o522_1", "o545_3", "o13s")

x1 = copy.deepcopy(m1)
rc = dict(m1["rule_config"])
for g in m1["rule_oat_gates"]:
    if g["id"] in DROP:
        for k in g["on"]:
            if k in b2["rule_config"]:
                rc[k] = b2["rule_config"][k]
            else:
                rc.pop(k, None)
x1["rule_config"] = rc
x1["rule_oat_gates"] = [copy.deepcopy(g) for g in m1["rule_oat_gates"] if g["id"] not in DROP]
x1.setdefault("source", {})["reviewer_final_20261112"] = "gt_m1 without the gates o538_2, o522_1, o545_3, o13s"
x2 = copy.deepcopy(x1)
x2["room_policy"] = copy.deepcopy(b2["room_policy"])
x2["source"]["reviewer_final_20261112"] = "gt_x1 with the room policy of gt_b2"
OUT.mkdir(parents=True, exist_ok=True)
for n, m in (("gt_x1", x1), ("gt_x2", x2)):
    (OUT / f"model-{n}.json").write_text(json.dumps(m, default=float), encoding="utf-8")

# checks
keys = lambda m: {k: v for k, v in m.items() if k not in ("rule_config", "rule_oat_gates", "room_policy", "source")}
print("x1 gates:", [(g["id"], g["lo"]) for g in x1["rule_oat_gates"]])
print("x1 rule_config minus gt_b2:", {k: v for k, v in x1["rule_config"].items() if b2["rule_config"].get(k, "<none>") != v})
print("gt_b2 rule_config minus x1:", {k: v for k, v in b2["rule_config"].items() if k not in x1["rule_config"]})
print("x1 room_policy == gt_m1's:", x1["room_policy"] == m1["room_policy"], "| other keys equal gt_m1:", keys(x1) == keys(m1))
print("x2 room_policy == gt_b2's:", x2["room_policy"] == b2["room_policy"])
print("x2 rule_config == gt_final_a2 minus hot-water gate key:", {k: v for k, v in a2["rule_config"].items() if x2["rule_config"].get(k, "<none>") != v},
      {k: v for k, v in x2["rule_config"].items() if a2["rule_config"].get(k, "<none>") != v})
print("other keys of gt_b2 == gt_m1 (actor, layout ...):", keys(b2) == keys(m1))
