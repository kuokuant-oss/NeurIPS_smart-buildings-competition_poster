"""pr_room = the tuned rule (pr_opt2, all its building-wide settings, night setback in all climates) with only the
per-room selection (room_policy) of the rule with the learned layer (gt_x8) added."""
import json, sys
from pathlib import Path
D = Path(sys.argv[1])
m = json.loads((D / "model-pr_opt2.json").read_text()); x = json.loads((D / "model-gt_x8.json").read_text())
assert not m.get("room_policy") and not m.get("rule_oat_gates")
m["room_policy"] = x["room_policy"]
m.setdefault("source", {})["reviewer_pr_room_20261113"] = "pr_opt2 + room_policy of gt_x8"
(D / "model-pr_room.json").write_text(json.dumps(m, default=float))
print("added room_policy with", len(x["room_policy"]["options"]), "options; gt_x8 gates not copied:", [g["id"] for g in x.get("rule_oat_gates") or []])
print("room_policy keys", sorted(x["room_policy"]))
