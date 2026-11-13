"""Tree-free rule after the re-optimisation (reviewer 2026-11-12): copy the optimiser's end model as model-pr_opt1.json
and print exactly how it differs from gt_b2 (rule + 15-option room tree) and from gt_x8 (public 0.01056)."""
import json, sys, hashlib
from pathlib import Path
S = Path(sys.argv[1])
end = json.loads((S / "pr" / "end.json").read_text())
m = json.loads((S / "pr" / "models" / f"model-{end['key']}.json").read_text())
M = (__import__("pathlib").Path(__file__).resolve().parents[2] / "data/runs/b2rl/search/models")
b2 = json.loads((M / "model-gt_b2.json").read_text())
x8 = json.loads((S / "model-gt_x8.json").read_text())
assert not m.get("room_policy") and not m.get("rule_oat_gates")
m.setdefault("source", {})["reviewer_pure_rule_20261112"] = {"name": "pr_opt1", "optimiser_state": end["key"],
    "kept": [json.loads(l) for l in (S / "pr" / "kept.jsonl").read_text().splitlines() if l.strip()]}
out = S / "model-pr_opt1.json"
out.write_text(json.dumps(m, default=float))
print("written", out.name, "sha256", hashlib.sha256(out.read_bytes()).hexdigest().upper())
for name, o in (("gt_b2", b2), ("gt_x8", x8)):
    r, q = m["rule_config"], o["rule_config"]
    print(f"vs {name}: rule_config added {sorted(set(r) - set(q))} removed {sorted(set(q) - set(r))} changed { {k: (q[k], r[k]) for k in r if k in q and r[k] != q[k]} }")
    print(f"   layers identical: {m['layers'] == o['layers']} | room_policy in {name}: {bool(o.get('room_policy'))} | gates in {name}: {[g['id'] for g in o.get('rule_oat_gates') or []]}")
print("schedule:", json.dumps({b: end["spec"][b] for b in ("cold", "mild", "hot")}))
print("top-level keys:", sorted(m))
