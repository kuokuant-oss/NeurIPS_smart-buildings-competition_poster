"""Kaggle gates and submission for one per-room geometry candidate (2026-10-31, docs/plan/geo-room-stages.md), started
by pipeline/setup/geo_pipeline.py in the background. Steps (each logged to data/runs/b2rl/geo/pipeline.log; EVENT lines wake
the hourly watcher):
 1. policy module packages/smart_building/policies/b2rl_geo_<name>.py (b2rl_gate_policy.py; .gitignore'd);
 2. register the gate job in data/runs/b2rl/geo/gate-jobs.json (scripts/remote_jobs.py reads it), then
    workspace.py remote define / build / push (INNER 29 + S4 first 30, official simulator, 5 private sb- CPU kernels);
 3. kaggle_gate2.py collects both gates against the original rule (pipeline/setup/gates.jsonl);
 4. consistency: Kaggle INNER against the rule must be within 0.3 points of this host's fast-simulator INNER for the
    same candidate (a per-room switch that silently fell back would show up here);
 5. billing check on the fast simulator (billing_diag.py, BILLING_SIM=rom) over the cold scenarios of S4 first 30: the
    candidate's unbilled gas share, pooled over the heat the boiler supplied in those scenarios, may not exceed
    night1_dcv07's by more than 0.5 points (the boiler billing bug is never used, user 2026-10-29);
 6. submit_candidate.py under the standing rules (it checks the date limit, the daily count, the gates and parity).

python3 -B pipeline/setup/geo_gate.py <candidate name> <model json>   (host python3, standard library)
GATE_RESUME=1 skips steps 1-3 when the candidate's Kaggle gates are already in pipeline/setup/gates.jsonl.
"""
import datetime
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "pipeline/setup"
GEO = ROOT / "data/runs/b2rl/geo"
NAME, MODEL = sys.argv[1], sys.argv[2]
SLUG = NAME.replace("_", "-")
STEM = f"b2rl_geo_{NAME.removeprefix('gr_')}"
CLASS = "G36PolicyB2RLGeo" + "".join(p.capitalize() for p in NAME.removeprefix("gr_").split("_"))
JOB = f"jg-b2rl-{SLUG}"
ENV = dict(os.environ, PATH=f"{ROOT}/tools/bin:" + os.environ.get("PATH", ""),
           KAGGLE_CONFIG_DIR=str(ROOT / ".secrets/kaggle"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
ENV.pop("SB_FASTSIM", None)
SHARDS = ["k1", "k2", "k3", "k4", "k5"]
COLD_S4 = ["val/building_30275/fairbanks_1", "val/building_30275/fairbanks_3", "val/building_30638/fairbanks_1",
           "val/building_33726/denver_1", "val/building_36013/denver_1", "val/building_30275/fairbanks_5"]
# Pipeline 5 (2026-11-04, docs/plan/geo-pipeline5.md): GATE_NO_SUBMIT=1 stops before step 6 (the caller applies its
# own submission rule); GATE_RESULT=<path relative to the repository> receives the verdict as JSON. Unset: unchanged.
NO_SUBMIT = os.environ.get("GATE_NO_SUBMIT") == "1"
RESULT = os.environ.get("GATE_RESULT")
# GATE_SKIP_BILLING=1 (pipeline 5 competition track only, user 2026-11-04: the leaderboard version may use the
# competition-only options, the hot-water billing sawtooth among them): the billing share is measured and logged, not
# a refusal. The paper track never sets it.
SKIP_BILLING = os.environ.get("GATE_SKIP_BILLING") == "1"


def verdict(passed, reason, **extra):
    if RESULT:
        path = ROOT / RESULT
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": NAME, "job": JOB, "passed": passed, "reason": reason,
                                    "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}, indent=1) + "\n",
                        encoding="utf-8")


def log(kind, text):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {kind} gate {NAME}: {text}"
    print(line, flush=True)
    with (GEO / "pipeline.log").open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def run(args, env=None, check=True):
    result = subprocess.run(args, cwd=ROOT, env=env or ENV, capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(f"{' '.join(args[:4])} failed: {(result.stdout + result.stderr).strip()[-500:]}")
    return result.stdout


def mean_cost(rows):
    return st.mean(r["cost"] if "cost" in r else -r["mean_reward"] for r in rows.values())


def main():
    model = json.loads((ROOT / MODEL).read_text(encoding="utf-8"))
    room_keys = ("zone_min_ra_lps_m2", "zone_min_rp_lps_m2", "zone_min_scope", "zone_min_oa_divide",
                 "zone_heat_wall_ref_m", "zone_heat_wall_power")
    extras = {k: v for k, v in model["rule_config"].items() if k in room_keys}
    if model.get("room_policy"):            # stage M: say it is a per-room tree (the options are in the model)
        extras["room_policy"] = f"{model['room_policy']['mode']} over {len(model['room_policy']['options'])} options"
    # only the code-compliant per-room minimum may be gated or submitted (outdoor air -> primary air)
    for option in [model["rule_config"]] + list((model.get("room_policy") or {}).get("options", [])):
        if option.get("zone_min_ra_lps_m2") and not option.get("zone_min_oa_divide"):
            log("ERROR", "a per-room minimum without zone_min_oa_divide (upper bound only); nothing gated")
            verdict(False, "per-room minimum without zone_min_oa_divide")
            return
    if os.environ.get("GATE_RESUME") != "1":     # GATE_RESUME=1: the Kaggle gates are already in gates.jsonl
        push_and_collect(extras)
    check_and_submit(extras)


def push_and_collect(extras):
    # 1. module
    run(["python3", "-B", str(SETUP / "b2rl_gate_policy.py"), MODEL, STEM, CLASS])
    ignore = ROOT / ".gitignore"
    entry = f"packages/smart_building/policies/{STEM}.py"
    if entry not in ignore.read_text(encoding="utf-8").splitlines():
        with ignore.open("a", encoding="utf-8", newline="\n") as f:
            f.write(entry + "\n")
    # 2. job
    registry = GEO / "gate-jobs.json"
    jobs = json.loads(registry.read_text(encoding="utf-8")) if registry.is_file() else {}
    jobs[JOB] = {"name": STEM, "module": entry, "class": CLASS,
                 "stage": f"Per-room geometry ({NAME}): night1_dcv07 + {json.dumps(extras)}, on the gate sets",
                 "purpose": f"G36PolicyB2RL night1_dcv07 schedule + per-room geometry {json.dumps(extras)} x INNER 29 + "
                            "S4 first 30, runbook §7 gates (validation)."}
    registry.write_text(json.dumps(jobs, indent=1) + "\n", encoding="utf-8")
    if not (ROOT / f"configs/remote/{JOB}.json").exists():         # remote/jobs.py job_file
        run(["python3", "-B", "scripts/workspace.py", "remote", "define", JOB])
    run(["python3", "-B", "scripts/workspace.py", "remote", "build", JOB, "--mode", "full", "--shards", *SHARDS])
    since = time.strftime("%Y-%m-%d %H:%M")
    out = run(["python3", "-B", "scripts/workspace.py", "remote", "push", JOB, "--mode", "full", "--shards", *SHARDS,
               "--purpose", jobs[JOB]["purpose"]])
    log("EVENT", f"Kaggle gate job {JOB} pushed ({out.count('successfully pushed')} kernels)")
    # 3. collect
    pause = (datetime.datetime.now() + datetime.timedelta(hours=14)).strftime("%Y-%m-%d %H:%M")
    subprocess.run(["python3", "-B", str(SETUP / "kaggle_gate2.py"), JOB, STEM, STEM.replace("_", "-"), since],
                   cwd=ROOT, env={**ENV, "GATE_PAUSE_AT": pause})


def check_and_submit(extras):
    rows =[json.loads(x) for x in (SETUP / "gates.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    gate = ([r for r in rows if r["job"] == JOB] or [None])[-1]
    if not gate:
        log("ERROR", "no gate result; nothing submitted")
        verdict(False, "no gate result")
        return
    log("EVENT", f"Kaggle gates: INNER {100 * gate['inner']['rel']:+.2f}%, S4 first 30 "
                 f"{100 * gate['s4_first']['rel']:+.2f}% vs the original rule; passed {gate['passed']}")
    if not gate["passed"]:
        verdict(False, "Kaggle gates not passed", gate=gate)
        return
    # 4. consistency with this host's fast simulator
    local = json.loads((GEO / f"inner/{NAME}.json").read_text(encoding="utf-8"))["rows"]
    rule = json.loads((ROOT / "data/runs/b2rl/eval/rule-inner.json").read_text(encoding="utf-8"))["rows"]
    rel_local = 100 * (mean_cost({s: local[s] for s in rule if s in local}) / mean_cost(rule) - 1)
    if abs(rel_local - 100 * gate["inner"]["rel"]) > 0.3:
        log("ERROR", f"Kaggle INNER {100 * gate['inner']['rel']:+.2f}% vs fast simulator {rel_local:+.2f}%: more than "
                     "0.3 points apart (a per-room switch may have fallen back); nothing submitted")
        verdict(False, "Kaggle INNER and the fast simulator more than 0.3 points apart", gate=gate,
                local_inner_rel_pct=rel_local)
        return
    # 5. billing check. The share is pooled over the scenarios: unbilled heat / heat the boiler had to supply, summed
    #    (2026-11-02: the plain mean of per-scenario shares let a scenario with 1.5 MWh of heat, whose share swings by
    #    +-5 points, outweigh 300 MWh ones; gt_t2b was refused at 6.75 vs 6.22 while its pooled share was 5.69 vs 5.65)
    shares, plain = {}, {}
    for policy in (MODEL, "data/runs/b2rl/search/models/model-night1_dcv07.json"):
        name = Path(policy).stem.replace("model-", "")
        files = [ROOT / f"data/runs/b2rl/billing-diag/{name}__{s.replace('/', '__')}-rom.json" for s in COLD_S4]
        if not all(f.is_file() for f in files):
            run(["python3", "-B", str(ROOT / "pipeline/launch/launch.py"), str(SETUP / "billing_diag.py"), "6", policy,
                 *COLD_S4], env={**ENV, "BILLING_SIM": "rom"})
        recs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        billed = sum(r["billed_j"] for r in recs)
        need = sum(r["delivered_j"] + r["loss_j"] + r["stored_j"] for r in recs)
        shares[name] = 1.0 - billed / need
        plain[name] = st.mean(r["unbilled_share"] for r in recs)
    cand, base = shares[Path(MODEL).stem.replace("model-", "")], shares["night1_dcv07"]
    log("EVENT", f"billing check (fast simulator, 6 cold S4 scenarios, pooled over the heat supplied): unbilled gas "
                 f"{100 * cand:.2f}% vs night1_dcv07 {100 * base:.2f}% (plain mean of the scenarios "
                 f"{100 * plain[Path(MODEL).stem.replace('model-', '')]:.2f}% vs {100 * plain['night1_dcv07']:.2f}%)")
    if cand > base + 0.005 and SKIP_BILLING:
        log("EVENT", "unbilled share above night1_dcv07 by more than 0.5 points: allowed (GATE_SKIP_BILLING, the "
                     "competition track of pipeline 5, user 2026-11-04)")
    elif cand > base + 0.005:
        log("ERROR", "unbilled share above the current best by more than 0.5 points; nothing submitted")
        verdict(False, "unbilled gas share above night1_dcv07 by more than 0.5 points", gate=gate,
                unbilled=cand, unbilled_base=base)
        return
    # 6. submit
    gates_file = SETUP / f"gates-{SLUG}.json"
    gates_file.write_text(json.dumps(gate, indent=1) + "\n", encoding="utf-8")
    if NO_SUBMIT:
        log("EVENT", "checks passed; submission left to the caller (GATE_NO_SUBMIT)")
        verdict(True, "checks passed", gate=gate, gates_file=str(gates_file.relative_to(ROOT)), unbilled=cand,
                unbilled_base=base, local_inner_rel_pct=rel_local)
        return
    log("EVENT", "submitting under the standing rules (submit_candidate.py)")
    code = subprocess.run(["python3", "-B", str(SETUP / "submit_candidate.py"), SLUG, "G36PolicyB2RL", MODEL,
                           str(gates_file), f"night1_dcv07 + per-room geometry {json.dumps(extras)}"],
                          cwd=ROOT).returncode
    log("EVENT", f"submit_candidate.py ended with code {code} (see pipeline/setup/submit.log)")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - the watcher wakes the agent on this line
        log("ERROR", repr(error)[-800:])
        verdict(False, f"error: {repr(error)[-600:]}")
        raise
