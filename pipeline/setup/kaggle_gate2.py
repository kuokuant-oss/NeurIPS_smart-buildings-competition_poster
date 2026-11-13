"""The runbook §7 gates of one policy from a Kaggle gate job (build_b2rl_gate jobs: INNER 29 + S4 first 30 on 5 CPU
kernels), without the full-run pipeline (kaggle_gate.py did that for j14b). Starts the collector
(.tmp/reviewer/rules/kaggle_screen.py, KAGGLE_SCREEN_JOB=<job>; it writes .tmp/reviewer/rules/diag/<name>__*.npz),
waits for it, then per set: b2rl.py from-diag into data/runs/b2rl/eval/<out>-<set>.json (every scenario present, not
all equal to the rule's costs) and b2rl.py compare against this host's rule rows (data/runs/b2rl/eval/rule-<set>.json).
Prints one JSON line with both sets' results and "passed" (both gates), and appends it to pipeline/setup/gates.jsonl.
Host python3, standard library.

python3 -B pipeline/setup/kaggle_gate2.py <job> <policy name> <out prefix> <since "YYYY-MM-DD HH:MM">"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
JOB, NAME, OUT, SINCE = sys.argv[1:5]
RULES = ROOT / ".tmp/reviewer/rules"
EVAL = ROOT / "data/runs/b2rl/eval"
SETS = {"inner": 29, "s4_first": 30}
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           PATH=f"{ROOT}/tools/bin:" + os.environ.get("PATH", ""), KAGGLE_CONFIG_DIR=str(ROOT / ".secrets/kaggle"))
ENV.pop("SB_FASTSIM", None)


def b2rl(*args):
    result = subprocess.run(["python3", "-B", "scripts/b2rl.py", *args], cwd=ROOT, env=ENV, capture_output=True,
                            text=True)
    if result.returncode:
        raise SystemExit(f"b2rl.py {args[0]} failed: {(result.stdout + result.stderr).strip()[-300:]}")
    return result.stdout


def main():
    # the collector gives up at this time (kaggle_screen.py AUTORUN_PAUSE_AT); GATE_PAUSE_AT moves it for later jobs
    env = dict(ENV, KAGGLE_SCREEN_JOB=JOB, AUTORUN_PAUSE_AT=os.environ.get("GATE_PAUSE_AT", "2026-10-29 06:00"))
    code = subprocess.run(["python3", "-B", str(RULES / "kaggle_screen.py")], cwd=ROOT, env=env,
                          stdout=open(RULES / f"kaggle-screen-{JOB}.out", "a", encoding="utf-8"),
                          stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
    lines = (RULES / "autorun.log").read_text(encoding="utf-8").splitlines()
    start = max(i for i, line in enumerate(lines) if f" K collecting {JOB} " in line)
    if code or not any(" K all shards in" in line for line in lines[start:]):
        raise SystemExit(f"the collector ended without every shard (exit {code}): " + " | ".join(lines[-3:]))
    results = {}
    for subset, n in SETS.items():
        out = EVAL / f"{OUT}-{subset}.json"
        b2rl("from-diag", "--name", NAME, "--set", subset, "--since", SINCE, "--out", str(out.relative_to(ROOT)))
        record = json.loads(out.read_text(encoding="utf-8"))
        if record.get("missing") or len(record["rows"]) != n:
            raise SystemExit(f"{subset}: {len(record['rows'])} of {n} rows; missing {record.get('missing')}")
        rule = json.loads((EVAL / f"rule-{subset}.json").read_text(encoding="utf-8"))["rows"]
        if all(abs(record["rows"][s]["cost"] - rule[s]["cost"]) < 1e-12 for s in record["rows"]):
            raise SystemExit(f"{subset}: every cost equals the rule's (the policy did not act)")
        result = json.loads(b2rl("compare", "--policy", str(out.relative_to(ROOT)), "--baseline",
                                 str((EVAL / f"rule-{subset}.json").relative_to(ROOT))))
        results[subset] = {k: result.get(k) for k in ("n", "mean_diff", "se", "better", "worse", "rel", "gate",
                                                      "policy_mean", "baseline_mean")}
    line = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "job": JOB, "policy": NAME, **results,
            "passed": all(results[s]["gate"] for s in SETS)}
    with (ROOT / "pipeline/setup/gates.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(line) + "\n")
    print(json.dumps(line))


if __name__ == "__main__":
    main()
