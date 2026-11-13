"""The runbook §7 gates of the full run's 15 M-step actor from the Kaggle job j14b-b2rl-full15m (user 2026-10-28 08:3x:
"看能不能用kaggle cpu 五台去驗證"), in place of the pipeline's local gate stage.

Starts the job's collector (.tmp/reviewer/rules/kaggle_screen.py; KAGGLE_SCREEN_JOB=j14b-b2rl-full15m; it fetches each
finished shard and writes .tmp/reviewer/rules/diag/b2rl_full15m__<building>__<weather>.npz) and waits for it. Then,
per set (INNER 29, then S4 first 30): b2rl.py from-diag into data/runs/b2rl/eval/full-rom-s42-15M-<set>.json,
every scenario present, the rows not equal to the rule's (a silent fall-back to the rule would give the rule's costs),
and b2rl.py compare against this host's rule rows (data/runs/b2rl/eval/rule-<set>.json). The result goes into
pipeline/setup/pipeline.log (INFO per set, RESULT at the end, ERROR on a problem: the agent's watcher wakes on those) and
into pipeline.json (gates, passed, done_resume and done_gate: the local lookahead pilot and the local gate are replaced),
and the pipeline's hold file is removed, so the held pipeline goes on: submit under the standing rules if both gates
pass, else it ends. On a problem the hold stays. Host python3, standard library.

python3 -B pipeline/setup/kaggle_gate.py"""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
JOB = "j14b-b2rl-full15m"     # j14 v1 stopped in its precheck (INNER tasks had the purpose "validate")
NAME = "b2rl_full15m"
RULES = ROOT / ".tmp/reviewer/rules"
STATE = ROOT / "pipeline/setup/pipeline.json"
LOG = ROOT / "pipeline/setup/pipeline.log"
HOLD = ROOT / "pipeline/setup/pipeline.hold"
EVAL = ROOT / "data/runs/b2rl/eval"
SINCE = "2026-10-28 09:00"
SETS = {"inner": 29, "s4_first": 30}
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           PATH=f"{ROOT}/tools/bin:" + os.environ.get("PATH", ""), KAGGLE_CONFIG_DIR=str(ROOT / ".secrets/kaggle"))
ENV.pop("SB_FASTSIM", None)


def log(tag, text):
    with LOG.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(time.strftime("%m-%d %H:%M:%S ") + f"{tag} {text}\n")


def mark(**values):
    state = json.loads(STATE.read_text(encoding="utf-8"))
    state.update(values)
    temporary = STATE.with_name(STATE.name + ".partial")
    temporary.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, STATE)


def b2rl(*args):
    result = subprocess.run(["python3", "-B", "scripts/b2rl.py", *args], cwd=ROOT, env=ENV, capture_output=True,
                            text=True)
    if result.returncode:
        raise RuntimeError(f"b2rl.py {args[0]} failed: {(result.stdout + result.stderr).strip()[-300:]}")
    return result.stdout


def collector_lines():
    lines = (RULES / "autorun.log").read_text(encoding="utf-8").splitlines()
    start = max((i for i, line in enumerate(lines) if f" K collecting {JOB} " in line), default=None)
    return [] if start is None else lines[start:]


def main():
    env = dict(ENV, KAGGLE_SCREEN_JOB=JOB, AUTORUN_PAUSE_AT="2026-10-28 22:00")
    collector = subprocess.Popen(["python3", "-B", str(RULES / "kaggle_screen.py")], cwd=ROOT, env=env,
                                 stdout=open(RULES / f"kaggle-screen-{JOB}.out", "a", encoding="utf-8"),
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    log("INFO", f"Kaggle gate: {JOB} pushed (5 CPU kernels, INNER 29 + S4 first 30), collector pid {collector.pid}")
    code = collector.wait()
    lines = collector_lines()
    if code or not any(" K all shards in" in line for line in lines):
        log("ERROR", f"Kaggle gate: the collector ended without every shard (exit {code}): "
                     + " | ".join(line[15:] for line in lines[-3:]))
        return
    try:
        rule_rows = {}
        results = {}
        for subset, n in SETS.items():
            out = EVAL / f"full-rom-s42-15M-{subset}.json"
            b2rl("from-diag", "--name", NAME, "--set", subset, "--since", SINCE, "--out", str(out.relative_to(ROOT)))
            record = json.loads(out.read_text(encoding="utf-8"))
            if record.get("missing") or len(record["rows"]) != n:
                raise RuntimeError(f"{subset}: {len(record['rows'])} of {n} rows; missing {record.get('missing')}")
            baseline = EVAL / f"rule-{subset}.json"
            rule_rows = json.loads(baseline.read_text(encoding="utf-8"))["rows"]
            same = sum(abs(record["rows"][s]["cost"] - rule_rows[s]["cost"]) < 1e-12 for s in record["rows"])
            if same == n:
                raise RuntimeError(f"{subset}: every cost equals the rule's (the actor did not act)")
            result = json.loads(b2rl("compare", "--policy", str(out.relative_to(ROOT)), "--baseline",
                                     str(baseline.relative_to(ROOT))))
            results[subset] = {k: result.get(k) for k in ("n", "mean_diff", "se", "better", "worse", "rel", "gate",
                                                          "policy_mean", "baseline_mean")}
            log("INFO", f"gate {subset} (Kaggle): " + json.dumps(results[subset]))
            if not result.get("gate"):
                break
    except Exception as exc:  # noqa: BLE001 - every surprise goes to the agent
        log("ERROR", f"Kaggle gate: {exc}")
        return
    passed = all(results.get(s, {}).get("gate") for s in SETS)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    mark(gates=results, passed=passed, gate_source=f"Kaggle {JOB}", done_resume=now + " (replaced: agent)",
         done_gate=now + " (Kaggle)")
    if passed:
        log("RESULT", "the 15 M-step actor passed both gates on Kaggle (INNER 29 and S4 first 30); the pipeline goes on "
                      "to submit under the rules")
    else:
        failed = next(s for s in SETS if not results.get(s, {}).get("gate"))
        r = results[failed]
        log("RESULT", f"the 15 M-step actor did not pass the {failed} gate on Kaggle (mean difference "
                      f"{r['mean_diff']:+.6f}, {r['better']} of {r['n']} better); nothing is submitted")
    HOLD.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
