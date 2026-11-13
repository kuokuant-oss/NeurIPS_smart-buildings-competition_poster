"""Windows host night plan, 2026-10-27/28 (runbook §2.5 R7 via Kaggle per §2.7, §2.6 extraction, §5, §7).

Host python3, standard library. State pipeline/setup/night.json, log pipeline/setup/night.log; a finished step is skipped, so a
restart continues (the evaluations themselves keep their finished rows: scripts/b2rl.py eval resumes).

 1. autorun2 at R5 (private kernel pushed): restart it with AUTORUN_PAUSE_AT 30 minutes ahead so that its R7 (the other
    90 S4 scenarios, about 3 hours of this host) is skipped; never during R6 (a restart there could submit twice).
 2. R6 done (stage R7 or done): define, build and push j13-b2r2-s4rest (5 CPU kernels) for the validated rule and collect
    it (kaggle_screen.py writes diag/<rule>__<building>__<weather>.npz) — the R7 record, on Kaggle.
 3. R4 wrote the final rule: the rule baseline of this host for §7 — this host's diag rows since 14:55 seed
    eval/rule-<set>.json, then b2rl.py eval runs what is missing (2 workers).
 4. The lookahead pilot ended: decision tree (depth 4) -> G36PolicyB2RX on INNER 29 then S4 first 30 (4 workers), gate
    tables against the rule baseline.
 5. RL_DEADLINE: STOP the three seeds, export their last checkpoints, INNER 29 for each (8 workers), the best mean cost on
    S4 first 30, gate tables. Nothing is submitted here (runbook §8: the submission follows the gates)."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / ".tmp/reviewer/rules"
LOG = RULES / "autorun.log"
STATE = ROOT / "pipeline/setup/night.json"
NOTE = ROOT / "pipeline/setup/night.log"
EVAL = ROOT / "data/runs/b2rl/eval"
START = int(sys.argv[1])                      # autorun.log lines before tonight's events
RL_DEADLINE = "2026-10-28 05:00"
SINCE = "2026-10-27 14:55"                     # diag rows of this host start here
SEEDS = (42, 1337, 2026)
PILOT = ["fairbanks_1", "phoenix_5", "denver_1", "singapore_5", "atlanta_3"]
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           PATH=f"{ROOT}/tools/bin:" + os.environ.get("PATH", ""), KAGGLE_CONFIG_DIR=str(ROOT / ".secrets/kaggle"))
LOCK = threading.Lock()


def note(text):
    with LOCK, NOTE.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(time.strftime("%m-%d %H:%M:%S ") + text + "\n")


def load():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}


def mark(key, value=True):
    with LOCK:
        state = load()
        state[key] = value
        temporary = STATE.with_name(STATE.name + ".partial")
        temporary.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")
        os.replace(temporary, STATE)


def new_lines():
    return LOG.read_text(encoding="utf-8").splitlines()[START:]


def rules_state():
    return json.loads((RULES / "autorun2.json").read_text(encoding="utf-8"))


def run(args, label, log_name=None):
    """A command in the project root; output to pipeline/setup/night-<label>.out; returns (code, last lines)."""
    out = ROOT / "pipeline/setup" / f"night-{log_name or label}.out"
    with out.open("a", encoding="utf-8") as stream:
        stream.write(f"\n=== {time.strftime('%m-%d %H:%M:%S')} {' '.join(map(str, args))}\n")
        stream.flush()
        code = subprocess.run([str(a) for a in args], cwd=ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL).returncode
    tail = out.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
    note(f"{label}: exit {code}; {' | '.join(tail)[-300:]}")
    return code, tail


def py(script, *args):
    return [sys.executable, "-B", str(ROOT / script), *args]


def autorun2_pids():
    out = subprocess.run(["pgrep", "-f", r"^python3 -B \.tmp/reviewer/rules/autorun2\.py$"], capture_output=True,
                         text=True).stdout.split()
    return [int(p) for p in out]


def start_autorun2(pause_at):
    env = dict(ENV, AUTORUN_WORKERS="8", AUTORUN_PAUSE_AT=pause_at, AUTORUN2_SUBMIT="1")
    out = open(RULES / "autorun2-win.out", "a", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-B", ".tmp/reviewer/rules/autorun2.py"], cwd=ROOT, env=env,
                               stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    note(f"autorun2 restarted (pid {process.pid}, AUTORUN_PAUSE_AT {pause_at})")


def step_skip_r7(state):
    if state.get("skip_r7") or not any(" R R5: pushed" in line for line in new_lines()):
        return
    s = rules_state()
    if s.get("problem"):
        mark("skip_r7", "autorun2 has a problem; nothing restarted")
        return
    if s["stage"] != "R5":
        if s["stage"] in ("R7", "done"):
            mark("skip_r7", f"too late: autorun2 already at {s['stage']}")
        return                                  # R6 (submitting): wait, never interrupt it
    pids = autorun2_pids()
    for pid in pids:
        os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not autorun2_pids():
            break
        time.sleep(1)
    if autorun2_pids():
        note("autorun2 did not stop; R7 stays local")
        mark("skip_r7", "autorun2 did not stop")
        return
    if rules_state()["stage"] != "R5":         # it moved on while stopping: start it again as it was
        start_autorun2("2026-10-28 11:00")
        mark("skip_r7", "stage changed while stopping; restarted unchanged")
        return
    start_autorun2(time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + 1800)))
    mark("skip_r7", "restarted at R5 with an early pause time (R7 on Kaggle)")


def step_j13(state):
    if state.get("j13"):
        return
    s = rules_state()
    if s.get("problem") or s["stage"] not in ("R7", "done") or not s.get("submitted"):
        return
    name = s["joined"]
    ws = [sys.executable, "-B", str(ROOT / "scripts/workspace.py"), "remote"]
    for args in (["define", "j13-b2r2-s4rest", name, "5"], ["build", "j13-b2r2-s4rest", "--mode", "full"],
                 ["push", "j13-b2r2-s4rest", "--mode", "full", "--shards", "k1", "k2", "k3", "k4", "k5",
                  "--purpose", f"Baseline v2 final rule {name} on the other 90 S4 scenarios (record; autorun2 R7)"]):
        code, _ = run(ws + args, f"j13 {args[0]}", "j13")
        if code:
            mark("j13", f"{args[0]} failed (exit {code})")
            return
    env = dict(ENV, KAGGLE_SCREEN_JOB="j13-b2r2-s4rest", AUTORUN_PAUSE_AT="2026-10-28 11:00")
    subprocess.Popen([sys.executable, "-B", str(RULES / "kaggle_screen.py")], cwd=ROOT, env=env,
                     stdout=open(RULES / "kaggle-screen-j13.out", "a", encoding="utf-8"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    mark("j13", f"pushed for {name}; collector started")


def final_rule_written():
    # logged only after autorun2's "B2R2 OK" check passed (else it stops); the word itself is cut off the line
    return any(" R R4: B2R2_CHANGES" in line and " written; " in line for line in new_lines())


def chain_rule_baseline():
    name = rules_state()["joined"]
    EVAL.mkdir(parents=True, exist_ok=True)
    for subset in ("inner", "s4_first"):
        out = EVAL / f"rule-{subset}.json"
        if not out.is_file():
            run(py("scripts/b2rl.py", "from-diag", "--name", name, "--set", subset, "--since", SINCE, "--as-rule-eval",
                   "--out", out.relative_to(ROOT)), f"rule baseline seed {subset}", "rule-baseline")
        code, _ = run(py("scripts/b2rl.py", "eval", "--policy", "rule", "--set", subset, "--workers", "2", "--out",
                         out.relative_to(ROOT)), f"rule baseline {subset}", "rule-baseline")
        if code:
            return False
    mark("rule_baseline", f"done for {name}")
    return True


def gate(policy_json, baseline_json, label):
    out = subprocess.run(py("scripts/b2rl.py", "compare", "--policy", policy_json, "--baseline", baseline_json), cwd=ROOT,
                         env=ENV, capture_output=True, text=True)
    try:
        result = json.loads(out.stdout)
    except ValueError:
        result = {"error": out.stdout[-300:] + out.stderr[-300:]}
    note(f"gate {label}: " + json.dumps({k: result.get(k) for k in ("n", "mean_diff", "se", "better", "worse", "rel",
                                                                     "gate")}))
    return result


def wait_for(key):
    while not load().get(key):
        time.sleep(60)


def chain_b2rx():
    runs = [f"data/runs/oracle/{n}-w1" for n in PILOT if (ROOT / f"data/runs/oracle/{n}-w1/summary.json").is_file()]
    tree = ROOT / "data/runs/oracle/tree-w1.json"
    if not tree.is_file():
        run(py("scripts/oracle_rollout.py", "tree", "--runs", *runs, "--out", tree.relative_to(ROOT), "--depth", "4"),
            "tree", "b2rx")
    if not tree.is_file():
        mark("b2rx", "tree failed")
        return
    results = {}
    for subset in ("inner", "s4_first"):
        out = EVAL / f"b2rx-{subset}.json"
        code, _ = run(py("scripts/b2rl.py", "eval", "--policy", tree.relative_to(ROOT), "--set", subset, "--workers",
                         "4", "--out", out.relative_to(ROOT)), f"b2rx {subset}", "b2rx")
        if code:
            mark("b2rx", f"eval {subset} failed")
            return
        wait_for("rule_baseline")
        results[subset] = gate(out.relative_to(ROOT), (EVAL / f"rule-{subset}.json").relative_to(ROOT), f"b2rx {subset}")
    mark("b2rx", results)


def trainers_running():
    out = subprocess.run(["pgrep", "-f", "scripts/b2rl.py train"], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


def chain_rl():
    for seed in SEEDS:
        (ROOT / f"data/runs/b2rl/s{seed}/STOP").touch()
    note("RL deadline: STOP written for the three seeds")
    for _ in range(240):
        if not trainers_running():
            break
        time.sleep(30)
    results = {}
    for seed in SEEDS:
        model = ROOT / f"data/runs/b2rl/s{seed}/model.json"
        if not (ROOT / f"data/runs/b2rl/s{seed}/latest.json").is_file():
            results[f"s{seed}"] = "no checkpoint"
            continue
        run(py("scripts/b2rl.py", "export", "--checkpoint", f"data/runs/b2rl/s{seed}", "--out", model.relative_to(ROOT)),
            f"export s{seed}", "rl")
        out = EVAL / f"s{seed}-inner.json"
        code, _ = run(py("scripts/b2rl.py", "eval", "--policy", model.relative_to(ROOT), "--set", "inner", "--workers",
                         "8", "--out", out.relative_to(ROOT)), f"rl s{seed} inner", "rl")
        if code:
            results[f"s{seed}"] = f"inner eval failed ({code})"
            continue
        wait_for("rule_baseline")
        results[f"s{seed}"] = {"inner": gate(out.relative_to(ROOT), (EVAL / "rule-inner.json").relative_to(ROOT),
                                             f"rl s{seed} inner")}
    scored = {k: v for k, v in results.items() if isinstance(v, dict) and "policy_mean" in v["inner"]}
    if scored:
        best = min(scored, key=lambda k: scored[k]["inner"]["policy_mean"])
        out = EVAL / f"{best}-s4_first.json"
        code, _ = run(py("scripts/b2rl.py", "eval", "--policy", f"data/runs/b2rl/{best}/model.json", "--set", "s4_first",
                         "--workers", "8", "--out", out.relative_to(ROOT)), f"rl {best} s4_first", "rl")
        if not code:
            results[best]["s4_first"] = gate(out.relative_to(ROOT), (EVAL / "rule-s4_first.json").relative_to(ROOT),
                                             f"rl {best} s4_first")
        results["best"] = best
    mark("rl", results)


def start_thread(key, target):
    def body():
        try:
            target()
        except Exception as error:  # noqa: BLE001 - recorded, the other steps go on
            note(f"{key} failed: {error!r}")
            mark(key, f"failed: {error!r}")
    thread = threading.Thread(target=body, daemon=True, name=key)
    thread.start()
    return thread


def main():
    note(f"night plan watching autorun.log from line {START + 1}; RL deadline {RL_DEADLINE}")
    threads = {}
    deadline = time.mktime(time.strptime(RL_DEADLINE, "%Y-%m-%d %H:%M"))
    while True:
        state = load()
        try:
            step_skip_r7(state)
            step_j13(load())
        except Exception as error:  # noqa: BLE001
            note(f"step failed: {error!r}")
        if final_rule_written():
            if not state.get("rule_baseline") and "rule_baseline" not in threads:
                threads["rule_baseline"] = start_thread("rule_baseline", chain_rule_baseline)
            pilot_done = any(line.startswith("end ") for line in
                             (ROOT / "data/runs/oracle/pilot.log").read_text(encoding="utf-8").splitlines())
            if pilot_done and not state.get("b2rx") and "b2rx" not in threads:
                threads["b2rx"] = start_thread("b2rx", chain_b2rx)
            if time.time() >= deadline and not state.get("rl") and "rl" not in threads:
                threads["rl"] = start_thread("rl", chain_rl)
        elif any(" STOP " in line for line in new_lines()) and not state.get("stopped"):
            mark("stopped", "autorun2 recorded a problem before R4")
            note("autorun2 stopped before R4; only the Kaggle and pilot parts can follow")
        state = load()
        if all(state.get(k) for k in ("rule_baseline", "b2rx", "rl")) and state.get("j13"):
            note("night plan finished")
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
