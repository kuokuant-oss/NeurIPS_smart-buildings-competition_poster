"""One B2-RL run instead of three (user 2026-10-27 22:0x: "合併成一組，用全部 6 個模擬程序重新開始訓練，並把明早的評估
改成只評這一組"). Host python3, standard library. State pipeline/setup/rl_single.json, log pipeline/setup/rl_single.log; a
finished step is skipped on a restart.

1. STOP files for the three seeds (42, 1337, 2026; 2 environments each since 21:03); wait until each has saved its
   current update and exited. Their folders stay as they are.
2. Seed 42 with 6 environments in data/runs/b2rl/s42-e6, total decisions 40,000: what this host makes by 05:00 (about
   1.6 decisions/s for 6.8 h; runbook §4.4 lets the hardware set the length), so the learning-rate decay ends near
   the deadline. If the trainer ends early without reaching that, it is restarted once (it resumes its checkpoint).
3. At 2026-10-28 05:00, or when the run has ended by itself: STOP, export, INNER 29, gate against night.py's rule
   baseline; S4 first 30 only when the INNER gate passes (runbook §7 needs both). Evaluation workers follow the free
   memory (about 1.7 GB each, 2 to 8). Nothing is submitted here (runbook §8 follows the gates).
night.py's three-seed chain is switched off through its state key "rl", written once here."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "pipeline/setup/rl_single.json"
NOTE = ROOT / "pipeline/setup/rl_single.log"
NIGHT_STATE = ROOT / "pipeline/setup/night.json"
NIGHT_LOG = ROOT / "pipeline/setup/night.log"
RUNS = ROOT / "data/runs/b2rl"
EVAL = RUNS / "eval"
OLD_SEEDS = (42, 1337, 2026)
NAME = "s42-e6"
RUN = RUNS / NAME
DEADLINE = "2026-10-28 05:00"
TOTAL = 40_000
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def note(text):
    with NOTE.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(time.strftime("%m-%d %H:%M:%S ") + text + "\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path, data):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def mark(key, value=True):
    state = read_json(STATE)
    state[key] = value
    write_json(STATE, state)


def pids(pattern):
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


# anchored at the interpreter so that a shell whose command line merely contains the words does not count
TRAINER = r"^[^ ]*python[^ ]* -B [^ ]*scripts/b2rl\.py train"


def new_run_alive():
    return bool(pids(TRAINER + f" .*--out data/runs/b2rl/{NAME}"))


def last_iteration(folder):
    path = folder / "iterations.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    return json.loads(lines[-1]) if lines else {}


def available_mb():
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def run(args, label):
    out = ROOT / "pipeline/setup/rl_single.out"
    with out.open("a", encoding="utf-8") as stream:
        stream.write(f"\n=== {time.strftime('%m-%d %H:%M:%S')} {' '.join(map(str, args))}\n")
        stream.flush()
        code = subprocess.run([str(a) for a in args], cwd=ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL).returncode
    tail = out.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-2:]
    note(f"{label}: exit {code}; {' | '.join(tail)[-300:]}")
    return code


def b2rl(*args):
    return [sys.executable, "-B", str(ROOT / "scripts/b2rl.py"), *args]


def stop_old_seeds():
    for seed in OLD_SEEDS:
        (RUNS / f"s{seed}" / "STOP").touch()
    note("STOP written for seeds 42, 1337, 2026 (each ends after its current update)")
    while pids(TRAINER):
        time.sleep(20)
    done = {f"s{s}": {k: last_iteration(RUNS / f"s{s}").get(k) for k in ("update", "decisions")} for s in OLD_SEEDS}
    note(f"the three seeds have stopped: {json.dumps(done)}")
    mark("old_seeds_stopped", done)


def start_run():
    log = open(RUNS / f"{NAME}.log", "a", encoding="utf-8")
    process = subprocess.Popen(["nice", "-n", "10", sys.executable, "-B", str(ROOT / "scripts/b2rl.py"), "train",
                                "--seed", "42", "--envs", "6", "--out", f"data/runs/b2rl/{NAME}",
                                "--total-decisions", str(TOTAL)],
                               cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               start_new_session=True)
    note(f"seed 42 with 6 environments started (pid {process.pid}, out data/runs/b2rl/{NAME}, {TOTAL} decisions; "
         f"{available_mb()} MB available)")
    return process


def switch_off_night_chain():
    night = read_json(NIGHT_STATE)
    if not night.get("rl"):
        night["rl"] = f"handled by pipeline/setup/rl_single.py (one run {NAME}; user 2026-10-27)"
        write_json(NIGHT_STATE, night)
    note("night.py's three-seed chain switched off (night.json rl)")
    mark("night_chain_off")


def rule_baseline():
    """True when night.py has the rule baseline, None when its chain failed, False while it runs."""
    if read_json(NIGHT_STATE).get("rule_baseline"):
        return True
    lines = NIGHT_LOG.read_text(encoding="utf-8").splitlines() if NIGHT_LOG.is_file() else []
    failed = any(("rule baseline inner: exit" in line or "rule baseline s4_first: exit" in line)
                 and ": exit 0;" not in line for line in lines) or any("rule_baseline failed" in line for line in lines)
    return None if failed else False


def gate(policy_json, subset):
    while rule_baseline() is False:
        time.sleep(60)
    if rule_baseline() is None:
        note(f"gate {subset}: no rule baseline (night.py's chain failed); left to the agent")
        return {"error": "no rule baseline"}
    baseline = (EVAL / f"rule-{subset}.json").relative_to(ROOT)
    out = subprocess.run(b2rl("compare", "--policy", policy_json, "--baseline", baseline), cwd=ROOT, env=ENV,
                         capture_output=True, text=True)
    try:
        result = json.loads(out.stdout)
    except ValueError:
        result = {"error": out.stdout[-300:] + out.stderr[-300:]}
    note(f"gate {subset}: " + json.dumps({k: result.get(k) for k in ("n", "mean_diff", "se", "better", "worse", "rel",
                                                                       "gate")}))
    return result


def evaluate(subset):
    workers = max(2, min(8, available_mb() // 1700))
    out = (EVAL / f"{NAME}-{subset}.json").relative_to(ROOT)
    code = run(b2rl("eval", "--policy", f"data/runs/b2rl/{NAME}/model.json", "--set", subset, "--workers", str(workers),
                    "--out", out), f"eval {subset} ({workers} workers)")
    return out if code == 0 else None


def finish():
    (RUN / "STOP").touch()
    note("deadline (or the run ended): STOP written")
    for _ in range(240):
        if not new_run_alive():
            break
        time.sleep(30)
    last = last_iteration(RUN)
    note(f"run stopped at update {last.get('update')}, {last.get('decisions')} decisions")
    if not (RUN / "latest.json").is_file():
        mark("result", "no checkpoint")
        return
    run(b2rl("export", "--checkpoint", f"data/runs/b2rl/{NAME}", "--out", f"data/runs/b2rl/{NAME}/model.json"),
        "export")
    results = {"update": last.get("update"), "decisions": last.get("decisions")}
    inner = evaluate("inner")
    if inner is None:
        mark("result", {**results, "inner": "eval failed"})
        return
    results["inner"] = gate(inner, "inner")
    if results["inner"].get("gate"):
        s4 = evaluate("s4_first")
        results["s4_first"] = gate(s4, "s4_first") if s4 else "eval failed"
    else:
        note("the INNER gate did not pass; S4 first 30 skipped (both are needed)")
    mark("result", results)
    note("single run plan finished")


def main():
    note(f"single run plan: {NAME}, deadline {DEADLINE}")
    state = read_json(STATE)
    if not state.get("old_seeds_stopped"):
        stop_old_seeds()
    if not read_json(STATE).get("started"):
        start_run()
        mark("started", time.strftime("%Y-%m-%d %H:%M:%S"))
    if not read_json(STATE).get("night_chain_off"):
        switch_off_night_chain()
    deadline = time.mktime(time.strptime(DEADLINE, "%Y-%m-%d %H:%M"))
    restarted = read_json(STATE).get("restarted", 0)
    while time.time() < deadline and not read_json(STATE).get("result"):
        time.sleep(60)
        if new_run_alive():
            continue
        time.sleep(30)                         # a trainer that is just starting may not be visible yet
        if new_run_alive():
            continue
        decisions = last_iteration(RUN).get("decisions") or 0
        if decisions >= TOTAL or restarted >= 1:
            note(f"the run ended before the deadline ({decisions} decisions)")
            break
        restarted += 1
        mark("restarted", restarted)
        note(f"the run ended early at {decisions} decisions; restarted once (it resumes its checkpoint)")
        start_run()
    if not read_json(STATE).get("result"):
        finish()


if __name__ == "__main__":
    main()
