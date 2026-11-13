"""Automated pipeline of the full B2-RL run (user 2026-10-28 00:5x: "把這整套變成一個pipeline，和自動化程序，完成後你
只負責每小時檢查以及結果出來或是報錯時，出來解決"). Host python3, standard library.

State pipeline/setup/pipeline.json (finished stages are skipped on a restart); log pipeline/setup/pipeline.log, one line per
step tagged INFO, EVENT, RESULT or ERROR: the agent's watcher (pipeline/setup/watch.py) wakes the agent on the last three.
A stage that fails logs ERROR and the pipeline stops (exit 1) for the agent; started again, it continues there.

Stages
  wait    until the fast-sim work writes data/runs/fastsim/READY.json with validated and verdicts_agree true and its
          TRAIN-pool cache complete ("validated": false stops the pipeline with its summary).
  free    the full run supersedes tonight's pilot: stop pipeline/setup/rl_single.py with its evaluation, if one runs, and
          the pilot trainer of data/runs/b2rl/s42-e6 (SIGTERM: it saves its update), and the lookahead pilot (its
          decisions replay on resume). Their folders stay as they are.
  train   b2rl.py train --seed 42 --envs N --out data/runs/b2rl/full-rom-s42 --total-decisions 1250000 --fast <mode>
          (runbook §4.2 settings unchanged; 15 M simulator steps, Zhang 2018's amount); N from the free memory and
          READY.json's worker memory. Up to 3 restarts from the checkpoint; a STOP file in the run folder ends it.
  resume  the lookahead pilot again (2 workers).
  gate    export; INNER 29, then S4 first 30, on the official simulator; runbook §7 gates against the rule baseline
          of this host (data/runs/b2rl/eval/rule-*.json). A failed gate: RESULT, nothing is submitted.
  submit  runbook §8 under the user's standing rules (until 2026-11-13 00:00 Taipei; Kaggle allows 5 a UTC day, 4 of
          them automatic, 1 left for the Mac): build with the model embedded, local notebook test, private kernel,
          Kaggle rows within 1e-6 relative of the local ones, at most 3 submissions already used today (UTC; else wait
          for the next UTC day), submit once, leaderboard-log row. RESULT.
  score   once an hour (scoring takes about 8 hours) until the score is in; it goes into the leaderboard-log row.
"""
import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "pipeline/setup/pipeline.json"
LOG = ROOT / "pipeline/setup/pipeline.log"
OUT = ROOT / "pipeline/setup/pipeline.out"
# 2026-10-28 08:3x user: check on 5 Kaggle CPU kernels and decide on more training first; while this file exists the
# pipeline waits before every stage after train (the trainer itself goes on)
HOLD = ROOT / "pipeline/setup/pipeline.hold"
READY = ROOT / "data/runs/fastsim/READY.json"
RUNS = ROOT / "data/runs/b2rl"
EVAL = RUNS / "eval"
NAME = "full-rom-s42"
RUN = RUNS / NAME
TOTAL = 1_250_000
RESTARTS = 3
PILOT = "s42-e6"
SLUG = f"sb-b2rl-{NAME}"
SUBMISSION = ROOT / f"data/submissions/b2rl-{NAME}"
COMPETITION = "neurips-2026-smart-buildings"
USER = "antonktt"
AUTHORIZED_UNTIL = datetime.datetime(2026, 11, 13, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
TRAINER = r"^[^ ]*python[^ ]* -B [^ ]*scripts/b2rl\.py train"
LAUNCH = [sys.executable, "-B", str(ROOT / "pipeline/launch/launch.py")]
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           PATH=f"{ROOT}/tools/bin:" + os.environ.get("PATH", ""), KAGGLE_CONFIG_DIR=str(ROOT / ".secrets/kaggle"))
ENV.pop("SB_FASTSIM", None)            # only the trainer uses the fast simulator (its own --fast); gates are official
PILOT_LOOP = ('echo "restart $(date "+%F %T") (2 workers, after the full run)" >> data/runs/oracle/pilot.log; '
              'for s in building_18662/denver_1 building_17214/singapore_5 building_10419/atlanta_3; do '
              'n=$(basename $s)-w1; nice -n 15 python3 -B scripts/oracle_rollout.py run --scenario train/$s '
              '--rule a1_cold3 --hours 168 --horizon 8 --workers 2 --out data/runs/oracle/$n >> data/runs/oracle/$n.log '
              '2>&1; echo "$(date "+%F %T") $n rc=$?" >> data/runs/oracle/pilot.log; done; '
              'echo "end $(date "+%F %T")" >> data/runs/oracle/pilot.log')


class Stop(Exception):
    """A stage failed: logged as ERROR, the pipeline exits for the agent."""


def log(tag, text):
    with LOG.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(time.strftime("%m-%d %H:%M:%S ") + f"{tag} {text}\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def mark(key, value=True):
    state = read_json(STATE)
    state[key] = value
    temporary = STATE.with_name(STATE.name + ".partial")
    temporary.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, STATE)


def pids(pattern):
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


def available_mb():
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def run(args, label, stdout_only=False):
    """A command in the project root; its output appended to pipeline.out; (exit code, its output: stdout and
    stderr, or stdout alone)."""
    with OUT.open("a", encoding="utf-8") as stream:
        stream.write(f"\n=== {time.strftime('%m-%d %H:%M:%S')} {label}: {' '.join(map(str, args))}\n")
        stream.flush()
        result = subprocess.run([str(a) for a in args], cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True)
        stream.write(result.stdout + result.stderr)
    return result.returncode, result.stdout if stdout_only else result.stdout + result.stderr


def b2rl(*args):
    return [sys.executable, "-B", str(ROOT / "scripts/b2rl.py"), *args]


def kaggle(*args):
    code, text = run(["kaggle", *args], "kaggle " + args[0] + " " + args[1])
    if code:
        raise Stop(f"kaggle {' '.join(args[:2])} failed: {text.strip()[-300:]}")
    return text


def last_iteration(folder=RUN):
    path = folder / "iterations.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    return json.loads(lines[-1]) if lines else {}


# --- wait -------------------------------------------------------------------------------------------------------------
def stage_wait():
    log("INFO", "waiting for data/runs/fastsim/READY.json (validated, verdicts agree, TRAIN cache complete)")
    while True:
        ready = read_json(READY) if READY.is_file() else None
        if ready is not None:
            if ready.get("validated") is False:
                raise Stop(f"the fast simulator is not validated: {ready.get('summary')}")
            built, total = (ready.get("cache") or {}).get("train") or (0, 1)
            if ready.get("validated") and ready.get("verdicts_agree") and built >= total:
                mode = ready.get("mode") or "rom"
                mark("fast", {"mode": mode, "worker_memory_mb": ready.get("worker_memory_mb"),
                              "worst_cost_rel_err": ready.get("worst_cost_rel_err"), "summary": ready.get("summary")})
                log("EVENT", f"fast simulator ready ({mode}, worst cost error {ready.get('worst_cost_rel_err')}, "
                             f"TRAIN cache {built}/{total}): {ready.get('summary')}")
                return
        time.sleep(60)


# --- free -------------------------------------------------------------------------------------------------------------
def stage_free():
    single = pids(r"^python3 -B \pipeline/setup/rl_single\.py$")
    for pid in single:   # its own session (setsid): the group holds it and any evaluation it started
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    if single:
        with (ROOT / "pipeline/setup/rl_single.log").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(time.strftime("%m-%d %H:%M:%S ") + "stopped by pipeline/setup/pipeline.py: the full run "
                         "supersedes the pilot\n")
    trainer = pids(TRAINER + f" .*--out data/runs/b2rl/{PILOT}( |$)")
    for pid in trainer:
        os.kill(pid, signal.SIGTERM)            # saves the update it is in, then exits
    deadline = time.time() + 20 * 60
    while pids(TRAINER + f" .*--out data/runs/b2rl/{PILOT}( |$)") and time.time() < deadline:
        time.sleep(20)
    for pid in pids(TRAINER + f" .*--out data/runs/b2rl/{PILOT}( |$)"):
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    pilot = pids(r"^bash -c echo \"restart .*oracle_rollout\.py run")
    for pid in pilot:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    if pilot:
        with (ROOT / "data/runs/oracle/pilot.log").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"stopped {time.strftime('%F %T')} by the pipeline (the full run needs the memory); "
                         "resumes afterwards\n")
    time.sleep(30)
    last = last_iteration(RUNS / PILOT)
    log("INFO", f"freed: pilot trainer {PILOT} stopped at update {last.get('update')} ({last.get('decisions')} "
                f"decisions), rl_single {'stopped' if single else 'not running'}, lookahead pilot "
                f"{'stopped' if pilot else 'not running'}; {available_mb()} MB available")


# --- train ------------------------------------------------------------------------------------------------------------
def environments():
    fast = read_json(STATE).get("fast") or {}
    per_worker = int(fast.get("worker_memory_mb") or 1600) + 200
    return max(4, min(16, (available_mb() - 3000) // per_worker))


def start_trainer(envs, mode):
    log_file = open(RUNS / f"{NAME}.log", "a", encoding="utf-8")
    process = subprocess.Popen(["nice", "-n", "5", *b2rl("train", "--seed", "42", "--envs", str(envs), "--out",
                                                        f"data/runs/b2rl/{NAME}", "--total-decisions", str(TOTAL),
                                                        "--fast", mode)],
                               cwd=ROOT, env=ENV, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               start_new_session=True)
    return process.pid


def alive():
    return pids(TRAINER + f" .*--out data/runs/b2rl/{NAME}( |$)")


def stage_train():
    state = read_json(STATE)
    mode = state["fast"]["mode"]
    envs = state.get("envs") or environments()
    mark("envs", envs)
    if not alive():
        pid = start_trainer(envs, mode)
        log("EVENT", f"full run started: {NAME}, {envs} environments, {TOTAL} decisions (15 M steps), fast simulator "
                     f"{mode}, pid {pid}; {available_mb()} MB available")
    restarts = state.get("restarts", 0)
    hour = time.time()
    while True:
        time.sleep(120)
        if time.time() - hour >= 3600:
            hour = time.time()
            last = last_iteration()
            log("INFO", f"training: update {last.get('update')}, {last.get('decisions')} of {TOTAL} decisions, "
                        f"{last.get('decisions_per_second', 0):.1f} decisions/s, 12-step reward "
                        f"{last.get('reward_12_mean')}")
        if alive():
            continue
        time.sleep(30)
        if alive():
            continue
        decisions = last_iteration().get("decisions") or 0
        if decisions >= TOTAL or (RUN / "STOP").exists():
            log("EVENT", f"training finished: {decisions} decisions")
            return
        if restarts >= RESTARTS:
            raise Stop(f"the trainer ended early {restarts + 1} times (last at {decisions} decisions); see "
                       f"data/runs/b2rl/{NAME}.log")
        restarts += 1
        mark("restarts", restarts)
        log("ERROR", f"the trainer ended early at {decisions} decisions (see data/runs/b2rl/{NAME}.log); restarted "
                     f"from its checkpoint ({restarts} of {RESTARTS}), the pipeline goes on")
        start_trainer(envs, mode)


# --- resume -----------------------------------------------------------------------------------------------------------
def stage_resume():
    if any(line.startswith("end ") for line in (ROOT / "data/runs/oracle/pilot.log").read_text(
            encoding="utf-8").splitlines()):
        log("INFO", "lookahead pilot already finished")
        return
    process = subprocess.Popen(["bash", "-c", PILOT_LOOP], cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    log("INFO", f"lookahead pilot resumed with 2 workers (process group {process.pid})")


# --- gate -------------------------------------------------------------------------------------------------------------
def gate_one(subset):
    workers = max(2, min(8, available_mb() // 1700))
    rows = (EVAL / f"{NAME}-{subset}.json").relative_to(ROOT)
    code, text = run(b2rl("eval", "--policy", f"data/runs/b2rl/{NAME}/model.json", "--set", subset, "--workers",
                          str(workers), "--out", str(rows)), f"eval {subset}")
    if code:
        raise Stop(f"evaluation {subset} failed: {text.strip()[-300:]}")
    code, text = run(b2rl("compare", "--policy", str(rows), "--baseline",
                          str((EVAL / f"rule-{subset}.json").relative_to(ROOT))), f"compare {subset}", stdout_only=True)
    try:
        result = json.loads(text)
    except ValueError as exc:
        raise Stop(f"compare {subset}: unreadable output {text.strip()[-300:]}") from exc
    return result


def stage_gate():
    if not (RUN / "latest.json").is_file():
        raise Stop("the run has no checkpoint")
    code, text = run(b2rl("export", "--checkpoint", f"data/runs/b2rl/{NAME}", "--out",
                          f"data/runs/b2rl/{NAME}/model.json"), "export")
    if code:
        raise Stop(f"export failed: {text.strip()[-300:]}")
    results = {}
    for subset in ("inner", "s4_first"):
        result = gate_one(subset)
        results[subset] = {k: result.get(k) for k in ("n", "mean_diff", "se", "better", "worse", "rel", "gate",
                                                      "policy_mean", "baseline_mean", "parts")}
        log("INFO", f"gate {subset}: " + json.dumps(results[subset]))
        mark("gates", results)
        if not result.get("gate"):
            log("RESULT", f"the trained policy did not pass the {subset} gate (mean difference "
                          f"{result.get('mean_diff')}, {result.get('better')} of {result.get('n')} better); "
                          "nothing is submitted")
            return False
    log("RESULT", "the trained policy passed both gates (INNER 29 and S4 first 30); submitting under the rules")
    return True


# --- submit -----------------------------------------------------------------------------------------------------------
def used_today():
    text = kaggle("competitions", "submissions", COMPETITION, "--csv")
    lines = text.splitlines()
    header = next((i for i, x in enumerate(lines) if x.startswith("ref,")), None)
    if header is None:
        raise Stop(f"submissions list unreadable: {text.strip()[-200:]}")
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    rows = lines[header + 1:]
    return sum(1 for row in rows if len(row.split(",")) > 2 and row.split(",")[2].startswith(today)), rows


def stage_submit():
    state = read_json(STATE)
    local = state.get("local")
    if not local:
        code, text = run([*LAUNCH, str(ROOT / "scripts/build_submission.py"), "--policy-class", "G36PolicyB2RL",
                          "--slug", SLUG, "--title", f"SB B2RL {NAME}", "--username", USER, "--out",
                          SUBMISSION.relative_to(ROOT).as_posix(), "--embed", f"data/runs/b2rl/{NAME}/model.json"],
                         "build")
        if code:
            raise Stop(f"build failed: {text.strip()[-300:]}")
        notebook = SUBMISSION / f"{SLUG}.ipynb"
        code, text = run([*LAUNCH, str(ROOT / "scripts/test_notebook.py"), notebook.relative_to(ROOT).as_posix()],
                         "local notebook test")
        report = None
        for line in text.splitlines():
            if line.startswith("Output:"):
                report = read_json(ROOT / line.split(":", 1)[1].strip() / "run.json")
        if code or not report or not report.get("rows"):
            raise Stop(f"local notebook run failed: {text.strip()[-300:]}")
        local = {"rows": report["rows"], "seconds": report["elapsed_seconds"]}
        mark("local", local)
        log("INFO", f"built {SLUG}; local rows {report['rows']} in {report['elapsed_seconds']:.0f} s")
    ref = f"{USER}/{SLUG}"
    if not state.get("version"):
        text = kaggle("kernels", "push", "-p", SUBMISSION.relative_to(ROOT).as_posix())
        match = re.search(r"version\s+(\d+)", text, re.IGNORECASE)
        if not match:
            raise Stop(f"kernel version not in the push output: {text.strip()[-200:]}")
        mark("version", match.group(1))
        log("INFO", f"pushed {ref} version {match.group(1)} (private)")
    version = read_json(STATE)["version"]
    if read_json(STATE).get("parity") is None:
        deadline = time.time() + 3 * 3600
        while True:
            code, status = run(["kaggle", "kernels", "status", ref], "kernel status")
            if "COMPLETE" in status.upper():
                break
            if "ERROR" in status.upper() or "CANCEL" in status.upper():
                raise Stop(f"kernel {ref}: {status.strip()[-200:]}")
            if time.time() > deadline:
                raise Stop(f"kernel {ref} not complete after 3 hours")
            time.sleep(300)
        fetch = ROOT / "data/runs" / f"kaggle-{SLUG}"
        fetch.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", ref, "-p", fetch.relative_to(ROOT).as_posix())
        code, text = run([*LAUNCH, "-c", "import json,sys,polars as pl; print(json.dumps(pl.read_parquet(sys.argv[1])"
                          ".to_dicts()))", (fetch / "submission.parquet").as_posix()], "read Kaggle rows")
        kaggle_rows = json.loads(text.strip().splitlines()[-1]) if code == 0 else None
        rows = {r["row_id"]: r["reward"] for r in local["rows"]}
        if not kaggle_rows or {r["row_id"] for r in kaggle_rows} != set(rows):
            raise Stop(f"Kaggle output rows missing or different: {text.strip()[-200:]}")
        worst = max(abs(r["reward"] - rows[r["row_id"]]) / max(abs(rows[r["row_id"]]), 1e-12) for r in kaggle_rows)
        mark("parity", worst)
        if worst > 1e-6:
            raise Stop(f"Kaggle and local rewards differ (relative {worst:.2e}); nothing submitted")
        log("INFO", f"Kaggle rows match the local ones (relative {worst:.1e})")
    if read_json(STATE).get("submitted"):
        return
    while True:
        if datetime.datetime.now(datetime.timezone.utc) >= AUTHORIZED_UNTIL:
            raise Stop("the automatic-submission authorization (until 2026-11-13 00:00 Taipei) has ended")
        count, _ = used_today()
        if count <= 3:
            break
        log("INFO", f"{count} submissions already today (UTC); waiting for the next UTC day (one stays for the Mac)")
        now = datetime.datetime.now(datetime.timezone.utc)
        tomorrow = (now + datetime.timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
        time.sleep(max(60, (tomorrow - now).total_seconds()))
    gates = read_json(STATE)["gates"]
    message = (f"B2-RL {NAME}: PPO on the hourly rule parameters of G36PolicyB2R2, 15 M steps on the fast simulator; "
               f"INNER 29 {100 * gates['inner']['rel']:+.1f}% and S4 first 30 {100 * gates['s4_first']['rel']:+.1f}% vs "
               "the rule (official simulator)")
    kaggle("competitions", "submit", COMPETITION, "-k", ref, "-v", str(version), "-f", "submission.parquet", "-m",
           message)
    _, rows = used_today()
    submission_ref = rows[0].split(",")[0] if rows else "?"
    mark("submitted", {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "ref": submission_ref})
    i, s = gates["inner"], gates["s4_first"]
    row = (f"| {time.strftime('%Y-%m-%d %H:%M:%S')} | {submission_ref} | B2-RL {NAME} G36PolicyB2RL（G36PolicyB2R2 上每小時"
           f"調 5 個規則參數的 PPO，1500 萬步，加速模擬器 {read_json(STATE)['fast']['mode']}；docs/plan/b2rl-batch.md） | "
           f"Notebook {ref} v{version} | INNER {i['n']} 場與規則基準配對差 {i['mean_diff']:+.6f}（{100 * i['rel']:+.1f}%，"
           f"{i['better']}/{i['n']} 較好）；S4 前 {s['n']} 場 {s['mean_diff']:+.6f}（{100 * s['rel']:+.1f}%，SE "
           f"{s['se']:.6f}，{s['better']} 較好、{s['worse']} 較差）；官方模擬器 | 無 | （待出分） | 本機公開場景 "
           f"{local['seconds']:.0f} 秒 | 使用者 2026-10-27「授權自動提交」「照規則走」、2026-10-28「把這整套變成一個"
           f"pipeline」 | Kaggle 與本機逐場相對差 {read_json(STATE)['parity']:.1e} |\n")
    with (ROOT / "docs/experiments/leaderboard-log.md").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(row)
    log("RESULT", f"submitted {ref} v{version} (ref {submission_ref}); score in about 8 hours")


# --- score ------------------------------------------------------------------------------------------------------------
def stage_score():
    import csv
    ref = read_json(STATE)["submitted"]["ref"]
    for _ in range(14):
        lines = kaggle("competitions", "submissions", COMPETITION, "--csv").splitlines()
        line = next((r for r in lines if r.startswith(f"{ref},")), "")
        fields = next(csv.reader([line])) if line else []
        status = fields[4] if len(fields) > 4 else ""      # ref, fileName, date, description, status, public, private
        if "COMPLETE" in status.upper() or "ERROR" in status.upper():
            public = fields[5] if len(fields) > 5 else ""
            path = ROOT / "docs/experiments/leaderboard-log.md"
            text = path.read_text(encoding="utf-8")
            start = text.find(f"| {ref} |")
            if start >= 0 and public:
                end = text.find("\n", start)
                text = text[:start] + text[start:end].replace("（待出分）", public, 1) + text[end:]
                with path.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(text)
            mark("score", {"status": status, "public": public})
            log("RESULT", f"submission {ref} scored: {status}, public {public or '?'}")
            return
        time.sleep(3600)
    log("ERROR", f"submission {ref} still not scored after 14 hours")


STAGES = [("wait", stage_wait), ("free", stage_free), ("train", stage_train), ("resume", stage_resume),
          ("gate", stage_gate), ("submit", stage_submit), ("score", stage_score)]


def main():
    log("INFO", f"pipeline started (pid {os.getpid()})")
    for name, stage in STAGES:
        state = read_json(STATE)
        if state.get(f"done_{name}"):
            continue
        if name in ("submit", "score") and state.get("passed") is False:
            break
        if name not in ("wait", "free", "train") and HOLD.exists():
            log("EVENT", f"held before the {name} stage ({HOLD.relative_to(ROOT)} exists; removing it goes on)")
            while HOLD.exists():
                time.sleep(30)
            log("INFO", f"hold released; the {name} stage goes on")
            if read_json(STATE).get(f"done_{name}"):
                continue
        try:
            outcome = stage()
        except Stop as exc:
            log("ERROR", f"{name}: {exc}")
            raise SystemExit(1)
        except Exception as exc:  # noqa: BLE001 - every surprise goes to the agent
            log("ERROR", f"{name}: {exc!r}")
            raise SystemExit(1)
        if name == "gate":
            mark("passed", bool(outcome))
        mark(f"done_{name}", time.strftime("%Y-%m-%d %H:%M:%S"))
    log("RESULT", "pipeline finished")


if __name__ == "__main__":
    main()
