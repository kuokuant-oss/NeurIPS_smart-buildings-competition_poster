"""Submit one candidate that passed the runbook §7 gates on the official simulator, under the user's standing rules
(until 2026-11-13 00:00 Taipei; up to 4 automatic submissions a day, 1 kept spare (user 2026-10-28: "每天可以自動提交四次。留一備用"; counted by Kaggle's own day); private sb-
kernels; Kaggle rows within 1e-6 relative of the local ones). The steps of pipeline/setup/pipeline.py stage_submit for any
policy class build_submission.py embeds (G36PolicyB2RL with an actor, G36PolicyB2RX with a candidate tree).
User 2026-10-28 15:0x: the lookahead's extracted rule "如果驗證結果不錯就可以提交然後換新底"; the fixed rule of the
offset check is submitted under the same rules when its gates pass.

Steps (state pipeline/setup/submit-<name>.json, each done once): refuse unless the gates file says both gates passed; build
the notebook with the policy embedded; run it locally on the public scenario; push the private kernel; wait for it;
compare its submission.parquet with the local rows (<= 1e-6 relative); wait for a day with fewer than 4 submissions;
submit once; append the leaderboard-log row (score pending). Log lines go to pipeline/setup/submit.log. Host python3.

python3 -B pipeline/setup/submit_candidate.py <name> <policy class> <embed json> <gates json> <description>"""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
NAME, POLICY_CLASS, EMBED, GATES, DESCRIPTION = sys.argv[1:6]
SLUG = f"sb-{NAME}"
SUBMISSION = ROOT / f"data/submissions/{NAME}"
STATE = ROOT / f"pipeline/setup/submit-{NAME}.json"
LOG = ROOT / "pipeline/setup/submit.log"
COMPETITION = "neurips-2026-smart-buildings"
USER = "antonktt"
AUTHORIZED_UNTIL = datetime.datetime(2026, 11, 13, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
# One-day exceptions: the spare 5th submission may also go out automatically, only for an autotune candidate (a name
# starting "autotune"). User 2026-10-29 19:4x: "我今天沒有什麼其他要提交的，備用那個今天允許自動提交用掉。如果後續自動重調
# 參數有其他新進展可用" (Kaggle's own day 2026-10-29). No other day and no other candidate.
SPARE_ALLOWED = {"2026-10-29": "autotune",
                 # user 2026-11-10 16:5x "不需動用備用額度: 請動用" (Kaggle days 11-10 .. 11-12, gt- candidates)
                 "2026-11-10": "gt-", "2026-11-11": "gt-",
                 # user 2026-11-12 18:3x "依照這提交" (reviewer's pure-rule baseline pr_opt2; the day's 4 were used)
                 "2026-11-12": ("gt-", "pr-opt2")}
LAUNCH = [sys.executable, "-B", str(ROOT / "pipeline/launch/launch.py")]
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           PATH=f"{ROOT}/tools/bin:" + os.environ.get("PATH", ""), KAGGLE_CONFIG_DIR=str(ROOT / ".secrets/kaggle"))
ENV.pop("SB_FASTSIM", None)


def log(tag, text):
    with LOG.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(time.strftime("%m-%d %H:%M:%S ") + f"{tag} {NAME}: {text}\n")
    print(f"{tag} {text}", flush=True)


def state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}


def mark(key, value):
    s = state()
    s[key] = value
    STATE.write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")


def run(args):
    result = subprocess.run([str(a) for a in args], cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True)
    return result.returncode, result.stdout + result.stderr


def kaggle(*args):
    code, text = run(["kaggle", *args])
    if code:
        raise SystemExit(f"kaggle {' '.join(args[:2])} failed: {text.strip()[-300:]}")
    return text


def used_today():
    text = kaggle("competitions", "submissions", COMPETITION, "--csv")
    lines = text.splitlines()
    header = next(i for i, x in enumerate(lines) if x.startswith("ref,"))
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    rows = lines[header + 1:]
    return sum(1 for row in rows if len(row.split(",")) > 2 and row.split(",")[2].startswith(today)), rows


def main():
    mandated = os.environ.get("SUBMIT_USER_MANDATED", "").strip()
    if GATES == "none":
        # user 2026-11-10 16:4x: the rule version (paper rule + levers) and the second merged version with its
        # components must be submitted regardless (paper baselines / ablation); no Kaggle gate file. The reason is logged.
        if not mandated:
            raise SystemExit("no gates file: only with SUBMIT_USER_MANDATED=<the user instruction>")
        log("INFO", f"submitted without Kaggle gates by user instruction: {mandated}")
    else:
        gates = json.loads((ROOT / GATES).read_text(encoding="utf-8"))
        if not (gates["inner"]["gate"] and gates["s4_first"]["gate"]):
            raise SystemExit(f"{GATES}: not both gates passed; nothing submitted")
    if not state().get("local"):
        code, text = run([*LAUNCH, ROOT / "scripts/build_submission.py", "--policy-class", POLICY_CLASS, "--slug", SLUG,
                          "--title", f"SB {NAME}", "--username", USER, "--out", SUBMISSION.relative_to(ROOT).as_posix(),
                          "--embed", EMBED])
        if code:
            raise SystemExit(f"build failed: {text.strip()[-300:]}")
        metadata = json.loads((SUBMISSION / "kernel-metadata.json").read_text(encoding="utf-8"))
        if not metadata.get("is_private") or not metadata["id"].split("/")[1].startswith("sb-"):
            raise SystemExit(f"kernel metadata is not a private sb- kernel: {metadata}")
        code, text = run([*LAUNCH, ROOT / "scripts/test_notebook.py", (SUBMISSION / f"{SLUG}.ipynb").relative_to(ROOT)
                          .as_posix()])
        report = None
        for line in text.splitlines():
            if line.startswith("Output:"):
                report = json.loads((ROOT / line.split(":", 1)[1].strip() / "run.json").read_text(encoding="utf-8"))
        if code or not report or not report.get("rows"):
            raise SystemExit(f"local notebook run failed: {text.strip()[-300:]}")
        mark("local", {"rows": report["rows"], "seconds": report["elapsed_seconds"]})
        log("INFO", f"built {SLUG}; local rows in {report['elapsed_seconds']:.0f} s")
    ref = f"{USER}/{SLUG}"
    if not state().get("version"):
        text = kaggle("kernels", "push", "-p", SUBMISSION.relative_to(ROOT).as_posix())
        match = re.search(r"version\s+(\d+)", text, re.IGNORECASE)
        if not match:
            raise SystemExit(f"kernel version not in the push output: {text.strip()[-200:]}")
        mark("version", match.group(1))
        log("INFO", f"pushed {ref} version {match.group(1)} (private)")
    version = state()["version"]
    if state().get("parity") is None:
        deadline = time.time() + 3 * 3600
        while True:
            _, status = run(["kaggle", "kernels", "status", ref])
            if "COMPLETE" in status.upper():
                break
            if "ERROR" in status.upper() or "CANCEL" in status.upper():
                raise SystemExit(f"kernel {ref}: {status.strip()[-200:]}")
            if time.time() > deadline:
                raise SystemExit(f"kernel {ref} not complete after 3 hours")
            time.sleep(300)
        fetch = ROOT / "data/runs" / f"kaggle-{SLUG}"
        fetch.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", ref, "-p", fetch.relative_to(ROOT).as_posix())
        code, text = run([*LAUNCH, "-c", "import json,sys,polars as pl; print(json.dumps(pl.read_parquet(sys.argv[1])"
                          ".to_dicts()))", (fetch / "submission.parquet").as_posix()])
        kaggle_rows = json.loads(text.strip().splitlines()[-1]) if code == 0 else None
        rows = {r["row_id"]: r["reward"] for r in state()["local"]["rows"]}
        if not kaggle_rows or {r["row_id"] for r in kaggle_rows} != set(rows):
            raise SystemExit(f"Kaggle output rows missing or different: {text.strip()[-200:]}")
        worst = max(abs(r["reward"] - rows[r["row_id"]]) / max(abs(rows[r["row_id"]]), 1e-12) for r in kaggle_rows)
        mark("parity", worst)
        if worst > 1e-6:
            raise SystemExit(f"Kaggle and local rewards differ (relative {worst:.2e}); nothing submitted")
        log("INFO", f"Kaggle rows match the local ones (relative {worst:.1e})")
    if state().get("submitted"):
        return
    if os.environ.get("SUBMIT_PREPARE_ONLY") == "1":                 # user 2026-11-10: pre-run, submit later
        log("INFO", "prepared (kernel run and parity checked); submission left for a later run")
        return
    import fcntl
    lock = open(ROOT / "pipeline/setup/submit.lock", "a")                   # one submitter at a time
    fcntl.flock(lock, fcntl.LOCK_EX)
    while True:
        if datetime.datetime.now(datetime.timezone.utc) >= AUTHORIZED_UNTIL:
            raise SystemExit("the automatic-submission authorization (until 2026-11-13 00:00 Taipei) has ended")
        count, _ = used_today()
        day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        spare = day in SPARE_ALLOWED and NAME.startswith(SPARE_ALLOWED[day])
        if count <= (4 if spare else 3):
            if count == 4:
                log("INFO", f"using the spare 5th submission of {day} (user exception: {SPARE_ALLOWED[day]})")
            break
        log("INFO", f"{count} submissions already today (UTC); waiting for the next day (4 automatic a day, 1 kept spare)")
        now = datetime.datetime.now(datetime.timezone.utc)
        tomorrow = (now + datetime.timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
        time.sleep(max(60, (tomorrow - now).total_seconds()))
    if GATES == "none":
        message = f"{NAME}: {DESCRIPTION}; submitted by user instruction without Kaggle gates"
        kaggle("competitions", "submit", COMPETITION, "-k", ref, "-v", str(version), "-f", "submission.parquet", "-m",
               message)
        _, rows = used_today()
        submission_ref = rows[0].split(",")[0] if rows else "?"
        mark("submitted", {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "ref": submission_ref})
        row = (f"| {time.strftime('%Y-%m-%d %H:%M:%S')} | {submission_ref} | {NAME} {POLICY_CLASS}（{DESCRIPTION}） | "
               f"Notebook {ref} v{version} | 無 Kaggle 關卡（使用者 2026-11-10 16:4x 指示：{mandated}） | 無 | （待出分） | "
               f"本機公開場景 {state()['local']['seconds']:.0f} 秒 | 使用者指示提交 | Kaggle 與本機逐場相對差 "
               f"{state()['parity']:.1e} |\n")
        with (ROOT / "docs/experiments/leaderboard-log.md").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(row)
        log("RESULT", f"submitted {ref} v{version} (ref {submission_ref}); score in about 8 hours")
        return
    i, s = gates["inner"], gates["s4_first"]
    message = (f"{NAME}: {DESCRIPTION}; INNER 29 {100 * i['rel']:+.1f}% and S4 first 30 {100 * s['rel']:+.1f}% vs the "
               "rule a1_cold3+a1_econ (official simulator)")
    kaggle("competitions", "submit", COMPETITION, "-k", ref, "-v", str(version), "-f", "submission.parquet", "-m",
           message)
    _, rows = used_today()
    submission_ref = rows[0].split(",")[0] if rows else "?"
    mark("submitted", {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "ref": submission_ref})
    row = (f"| {time.strftime('%Y-%m-%d %H:%M:%S')} | {submission_ref} | {NAME} {POLICY_CLASS}（{DESCRIPTION}；底 "
           f"G36PolicyB2R2 = a1_cold3+a1_econ） | Notebook {ref} v{version} | INNER {i['n']} 場與規則配對差 "
           f"{i['mean_diff']:+.6f}（{100 * i['rel']:+.1f}%，SE {i['se']:.6f}，{i['better']} 較好、{i['worse']} 較差）；"
           f"S4 前 {s['n']} 場 {s['mean_diff']:+.6f}（{100 * s['rel']:+.1f}%，SE {s['se']:.6f}，{s['better']} 較好、"
           f"{s['worse']} 較差）；官方模擬器 | 無 | （待出分） | 本機公開場景 {state()['local']['seconds']:.0f} 秒 | "
           f"使用者 2026-10-27「授權自動提交」「照規則走」、2026-10-28「如果驗證結果不錯就可以提交然後換新底」 | "
           f"Kaggle 與本機逐場相對差 {state()['parity']:.1e} |\n")
    with (ROOT / "docs/experiments/leaderboard-log.md").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(row)
    log("RESULT", f"submitted {ref} v{version} (ref {submission_ref}); score in about 8 hours")


if __name__ == "__main__":
    main()
