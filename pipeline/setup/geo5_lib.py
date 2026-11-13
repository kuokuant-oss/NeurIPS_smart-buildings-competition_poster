"""Pipeline 5 common machinery (2026-11-04, docs/plan/geo-pipeline5.md): paths, the run context (state, log), the
wake-up exception, memory and process checks, and the monitored runners every long job goes through.

Reliability rules (mandatory, lessons of pipelines 1-4):
- every job prints a progress line and saves its checkpoint at least every 5 minutes (HEARTBEAT_S = 240 s);
- ETAs only from measured progress (the batch's own completions; before the first one no ETA is printed);
- our process tree stays under 12 GB RSS (about 0.8 GB per fast-simulator episode worker, at most 12 workers), read
  with `ps`; the host's used memory over 18 GB, a job over 2x its estimate, a crash or a stall wake the coordinator;
- never SIGSTOP-style pausing: a wake terminates the children (SIGTERM, then SIGKILL) and exits;
- naive local timestamps (time.strftime).
Standard library only (imported by host-python helpers too)."""
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "pipeline/setup"
GEO5 = ROOT / "data/runs/b2rl/geo5"
HEARTBEAT_S = 240.0                  # progress + checkpoint at least every 5 minutes
POLL_S = 15.0
RSS_BUDGET_GB = 12.0
GB_PER_WORKER = 0.8
MAX_WORKERS = 12
WAKE_MEM_GB = 18.0
OVERTIME = 2.0
EXIT_WAKE, EXIT_FINAL = 3, 4         # wake: the coordinator fixes and resumes; final: the second attempt failed too
EXIT_INFO = 5                        # an information wake (paper_direction_ready): not a failure, plain --resume


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Wake(Exception):
    """Stop the pipeline and wake the coordinator: reason (gate / mismatch / crash / memory / overtime / stall),
    a text, diagnostic data and the paths worth reading."""

    def __init__(self, reason, text, details=None, paths=(), info=False):
        super().__init__(f"{reason}: {text}")
        self.reason, self.text, self.details, self.paths = reason, text, details or {}, list(paths)
        self.info = info                 # True: not a failure, the stage's attempt is not used up


# ---------------------------------------------------------------- memory and processes
def mem_gb():
    """(used, available) GB of the host from /proc/meminfo; (None, None) elsewhere."""
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0]) / 1048576.0
        return info["MemTotal"] - info["MemAvailable"], info["MemAvailable"]
    except (OSError, KeyError, ValueError):
        return None, None


def _ps():
    try:
        out = subprocess.run(["ps", "-e", "-o", "pid=,ppid=,rss="], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    rows = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit():
            rows[int(parts[0])] = (int(parts[1]), int(parts[2]))
    return rows


def descendants(pid=None):
    pid = pid or os.getpid()
    rows = _ps()
    children, found = {}, []
    for p, (pp, _) in rows.items():
        children.setdefault(pp, []).append(p)
    todo = list(children.get(pid, []))
    while todo:
        p = todo.pop()
        found.append(p)
        todo += children.get(p, [])
    return found, rows


def tree_rss_gb(pid=None):
    """RSS of this process and all its descendants (ps), GB."""
    pid = pid or os.getpid()
    kids, rows = descendants(pid)
    return sum(rows.get(p, (0, 0))[1] for p in [pid] + kids) / 1048576.0


def kill_tree(pid=None, grace=10.0):
    """Terminate every descendant (SIGTERM, then SIGKILL after `grace` s). Never SIGSTOP."""
    kids, _ = descendants(pid)
    kids = [p for p in kids if not _is_tracker(p)]       # multiprocessing's resource tracker cleans up after us
    for p in kids:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    end = time.time() + grace
    while time.time() < end and any(_alive(p) for p in kids):
        time.sleep(0.5)
    for p in kids:
        if _alive(p):
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass


def _is_tracker(pid):
    try:
        return b"resource_tracker" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def worker_count(requested):
    """Workers that fit: at most 12, our tree under 12 GB at 0.8 GB per worker, and the host's available memory
    keeps 3 GB spare."""
    own = tree_rss_gb()
    _, avail = mem_gb()
    fit = int((RSS_BUDGET_GB - own) // GB_PER_WORKER)
    if avail is not None:
        fit = min(fit, int((avail - 3.0) // GB_PER_WORKER))
    return max(1, min(int(requested), MAX_WORKERS, fit))


# ---------------------------------------------------------------- the run context
class Ctx:
    """State (pipeline.json), log (pipeline.log), summary (summary.json) and timing (timing.json) of one state dir."""

    def __init__(self, state_dir, tag="pipeline5"):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path, self.log_path = self.dir / "pipeline.json", self.dir / "pipeline.log"
        self.summary_path, self.wake_path = self.dir / "summary.json", self.dir / "wake.json"
        self.timing_path = self.dir / "timing.json"
        self.tag = tag
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.is_file() else {}
        self.timing = json.loads(self.timing_path.read_text(encoding="utf-8")) if self.timing_path.is_file() else {}
        self.summary_fn = None

    def rel(self, path):
        try:
            return Path(path).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return str(path)

    def log(self, kind, text):
        line = f"{now()} {kind} {self.tag}: {text}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")

    def save(self):
        atomic_json(self.state_path, self.state)
        atomic_json(self.timing_path, self.timing)
        if self.summary_fn is not None:
            try:
                atomic_json(self.summary_path, self.summary_fn())
            except Exception:  # noqa: BLE001 - a summary bug must not stop the run; it is logged
                self.log("WARN", "summary not written: " + traceback.format_exc().replace("\n", " | ")[-600:])

    def measured(self, key, minutes):
        """Record a measured duration (minutes) of a job kind; the next estimate of that kind uses it."""
        self.timing[key] = {"minutes": round(float(minutes), 2), "measured": now()}

    def estimate(self, key, default=None):
        row = self.timing.get(key)
        return float(row["minutes"]) if row else default

    def write_wake(self, wake, stage, attempt, final):
        diag = self.dir / "diag"
        diag.mkdir(exist_ok=True)
        cmd = f"python3 -B pipeline/launch/launch.py pipeline/setup/geo_pipeline5.py --state {self.rel(self.dir)} --resume"
        if wake.info:
            resume = f"information only (not a failure): read the paths, then: {cmd}"
        elif final:
            resume = ("second attempt used: diagnose; a further run needs the coordinator's explicit --attempt 3 "
                      "(outside the plan's two chances)")
        else:
            resume = f"diagnose, fix, then: {cmd} --attempt {attempt + 1}"
        payload = {"time": now(), "stage": stage, "attempt": attempt, "final": final, "info": wake.info,
                   "reason": wake.reason, "text": wake.text, "details": wake.details,
                   "paths": [self.rel(p) for p in wake.paths] + [self.rel(self.log_path), self.rel(self.state_path)],
                   "resume": resume}
        atomic_json(self.wake_path, payload)
        atomic_json(diag / f"wake-{time.strftime('%Y%m%d-%H%M%S')}.json", payload)
        return payload


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


# ---------------------------------------------------------------- monitoring
class Monitor:
    """Heartbeat, checkpoint, memory and overtime checks of one job. tick(done) is called at least every POLL_S.
    estimate_min: the job's estimate (measured earlier) or None; then it is fixed from this job's own rate once
    `fix_after` units are done. stall_min: wake when no unit completes for that long."""

    def __init__(self, ctx, job, total, unit="episodes", estimate_min=None, fix_after=None, stall_min=None,
                 checkpoint=None):
        self.ctx, self.job, self.total, self.unit = ctx, job, total, unit
        self.t0 = time.time()
        self.estimate_min, self.fix_after, self.stall_min = estimate_min, fix_after, stall_min
        self.checkpoint = checkpoint
        self.last_beat = 0.0
        self.last_done, self.last_change = 0, time.time()
        self.mem_high = 0

    def elapsed_min(self):
        return (time.time() - self.t0) / 60.0

    def tick(self, done, extra="", force=False):
        t = time.time()
        if done != self.last_done:
            self.last_done, self.last_change = done, t
        el = self.elapsed_min()
        if self.estimate_min is None and self.fix_after and done >= self.fix_after and done > 0:
            self.estimate_min = el / done * self.total
            self.ctx.log("EVENT", f"{self.job}: estimate fixed from measured progress: {self.estimate_min:.1f} min for "
                                  f"{self.total} {self.unit} ({done} done in {el:.1f} min)")
        used, avail = mem_gb()
        own = tree_rss_gb()
        if used is not None and used > WAKE_MEM_GB:
            self.mem_high += 1
            if self.mem_high >= 2:
                raise Wake("memory", f"{self.job}: host memory used {used:.1f} GB > {WAKE_MEM_GB:.0f} GB (our tree "
                                     f"{own:.1f} GB)", {"used_gb": used, "own_gb": own, "available_gb": avail})
        else:
            self.mem_high = 0
        if own > WAKE_MEM_GB:
            raise Wake("memory", f"{self.job}: our process tree {own:.1f} GB > {WAKE_MEM_GB:.0f} GB",
                       {"own_gb": own, "used_gb": used})
        if self.estimate_min and el > OVERTIME * self.estimate_min:
            raise Wake("overtime", f"{self.job}: {el:.1f} min, over {OVERTIME:.0f}x its estimate "
                                   f"{self.estimate_min:.1f} min ({done}/{self.total} {self.unit})",
                       {"elapsed_min": el, "estimate_min": self.estimate_min, "done": done, "total": self.total})
        if self.stall_min and (t - self.last_change) / 60.0 > self.stall_min and done < self.total:
            raise Wake("stall", f"{self.job}: no {self.unit[:-1] if self.unit.endswith('s') else self.unit} finished "
                                f"for {(t - self.last_change) / 60:.0f} min ({done}/{self.total})",
                       {"stall_min": self.stall_min, "done": done})
        if force or t - self.last_beat >= HEARTBEAT_S:
            self.last_beat = t
            eta = ""
            if 0 < done < self.total:
                eta = f", ETA {el / done * (self.total - done):.0f} min (measured)"
            est = f", estimate {self.estimate_min:.0f} min" if self.estimate_min else ""
            self.ctx.log("PROGRESS", f"{self.job}: {done}/{self.total} {self.unit} in {el:.1f} min{eta}{est}; "
                                     f"memory: our tree {own:.1f} GB, host used {used if used is None else round(used, 1)}"
                                     f" GB{'; ' + extra if extra else ''}")
            if self.checkpoint:
                self.checkpoint()
            else:
                self.ctx.save()


def run_monitored(ctx, job, args, env=None, estimate_min=None, out_file=None, cwd=None):
    """Run a subprocess under a Monitor (heartbeat every <= 4 min, memory, 2x estimate). Its output goes to out_file.
    A wake or an interrupt terminates its process tree. Returns the elapsed minutes; a non-zero exit is a crash wake."""
    out_file = Path(out_file or ctx.dir / "jobs" / f"{job.replace(' ', '_').replace('/', '_')}.log")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    mon = Monitor(ctx, job, 1, unit="jobs", estimate_min=estimate_min)
    ctx.log("EVENT", f"{job} starts (estimate {estimate_min if estimate_min is None else round(estimate_min)} min; "
                     f"output {ctx.rel(out_file)})")
    with out_file.open("a", encoding="utf-8") as stream:
        proc = subprocess.Popen([str(a) for a in args], cwd=cwd or ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            while proc.poll() is None:
                tail = _tail(out_file)
                mon.tick(0, f"last output: {tail}" if tail else "")
                time.sleep(POLL_S)
        except BaseException:
            kill_tree(proc.pid)
            try:
                proc.terminate()
                proc.wait(20)
            except Exception:  # noqa: BLE001
                proc.kill()
            raise
    el = mon.elapsed_min()
    if proc.returncode:
        raise Wake("crash", f"{job} exited with code {proc.returncode}: {_tail(out_file, 600)}",
                   {"returncode": proc.returncode}, [out_file])
    ctx.log("EVENT", f"{job} done in {el:.1f} min")
    return el


def run_thread_monitored(ctx, job, fn, estimate_min=None):
    """Run fn() in a thread under a Monitor (for library calls that block); a wake kills our children and raises."""
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as error:  # noqa: BLE001 - re-raised in the main thread
            box["error"] = error
            box["trace"] = traceback.format_exc()

    th = threading.Thread(target=target, daemon=True)
    mon = Monitor(ctx, job, 1, unit="jobs", estimate_min=estimate_min)
    th.start()
    try:
        while th.is_alive():
            mon.tick(0)
            th.join(POLL_S)
    except BaseException:
        kill_tree()
        raise
    if "error" in box:
        if isinstance(box["error"], Wake):
            raise box["error"]
        raise Wake("crash", f"{job}: {box['trace'][-1200:]}")
    return box.get("value")


def _tail(path, n=160):
    try:
        with Path(path).open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4000))
            lines = [x for x in f.read().decode("utf-8", "replace").splitlines() if x.strip()]
        return lines[-1][-n:] if lines else ""
    except OSError:
        return ""
