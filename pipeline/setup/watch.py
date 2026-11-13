"""Wakes the agent (user 2026-10-28: check once an hour, and at once when results come out or something fails).

Waits until the pipeline log (pipeline/setup/pipeline.log) gets a line tagged EVENT, RESULT or ERROR, a traceback appears
in the full run's training log, the pipeline process is gone without its finish line, or the hour is over; prints
why and exits. The read positions are kept in pipeline/setup/watch.offset, so the next watcher starts where this one
stopped. Usage: watch.py [seconds, default 3600]. Host python3, standard library."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "pipeline/setup/pipeline.log"
OFFSET = ROOT / "pipeline/setup/watch.offset"
TRAIN_LOG = ROOT / "data/runs/b2rl/full-rom-s42.log"
SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3600
TAGS = (" EVENT ", " RESULT ", " ERROR ")


def pipeline_alive():
    out = subprocess.run(["pgrep", "-f", r"^python3 -B \pipeline/setup/pipeline\.py$"], capture_output=True, text=True)
    return bool(out.stdout.split())


def main():
    offsets = json.loads(OFFSET.read_text(encoding="utf-8")) if OFFSET.is_file() else {"log": 0, "train": 0}
    started = time.time()
    while True:
        data = LOG.read_bytes() if LOG.is_file() else b""
        end = data.rfind(b"\n", offsets["log"]) + 1
        lines = data[offsets["log"]:end].decode("utf-8", errors="replace").splitlines() if end > offsets["log"] else []
        if end > offsets["log"]:
            offsets["log"] = end
        hits = [line for line in lines if any(tag in line for tag in TAGS)]
        if TRAIN_LOG.is_file() and TRAIN_LOG.stat().st_size > offsets["train"]:
            text = TRAIN_LOG.read_bytes()[offsets["train"]:].decode("utf-8", errors="replace")
            offsets["train"] = TRAIN_LOG.stat().st_size
            if "Traceback" in text:
                hits.append("ERROR traceback in data/runs/b2rl/full-rom-s42.log: "
                            + text[text.find("Traceback"):][-600:].replace("\n", " | "))
        last = data.decode("utf-8", errors="replace").strip().splitlines()[-1:] if data else []
        if not hits and data and not pipeline_alive() and not (last and "pipeline finished" in last[0]):
            hits.append("ERROR the pipeline process is gone without its finish line: " + (last[0] if last else ""))
        if hits or time.time() - started >= SECONDS:
            OFFSET.write_text(json.dumps(offsets) + "\n", encoding="utf-8")
            print("\n".join(hits) if hits else "HOURLY check")
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
