"""The j7b-v5 DAgger chain entry with the Mac driver's stop file (the COLD3 student chain j7b-cold3,
docs/plan/plan-direction-20261020.md §13; build note docs/plan/reviews/COLD3-chain-build-20261024.md).

scripts/local_chain.py runs a chain task's entry on this Mac in short, crash-safe segments and passes it "stop_file"
(a pause: the segment stops after the iteration in progress has been saved). learned/dagger_v5.py, the j7b-v5s method
whose files stay byte for byte as pinned, has no such setting and refuses the key. This entry is
dagger_v5.run_dagger_segment with that one addition and nothing else:
  - without "stop_file" (every Kaggle segment, remote/runner.py TaskLoop.run_dagger) it calls
    dagger_v5.run_dagger_segment once with the same arguments and returns its result: the same chain, files and result;
  - with "stop_file" (a Mac segment) the key is taken out and dagger_v5.run_dagger_segment runs one iteration per call,
    each later call resuming in place from the checkpoint the previous one saved in out_dir (as a relay resumes from
    a mounted segment); the stop file is looked at before every call; the deadline and "max_iterations_per_segment"
    keep their meaning for the whole segment. DAgger's checkpoints, resume and identity make the chain the same, bit
    for bit, however its iterations are split into calls (learned/dagger_v5.py; tests/test_learned_dagger_v5.py,
    tests/test_cold3_chain.py).
"""
import json
from pathlib import Path

from smart_building.learned import dagger_v5

STOPPED = "stop_file"


def current_iteration(*directories):
    """(iteration, checkpoint name) of the newest checkpoint named by latest.json in the first directory that has one."""
    for directory in directories:
        latest = Path(directory) / "latest.json" if directory else None
        if latest is not None and latest.is_file():
            record = json.loads(latest.read_text(encoding="utf-8"))
            return int(record.get("iteration", 0)), record.get("checkpoint")
    return 0, None


def run_dagger_segment(config, state_dir, out_dir, deadline_utc):
    """One segment of a j7b-v5 DAgger chain (dagger_v5.run_dagger_segment) that also stops when config["stop_file"]
    exists before an iteration (result "stopped": "stop_file")."""
    config = dict(config)
    if "stop_file" not in config:
        return dagger_v5.run_dagger_segment(config, state_dir, out_dir, deadline_utc)
    stop = config.pop("stop_file")
    cap = config.get("max_iterations_per_segment")
    done, result, state = 0, None, state_dir
    while True:
        if stop and Path(stop).exists():
            if result is None:
                iteration, checkpoint = current_iteration(out_dir, state_dir)
                return {"finished": False, "iteration": iteration, "iterations_this_segment": 0, "stopped": STOPPED,
                        "checkpoint": checkpoint, "variant": dagger_v5.VARIANT,
                        "student_format": dagger_v5.STUDENT_FORMAT}
            return dict(result, iterations_this_segment=done, stopped=STOPPED)
        result = dagger_v5.run_dagger_segment(dict(config, max_iterations_per_segment=1), state, out_dir, deadline_utc)
        done += int(result.get("iterations_this_segment") or 0)
        state = out_dir                                  # the next call resumes from the checkpoint just saved
        if result.get("finished") or result.get("stopped") != "max_iterations_per_segment":
            return dict(result, iterations_this_segment=done)          # finished, or the deadline
        if cap is not None and done >= int(cap):
            return dict(result, iterations_this_segment=done)          # the segment's own iteration cap
