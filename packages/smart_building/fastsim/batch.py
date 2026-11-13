"""The batch-simulator contract of learned/b2rl_batch.py on the fast simulator, for
`scripts/b2rl.py train --batch smart_building.fastsim.batch:RomBatch` (or ExactBatch, ParallelOfficialBatch).

The slots are spread over worker processes (slot i lives in worker i % workers); each worker is an OfficialBatch of
its slots in a process where fastsim is installed (rom or exact; nothing for ParallelOfficialBatch), so every slot
is exactly what OfficialBatch gives on that simulator, and the per-slot random states stay OfficialBatch's. A step
scatters the actions to the workers and gathers their results, so the workers step their slots at the same time;
loads are sent without waiting and collected when a later call needs them, so new episodes also start in parallel.

Workers: SB_FASTSIM_BATCH_WORKERS, else one per slot up to half the CPUs. Each worker holds its slots' simulators
(about 0.6-1.6 GB each), so slots x that is the memory to plan for."""
import multiprocessing
import os
import sys
import traceback


def _worker(connection, init):
    sys.dont_write_bytecode = True
    for path in reversed(init["sys_path"]):
        if path not in sys.path:
            sys.path.insert(0, path)
    from smart_building.learned.b2rl_env import setup_simulator
    setup_simulator(init["root"])
    if init["mode"] != "official":
        from smart_building import fastsim
        fastsim.install(init["mode"])
    from smart_building.learned.b2rl_batch import OfficialBatch
    batch = OfficialBatch(init["slots"])
    while True:
        command, payload = connection.recv()
        try:
            if command == "close":
                batch.close()
                connection.send(("ok", None))
                return
            if command == "load":
                i, scenario_dir = payload
                batch.load(i, scenario_dir)
                connection.send(("ok", (i, batch.metadata(i), batch.action_spec(i))))
            elif command == "reset":
                connection.send(("ok", batch.reset(payload)))
            elif command == "step":
                connection.send(("ok", batch.step(payload)))
            else:
                raise ValueError(f"unknown command {command!r}")
        except Exception as exc:  # noqa: BLE001 - reported to the trainer, which raises it
            connection.send(("error", f"{exc!r}\n{traceback.format_exc()}"))


class _ParallelBatch:
    mode = "official"

    def __init__(self, slots, device=None, workers=None):
        from smart_building.learned.b2rl_env import ROOT
        self.slots = int(slots)
        count = workers or int(os.environ.get("SB_FASTSIM_BATCH_WORKERS", "0")) \
            or max(1, min(self.slots, (os.cpu_count() or 2) // 2))
        count = max(1, min(int(count), self.slots))
        context = multiprocessing.get_context("spawn")
        self.pipes, self.processes = [], []
        for _ in range(count):
            parent, child = context.Pipe()
            init = {"mode": self.mode, "slots": self.slots, "sys_path": list(sys.path), "root": str(ROOT)}
            process = context.Process(target=_worker, args=(child, init), daemon=True)
            process.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(process)
        self.outstanding = [0] * count          # load replies not collected yet, per worker
        self._metadata, self._spec = [None] * self.slots, [None] * self.slots

    def _owner(self, i):
        return i % len(self.pipes)

    def _receive(self, k):
        status, value = self.pipes[k].recv()
        if status != "ok":
            raise RuntimeError(f"fast batch worker {k} failed: {value}")
        return value

    def _drain(self, k):
        while self.outstanding[k]:
            i, metadata, spec = self._receive(k)
            self.outstanding[k] -= 1
            self._metadata[i], self._spec[i] = metadata, spec

    def _scatter(self, command, items):
        groups = {}
        for i, value in items.items():
            groups.setdefault(self._owner(i), {})[i] = value
        for k, group in groups.items():
            self._drain(k)
            self.pipes[k].send((command, group if command == "step" else sorted(group)))
        out = {}
        for k in groups:
            out.update(self._receive(k))
        return out

    def load(self, i, scenario_dir):
        k = self._owner(i)
        self._metadata[i] = self._spec[i] = None
        self.pipes[k].send(("load", (i, str(scenario_dir))))
        self.outstanding[k] += 1

    def metadata(self, i):
        self._drain(self._owner(i))
        return self._metadata[i]

    def action_spec(self, i):
        self._drain(self._owner(i))
        return self._spec[i]

    def reset(self, slots):
        return self._scatter("reset", {i: None for i in slots})

    def step(self, actions):
        return self._scatter("step", dict(actions))

    def close(self):
        for k, (pipe, process) in enumerate(zip(self.pipes, self.processes)):
            try:
                self._drain(k)
                pipe.send(("close", None))
                pipe.recv()
            except (EOFError, OSError, RuntimeError):
                pass
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
        self.pipes, self.processes = [], []


class ParallelOfficialBatch(_ParallelBatch):
    """OfficialBatch's slots on worker processes, official simulator (the reference for the two below)."""
    mode = "official"


class ExactBatch(_ParallelBatch):
    """fastsim exact: the official discretisation, one triangular solve per step (same rewards)."""
    mode = "exact"


class RomBatch(_ParallelBatch):
    """fastsim rom: the reduced thermal model (cached per scenario in data/runs/fastsim/rom-cache)."""
    mode = "rom"
