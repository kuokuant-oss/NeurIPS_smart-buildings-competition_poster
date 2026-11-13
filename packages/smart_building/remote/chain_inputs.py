"""Inputs a chain task takes from another, finished chain, named by rule rather than by file (J9).

J9 (B1 stage 3, docs/plan/plan-direction-20261020.md §13) starts each fold's PPO from that fold's B1-L v5 student
and uses the student chain's first-iteration teacher rollouts as demonstrations. Which student a chain selected
(its registered INNER rule, learned/dagger.py best_iteration) is known only when the chain has finished, so the job
names the chain and the rule, and the files are read from the chain's own chain.json, as J7c reads the selected
student (remote/tasks.py mounted_student):

    "student_model":  {"from_chain": {"job": "j7b-v5s", "shard": "f0"}, "take": "selected_student"}
    "demonstrations": {"from_chain": {"job": "j7b-v5s", "shard": "f0"}, "take": "rollouts", "iteration": 1}

J10 (round 2 of stage 3, docs/reference/notes/16-rl-tuning-after-baseline.md §C5 (b)) continues each fold's finished
J9 run from its final checkpoint state, the state.pt of the checkpoint chain.json's result names:

    "init_from":      {"from_chain": {"job": "j9", "shard": "f0"}, "take": "final_state"}

resolve() turns each such value into the specs the learned entry reads (learned/ppo.py resolve_input):
{"source": {"job", "shard", "attempt"}, "file": "dagger/<path>", "sha256"} (a list for "rollouts"). The attempt,
the file names and the SHA-256 digests all come from the mounted chain's chain.json (its result and state_files), so
every segment on every machine (Kaggle CPU or GPU, this Mac) resolves the same values and a resumed run keeps its
identity. Each file is checked against chain.json before the run starts. The same rule tells
scripts/remote.py fetch-sources which files of a Kaggle chain this computer needs (the Mac folds).

Standard library only: the host scripts import it as well as the code snapshot.
"""
import hashlib
import json
from pathlib import Path
import re

TAKES = ("selected_student", "rollouts", "final_state")
SINGLE = ("selected_student", "final_state")      # takes that name one file (resolved to one spec, not a list)


def is_chain_spec(value):
    return isinstance(value, dict) and "from_chain" in value


def chain_specs(config):
    """[(key, spec)] of the config values that name a chain's files by rule."""
    return [(key, value) for key, value in (config or {}).items() if is_chain_spec(value)]


def check_spec(spec):
    """Problems of one spec (empty list when usable)."""
    problems = []
    chain = spec.get("from_chain")
    if not isinstance(chain, dict) or not chain.get("job") or not chain.get("shard"):
        problems.append("from_chain must name the chain's job and shard")
    if spec.get("take") not in TAKES:
        problems.append(f"take must be one of {TAKES}")
    if spec.get("take") == "rollouts" and not isinstance(spec.get("iteration"), int):
        problems.append("take 'rollouts' needs the iteration (an integer)")
    return problems


def chain_files(chain, spec):
    """The chain's files (relative to its dagger/ folder) a spec takes, from a finished chain's chain.json."""
    result = chain.get("result") or {}
    if not chain.get("finished") or not result.get("finished"):
        raise ValueError(f"{spec['from_chain']}: the chain has not finished")
    if spec["take"] == "selected_student":
        model = result.get("model") or {}
        if not model.get("file") or not model.get("sha256"):
            raise ValueError(f"{spec['from_chain']}: chain.json names no selected student")
        return [model["file"]]
    if spec["take"] == "final_state":
        # the run's last checkpoint (learned/ppo.py run_ppo_segment "checkpoint"); its state.pt, without the
        # environment snapshots (a continuation starts new episodes)
        name = f"{result.get('checkpoint')}/state.pt"
        if not result.get("checkpoint") or name not in chain.get("state_files", {}):
            raise ValueError(f"{spec['from_chain']}: chain.json names no final checkpoint state")
        return [name]
    pattern = re.compile(rf"^data/rollout-{int(spec['iteration']):04d}-\d+\.npz$")
    files = sorted(name for name in chain.get("state_files", {}) if pattern.match(name))
    if not files:
        raise ValueError(f"{spec['from_chain']}: no rollout files of iteration {spec['iteration']} in chain.json")
    return files


def expected_digest(chain, spec, name):
    """The SHA-256 chain.json records for one of the files a spec takes (the result's model digest must agree)."""
    digest = chain.get("state_files", {}).get(name)
    if spec["take"] == "selected_student":
        recorded = (chain.get("result") or {}).get("model", {}).get("sha256")
        if digest and recorded and digest != recorded:
            raise ValueError(f"{name}: state_files and result.model disagree in chain.json")
        digest = digest or recorded
    if not digest:
        raise ValueError(f"{name}: no SHA-256 in chain.json")
    return digest


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_source(sources, chain):
    """The one mounted source (entries {"job_id", "shard", "attempt", "path"}) of the chain {"job", "shard"}."""
    matches = [s for s in sources if s.get("job_id") == chain["job"] and s.get("shard") == chain["shard"]]
    if len(matches) != 1:
        raise FileNotFoundError(f"{chain}: {len(matches)} mounted outputs of that chain (expected 1)")
    return matches[0]


def resolve(config, sources):
    """(config with every chain spec resolved, record of what was resolved). sources: the mounted outputs of other jobs
    (the runner's context["sources"]; scripts/local_chain.py passes the same fields)."""
    resolved, record = dict(config), {}
    for key, spec in chain_specs(config):
        problems = check_spec(spec)
        if problems:
            raise ValueError(f"{key}: {'; '.join(problems)}")
        source = find_source(sources, spec["from_chain"])
        root = Path(source["path"])
        chain = json.loads((root / "chain.json").read_text(encoding="utf-8"))
        origin = {"job": spec["from_chain"]["job"], "shard": spec["from_chain"]["shard"], "attempt": source["attempt"]}
        if chain.get("attempt") not in (None, source["attempt"]):
            raise ValueError(f"{key}: chain.json of {origin} says attempt {chain.get('attempt')}")
        items = []
        for name in chain_files(chain, spec):
            digest = expected_digest(chain, spec, name)
            path = root / "dagger" / name
            if not path.is_file():
                raise FileNotFoundError(f"{key}: {path} is missing (scripts/remote.py fetch-sources downloads it)")
            if sha256_file(path) != digest:
                raise RuntimeError(f"{key}: {path} differs from the SHA-256 in chain.json")
            items.append({"source": dict(origin), "file": f"dagger/{name}", "sha256": digest})
        resolved[key] = items[0] if spec["take"] in SINGLE else items
        record[key] = {"from_chain": dict(spec["from_chain"]), "take": spec["take"], "attempt": source["attempt"],
                       "files": {item["file"]: item["sha256"] for item in items},
                       "best_iteration": (chain.get("result") or {}).get("best_iteration")}
    return resolved, record
