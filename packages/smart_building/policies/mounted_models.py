"""Model files of a submission notebook, found in the mounted Kaggle inputs and checked against a manifest.

A fold ensemble's model files come from more than one Kaggle input: the private dataset of the folds trained on the
Mac (scripts/model_dataset.py, e.g. antonktt/sb-j9-models, files <fold>-agent-NNNN.pt) and the outputs of the chain
kernels trained on Kaggle (e.g. antonktt/sb-j9-f1-c2, file dagger/models/agent-NNNN.pt). Each manifest entry is
{"fold", "file" (path inside its input), "sha256", "source": {"type": "dataset" | "kernel", "ref": "user/slug"}}.

locate_models looks for each file where Kaggle mounts its input (<root>/<slug>/<file>, or the per-type layouts
<root>/datasets/<user>/<slug>/<file> and <root>/notebooks/<user>/<slug>/<file>); a file found there must have the
manifest's SHA-256, otherwise the notebook stops (no fallback to another copy). If none of those paths exists, the
mounted inputs are searched (a few levels deep) for a file at the same relative path with the manifest's SHA-256, the
way remote/paths.py finds earlier outputs by a marker instead of a fixed path. A file that cannot be found with its
SHA-256 stops the notebook: it never runs with a model other than the manifest's.
"""
import hashlib
import os
from pathlib import Path, PurePosixPath

SEARCH_DEPTH = 6
IGNORED_PREFIXES = ("._", ".DS_Store", "__pycache__")
SOURCE_TYPES = ("dataset", "kernel")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_entry(entry):
    """(user, slug, relative path) of a manifest entry; raises on a malformed entry."""
    digest = str(entry.get("sha256", ""))
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{entry.get('fold')}: the manifest needs a SHA-256 for every model file")
    source = entry.get("source") or {}
    if source.get("type") not in SOURCE_TYPES:
        raise ValueError(f"{entry.get('fold')}: source type must be one of {SOURCE_TYPES}")
    user, _, slug = str(source.get("ref", "")).partition("/")
    if not user or not slug or "/" in slug:
        raise ValueError(f"{entry.get('fold')}: source ref must be user/slug")
    relative = PurePosixPath(str(entry.get("file", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{entry.get('fold')}: file must be a relative path inside its input")
    return user, slug, relative


def candidates(root, entry):
    user, slug, relative = check_entry(entry)
    root = Path(root)
    kind = "datasets" if entry["source"]["type"] == "dataset" else "notebooks"
    return [root / slug / relative, root / kind / user / slug / relative]


def search(root, relative, depth=SEARCH_DEPTH):
    """Files under root (at most `depth` folder levels) whose path ends with `relative`, in a sorted order."""
    root = Path(root)
    parts = tuple(PurePosixPath(relative).parts)
    found = []
    for directory, folders, files in os.walk(root):
        level = len(Path(directory).relative_to(root).parts)
        folders[:] = sorted(f for f in folders if not f.startswith(IGNORED_PREFIXES)) if level < depth else []
        if parts[-1] in files:
            path = Path(directory) / parts[-1]
            if tuple(path.parts[-len(parts):]) == parts:
                found.append(path)
    return sorted(found)


def locate_models(entries, root):
    """The checked path of every manifest entry, in manifest order (see the module docstring); raises RuntimeError if a
    file at a mount path differs from its SHA-256 or no file with its SHA-256 is found."""
    paths = []
    for entry in entries:
        want = entry["sha256"]
        hit = None
        for path in candidates(root, entry):
            if path.is_file():
                if sha256_file(path) != want:
                    raise RuntimeError(f"{path}: SHA-256 differs from the manifest ({entry.get('fold')})")
                hit = path
                break
        if hit is None:
            _, _, relative = check_entry(entry)
            hit = next((p for p in search(root, relative) if sha256_file(p) == want), None)
        if hit is None:
            raise RuntimeError(f"{entry.get('fold')} {entry['file']} ({entry['source']['ref']}): no file with the "
                               f"manifest's SHA-256 under {root}")
        paths.append(hit)
    return paths
