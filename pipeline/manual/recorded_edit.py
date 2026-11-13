"""Hand-made steps of the chain, replayed from a recorded edit.

Three files of the chain were made by hand (no script wrote them): the autotune end state with dcv_area_share 0.7
(night1_dcv07), and the merged models gt_m1 (gt_r3d1 + levers) and gt_b2 (gt_r5pa2 + levers). Each edit file in
pipeline/manual/edits/ records the difference between the input file and the file that was used, as a list of
operations on JSON paths; `apply` turns the input into the recorded output, `make` records a new edit.

    python3 pipeline/manual/recorded_edit.py apply <edit.json>          # writes the edit's output file
    python3 pipeline/manual/recorded_edit.py check <edit.json>          # apply in memory and compare with the output
    python3 pipeline/manual/recorded_edit.py make <input> <output> <edit.json> <note>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MISSING = object()


def diff(a, b, path=()):
    """Operations that turn a into b: ("set", path, value) and ("del", path)."""
    if isinstance(a, dict) and isinstance(b, dict):
        ops = []
        for k in a:
            if k not in b:
                ops.append(["del", list(path + (k,))])
        for k in b:
            ops += diff(a.get(k, MISSING), b[k], path + (k,)) if k in a else [["set", list(path + (k,)), b[k]]]
        return ops
    if a is MISSING or a != b or type(a) is not type(b):
        return [["set", list(path), b]]
    return []


def apply(doc, ops):
    for op in ops:
        *parents, last = op[1]
        node = doc
        for k in parents:
            node = node[k]
        if op[0] == "del":
            del node[last]
        else:
            node[last] = op[2]
    return doc


def order_like(doc, keys):
    """Top-level key order of the recorded output (JSON object order is not part of the content, but keeps diffs
    of the written file readable)."""
    return {k: doc[k] for k in keys if k in doc} | {k: v for k, v in doc.items() if k not in keys}


def run_edit(edit):
    src = json.loads((ROOT / edit["input"]).read_text())
    out = order_like(apply(src, edit["ops"]), edit["key_order"])
    return out


def main():
    cmd = sys.argv[1]
    if cmd == "make":
        a, b, edit_path, note = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
        da, db = json.loads(a.read_text()), json.loads(b.read_text())
        edit = {"note": note, "input": a.resolve().relative_to(ROOT).as_posix(),
                "output": b.resolve().relative_to(ROOT).as_posix(), "key_order": list(db), "ops": diff(da, db)}
        assert apply(json.loads(a.read_text()), edit["ops"]) == db
        edit_path.parent.mkdir(parents=True, exist_ok=True)
        edit_path.write_text(json.dumps(edit, indent=1) + "\n")
        print(f"{edit_path}: {len(edit['ops'])} operations")
        return
    edit = json.loads(Path(sys.argv[2]).read_text())
    out = run_edit(edit)
    target = ROOT / edit["output"]
    if cmd == "check":
        same = target.is_file() and json.loads(target.read_text()) == out
        print(f"{edit['output']}: {'identical content' if same else 'DIFFERENT'}")
        raise SystemExit(0 if same else 1)
    if cmd == "apply":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {edit['output']}")


if __name__ == "__main__":
    main()
