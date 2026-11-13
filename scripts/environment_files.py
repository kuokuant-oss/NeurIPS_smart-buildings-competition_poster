"""Repair uv's exFAT relocation records and remove generated AppleDouble metadata."""
import base64
import csv
import hashlib
import os
from pathlib import Path


def reconcile(venv):
    """Touch only generated .venv files, never official data or source snapshots."""
    roots = [venv / "Lib/site-packages", *venv.glob("lib/python*/site-packages")]
    repaired = 0
    for base in roots:
        if not base.is_dir():
            continue
        for record in base.glob("*.dist-info/RECORD"):
            if record.parent.name.startswith("._"):
                continue
            with record.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            changed = False
            for row in rows:
                original = base / row[0]
                if original.exists() or row[0].endswith(".pyc"):
                    continue
                parts = Path(row[0]).parts
                if any(p.startswith("._") for p in parts):
                    continue
                if len(parts) < 3 or not parts[0].endswith(".data") or parts[1] not in {"scripts", "data"}:
                    raise RuntimeError(f"Missing installed package file: {row[0]}")
                destination = venv / ("Scripts" if os.name == "nt" else "bin") if parts[1] == "scripts" else venv
                destination = destination.joinpath(*parts[2:])
                if not destination.is_file():
                    raise RuntimeError(f"Incomplete wheel relocation: {row[0]}")
                content = destination.read_bytes()
                digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
                # .data/data is copied unchanged. Scripts have rewritten shebangs.
                if parts[1] == "data" and row[1] and row[1] != "sha256=" + digest:
                    raise RuntimeError(f"Relocated data checksum mismatch: {row[0]}")
                row[:] = [Path(os.path.relpath(destination, base)).as_posix(),
                          "sha256=" + digest, str(len(content))]
                changed = True
                repaired += 1
            if changed:
                with record.open("w", encoding="utf-8", newline="") as stream:
                    csv.writer(stream, lineterminator="\n").writerows(rows)
    removed = 0
    for folder, dirs, names in (entry for base in roots if base.is_dir() for entry in os.walk(base)):
        for name in names:
            if not name.startswith("._"):
                continue
            path = Path(folder) / name
            try:
                with path.open("rb") as stream:
                    magic = stream.read(4)
                if magic == b"\x00\x05\x16\x07":
                    path.unlink()
                    removed += 1
            except FileNotFoundError:
                pass
    print(f"Environment files: repaired {repaired} wheel records; removed {removed} AppleDouble sidecars", flush=True)
