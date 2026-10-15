"""Checks that regenerated files are byte-identical to the released ones.

Inputs:  checksums.sha256 (SHA-256 of every file the pipeline produces, in `sha256sum` format)
Outputs: a per-folder report; exit code 1 if any file differs.

Files that have not been generated yet (e.g. data written by 01_process_wvs.py when the raw
WVS file is absent) are reported as missing, not as failures.
Use --write to record new checksums after an intentional change.
"""

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "checksums.sha256"
CHECKED_DIRS = ["data", "results/final_scores", "results/robustness", "outputs/tables"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest() -> None:
    files = sorted(p for d in CHECKED_DIRS for p in (ROOT / d).rglob("*") if p.is_file())
    # The raw survey file is supplied by the user and is not part of the release.
    files = [p for p in files if not p.relative_to(ROOT).as_posix().startswith("data/raw/")]
    lines = [f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}" for p in files]
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} checksums to {MANIFEST.name}")


def verify() -> bool:
    status = Counter()
    for line in MANIFEST.read_text().splitlines():
        expected, rel_path = line.split(maxsplit=1)
        path = ROOT / rel_path
        folder = str(Path(rel_path).parent)
        if not path.exists():
            status[(folder, "missing")] += 1
        elif sha256(path) == expected:
            status[(folder, "identical")] += 1
        else:
            status[(folder, "DIFFERENT")] += 1
            print(f"DIFFERENT: {rel_path}")

    for folder in sorted({f for f, _ in status}):
        counts = ", ".join(f"{status[(folder, s)]} {s}" for s in ("identical", "DIFFERENT", "missing") if status[(folder, s)])
        print(f"{folder:<25} {counts}")
    return not any(s == "DIFFERENT" for _, s in status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="record checksums of the current files")
    args = parser.parse_args()
    if args.write:
        write_manifest()
    else:
        sys.exit(0 if verify() else 1)


if __name__ == "__main__":
    main()
