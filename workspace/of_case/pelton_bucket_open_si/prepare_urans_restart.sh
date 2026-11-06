#!/usr/bin/env bash
set -euo pipefail

CASE_DIR=$(cd "$(dirname "$0")" && pwd)
SOURCE_CASE=${SOURCE_CASE:?Set SOURCE_CASE to the rejected steady case}
SOURCE_TIME=${SOURCE_TIME:-6000}
TARGET_TIME=${TARGET_TIME:-0.002}
PYTHON=${PYTHON:-python3}
TARGET_DIR="$CASE_DIR/$TARGET_TIME"
SOURCE_DIR="$SOURCE_CASE/$SOURCE_TIME"

[[ -d "$SOURCE_DIR" ]] || { echo "Missing steady source time: $SOURCE_DIR" >&2; exit 2; }
[[ -d "$CASE_DIR/constant/polyMesh" ]] || { echo "Missing target polyMesh" >&2; exit 2; }
[[ ! -e "$TARGET_DIR" ]] || { echo "Refusing to replace existing restart: $TARGET_DIR" >&2; exit 2; }

for name in boundary faces neighbour owner points; do
    cmp -s "$SOURCE_CASE/constant/polyMesh/$name" "$CASE_DIR/constant/polyMesh/$name" || {
        echo "Steady and URANS polyMesh files differ: $name" >&2
        exit 3
    }
done
for name in U p k omega nut phi; do
    [[ -f "$SOURCE_DIR/$name" ]] || { echo "Missing steady restart field: $SOURCE_DIR/$name" >&2; exit 2; }
done

mkdir "$TARGET_DIR"
"$PYTHON" - "$SOURCE_DIR" "$TARGET_DIR" "$TARGET_TIME" <<'PY'
from pathlib import Path
import re
import shutil
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
time_name = sys.argv[3]


def copy_field(source_name: str, target_name: str) -> None:
    destination = target / target_name
    shutil.copy2(source / source_name, destination)
    text = destination.read_text(encoding="utf-8", errors="strict")
    text, location_count = re.subn(
        r'(?m)^(\s*location\s+)(?:"[^"]*"|[^;]+)(;)',
        rf'\1"{time_name}"\2',
        text,
        count=1,
    )
    text, object_count = re.subn(
        r'(?m)^(\s*object\s+)[^;]+(;)',
        rf'\1{target_name}\2',
        text,
        count=1,
    )
    if location_count != 1 or object_count != 1:
        raise SystemExit(f"invalid OpenFOAM header: {destination}")
    destination.write_text(text, encoding="utf-8", newline="\n")


for field in ("U", "p", "k", "omega", "nut", "phi"):
    copy_field(field, field)
for field in ("U", "k", "omega", "phi"):
    copy_field(field, f"{field}_0")
PY

echo "Prepared URANS restart $TARGET_DIR from $SOURCE_DIR"
