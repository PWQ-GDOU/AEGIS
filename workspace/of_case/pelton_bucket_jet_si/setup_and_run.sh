#!/usr/bin/env bash
set -euo pipefail

CASE_DIR=$(cd "$(dirname "$0")" && pwd)
SOURCE_STL=${SOURCE_STL:-}
SI_STL="$CASE_DIR/constant/triSurface/dec_full_runner_local_si.stl"

for command_name in blockMesh snappyHexMesh topoSet createPatch checkMesh potentialFoam simpleFoam postProcess foamToVTK foamListTimes surfaceTransformPoints; do
    command -v "$command_name" >/dev/null || { echo "Missing OpenFOAM command: $command_name" >&2; exit 2; }
done
if [[ -z "$SOURCE_STL" || ! -f "$SOURCE_STL" ]]; then
    echo "Set SOURCE_STL to the historical millimetre-coordinate dec_full_runner_local.stl" >&2
    exit 2
fi

cd "$CASE_DIR"
RESTORE_DIR=$(mktemp -d)
restore_dictionaries() {
    [[ -f "$RESTORE_DIR/fvSolution" ]] && cp "$RESTORE_DIR/fvSolution" system/fvSolution
    [[ -f "$RESTORE_DIR/controlDict" ]] && cp "$RESTORE_DIR/controlDict" system/controlDict
    rm -rf "$RESTORE_DIR"
}
trap restore_dictionaries EXIT INT TERM

mkdir -p constant/triSurface
surfaceTransformPoints -scale '(0.001 0.001 0.001)' "$SOURCE_STL" "$SI_STL" > log.surfaceTransformPoints 2>&1
sha256sum "$SI_STL" | tee si_stl.sha256

rm -rf constant/polyMesh 0 postProcessing VTK
find . -mindepth 1 -maxdepth 1 -type d -regextype posix-extended \
    -regex './[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?' -exec rm -rf {} +
cp -a 0.orig 0

blockMesh > log.blockMesh 2>&1
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
topoSet > log.topoSet 2>&1
createPatch -overwrite > log.createPatch 2>&1
checkMesh > log.checkMesh 2>&1
grep -q 'Mesh OK' log.checkMesh || { echo "checkMesh did not report Mesh OK" >&2; exit 3; }

cp system/fvSolution "$RESTORE_DIR/fvSolution"
cp system/controlDict "$RESTORE_DIR/controlDict"
cp system/fvSolution.stage1 system/fvSolution
cp system/controlDict.stage1 system/controlDict
potentialFoam -writePhi > log.potentialFoam 2>&1
simpleFoam > log.simpleFoam.stage1 2>&1

cp "$RESTORE_DIR/fvSolution" system/fvSolution
cp "$RESTORE_DIR/controlDict" system/controlDict
simpleFoam > log.simpleFoam.final 2>&1

postProcess -latestTime -func writeCellCentres > log.writeCellCentres 2>&1
foamToVTK -latestTime -patches '(bucket)' -overwrite > log.foamToVTK 2>&1
checkMesh -latestTime > log.checkMesh.final 2>&1
LATEST_TIME=$(foamListTimes -latestTime)
python3 stage_step30_artifacts.py \
    --case "$CASE_DIR" \
    --time "$LATEST_TIME" \
    --solver-log "$CASE_DIR/log.simpleFoam.final" \
    --mesh-log "$CASE_DIR/log.checkMesh.final"

echo "SI case completed in $CASE_DIR"
echo "Step30 source bundle: $CASE_DIR/step30_export"
echo "Inspect strict residual convergence in log.simpleFoam.final before using absolute results."
