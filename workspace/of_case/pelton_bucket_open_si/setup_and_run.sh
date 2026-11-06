#!/usr/bin/env bash
set -euo pipefail

CASE_DIR=$(cd "$(dirname "$0")" && pwd)
NPROCS=${NPROCS:-1}
SI_STL="$CASE_DIR/constant/triSurface/single_bucket_target_authoritative_local_m.stl"
SOURCE_STL=${SOURCE_STL:-$SI_STL}
EXPECTED_SOURCE_SHA256=181eb659091f187e1286ca27b689d654a83b5522b28d6f39aceaa4ebc3fc38ca

for command_name in blockMesh snappyHexMesh topoSet createPatch checkMesh potentialFoam simpleFoam postProcess foamToVTK foamListTimes sha256sum; do
    command -v "$command_name" >/dev/null || { echo "Missing OpenFOAM command: $command_name" >&2; exit 2; }
done
if [[ -z "$SOURCE_STL" || ! -f "$SOURCE_STL" ]]; then
    echo "Set SOURCE_STL to the COMSOL-authoritative metre-coordinate single-bucket STL" >&2
    exit 2
fi
if [[ "$NPROCS" != "1" && "$NPROCS" != "16" ]]; then
    echo "NPROCS must be 1 or the validated 16-way decomposition" >&2
    exit 2
fi
ACTUAL_SOURCE_SHA256=$(sha256sum "$SOURCE_STL" | cut -d' ' -f1)
if [[ "$ACTUAL_SOURCE_SHA256" != "$EXPECTED_SOURCE_SHA256" ]]; then
    echo "Unexpected authoritative single-bucket STL SHA256: $ACTUAL_SOURCE_SHA256" >&2
    exit 2
fi
if (( NPROCS > 1 )); then
    for command_name in mpirun decomposePar reconstructPar; do
        command -v "$command_name" >/dev/null || { echo "Missing parallel command: $command_name" >&2; exit 2; }
    done
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
if [[ "$SOURCE_STL" != "$SI_STL" ]]; then
    cp "$SOURCE_STL" "$SI_STL"
fi
sha256sum "$SI_STL" | tee authoritative_si_stl.sha256

rm -rf constant/polyMesh 0 postProcessing VTK processor*
find . -mindepth 1 -maxdepth 1 -type d -regextype posix-extended \
    -regex './[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?' -exec rm -rf {} +
cp -a 0.orig 0

blockMesh > log.blockMesh 2>&1
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
topoSet > log.topoSet 2>&1
createPatch -overwrite > log.createPatch 2>&1
checkMesh > log.checkMesh 2>&1
grep -q 'Mesh OK' log.checkMesh || { echo "checkMesh did not report Mesh OK" >&2; exit 3; }
grep -Eq 'bucket[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+ok \(closed singly connected\)' log.checkMesh \
    || { echo "bucket patch is not closed and singly connected" >&2; exit 3; }

cp system/fvSolution "$RESTORE_DIR/fvSolution"
cp system/controlDict "$RESTORE_DIR/controlDict"
cp system/fvSolution.stage1 system/fvSolution
cp system/controlDict.stage1 system/controlDict
potentialFoam -writePhi > log.potentialFoam 2>&1

if (( NPROCS > 1 )); then
    decomposePar -force > log.decomposePar 2>&1
    mpirun -np "$NPROCS" simpleFoam -parallel > log.simpleFoam.stage1 2>&1
    cp "$RESTORE_DIR/fvSolution" system/fvSolution
    cp "$RESTORE_DIR/controlDict" system/controlDict
    mpirun -np "$NPROCS" simpleFoam -parallel > log.simpleFoam.final 2>&1
    reconstructPar -latestTime > log.reconstructPar 2>&1
else
    simpleFoam > log.simpleFoam.stage1 2>&1
    cp "$RESTORE_DIR/fvSolution" system/fvSolution
    cp "$RESTORE_DIR/controlDict" system/controlDict
    simpleFoam > log.simpleFoam.final 2>&1
fi

if ! grep -q 'solution convergence criteria satisfied' log.simpleFoam.final; then
    echo "Strict SIMPLE residualControl was not satisfied; refusing to stage a production snapshot" >&2
    exit 4
fi
postProcess -latestTime -func writeCellCentres > log.writeCellCentres 2>&1
postProcess -latestTime -func wallShearStress > log.wallShearStress 2>&1
checkMesh -latestTime > log.checkMesh.final 2>&1
grep -q 'Mesh OK' log.checkMesh.final || { echo "final checkMesh did not report Mesh OK" >&2; exit 3; }
rm -rf VTK
foamToVTK -latestTime -no-point-data -ascii > log.foamToVTK 2>&1
LATEST_TIME=$(foamListTimes -latestTime)
EXPORT_DIR="$CASE_DIR/step30_export_${LATEST_TIME}_snapshot"
python3 stage_step30_artifacts.py \
    --case "$CASE_DIR" \
    --time "$LATEST_TIME" \
    --output "$EXPORT_DIR" \
    --solver-log "$CASE_DIR/log.simpleFoam.final" \
    --mesh-log "$CASE_DIR/log.checkMesh.final" \
    --application simpleFoam \
    --mode steady_rans \
    --field-statistic snapshot

echo "SI steady case completed in $CASE_DIR"
echo "Step30 source bundle v2: $EXPORT_DIR"
