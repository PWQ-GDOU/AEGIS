#!/usr/bin/env bash
set -euo pipefail

CASE_DIR=$(cd "$(dirname "$0")" && pwd)
NPROCS=${NPROCS:-32}
BURNIN_START=0.002
BURNIN_END=0.038056848
STABILIZATION_END=0.0452682176
AVERAGING_START=0.1209875984
MIN_AVERAGING_END=0.1931012944
AVERAGING_HARD_END=0.4094423824
STEADY_ASSESSMENT=${STEADY_ASSESSMENT:-$CASE_DIR/steady_fixed_point_assessment.json}
VALIDATOR=${VALIDATOR:-$CASE_DIR/../../step30_urans_validation.py}
WATCHER=${WATCHER:-$CASE_DIR/watch_urans_acceptance.py}

if [[ "$NPROCS" != "32" ]]; then
    echo "NPROCS must be 32 for the validated URANS decomposition" >&2
    exit 2
fi
for command_name in pimpleFoam decomposePar reconstructPar postProcess checkMesh foamToVTK mpirun python3; do
    command -v "$command_name" >/dev/null || { echo "Missing command: $command_name" >&2; exit 2; }
done
for path in "$CASE_DIR/$BURNIN_START" "$STEADY_ASSESSMENT" "$VALIDATOR" "$WATCHER"; do
    [[ -e "$path" ]] || { echo "Missing required URANS input: $path" >&2; exit 2; }
done
python3 - "$STEADY_ASSESSMENT" <<'PY'
import json
import sys

assessment = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    assessment.get("version") != "steady-rans-fixed-point-assessment-v1"
    or assessment.get("accepted_as_steady_fixed_point") is not False
):
    raise SystemExit("URANS requires a rejected steady fixed-point assessment")
PY

cd "$CASE_DIR"
cp system/fvSchemes.urans system/fvSchemes
cp system/fvSolution.urans system/fvSolution
cp system/decomposeParDict.urans system/decomposeParDict

if [[ ! -f urans_burnin.solver.rc ]]; then
    rm -rf processor* postProcessing postProcessing.urans-burnin
    cp system/controlDict.urans_burnin system/controlDict
    decomposePar -force > log.urans.burnin.decompose 2>&1
    mpirun -np "$NPROCS" pimpleFoam -parallel > log.urans.burnin 2>&1
    printf '0\n' > urans_burnin.solver.rc
    mv postProcessing postProcessing.urans-burnin
fi
for directory in processor*/"$BURNIN_END"; do
    [[ -d "$directory" ]] || { echo "Missing burn-in checkpoint: $directory" >&2; exit 3; }
done

if [[ ! -f urans_stabilize.solver.rc ]]; then
    rm -rf postProcessing postProcessing.urans-stabilize
    cp system/controlDict.urans_stabilize system/controlDict
    mpirun -np "$NPROCS" pimpleFoam -parallel > log.urans.stabilize 2>&1
    printf '0\n' > urans_stabilize.solver.rc
    mv postProcessing postProcessing.urans-stabilize
fi
for directory in processor*/"$STABILIZATION_END"; do
    [[ -d "$directory" ]] || { echo "Missing stabilization checkpoint: $directory" >&2; exit 3; }
done

if [[ ! -f urans_extended_burnin.solver.rc ]]; then
    rm -rf postProcessing postProcessing.urans-extended-burnin
    cp system/controlDict.urans_extended_burnin system/controlDict
    mpirun -np "$NPROCS" pimpleFoam -parallel > log.urans.extended-burnin 2>&1
    printf '0\n' > urans_extended_burnin.solver.rc
    mv postProcessing postProcessing.urans-extended-burnin
fi
for directory in processor*/"$AVERAGING_START"; do
    [[ -d "$directory" ]] || { echo "Missing extended burn-in checkpoint: $directory" >&2; exit 3; }
done

if [[ ! -f urans_average.solver.rc ]]; then
    cp system/controlDict.urans_average system/controlDict
    for directory in processor*/"$AVERAGING_START"; do
        rm -rf "$directory/uniform"
        for field in UMean pMean kMean omegaMean wallShearStressMean; do
            rm -f "$directory/$field"
        done
    done
    rm -rf postProcessing
    mpirun -np "$NPROCS" pimpleFoam -parallel > log.urans.average 2>&1 &
    SOLVER_PID=$!
    python3 "$WATCHER" \
        --case "$CASE_DIR" \
        --validator "$VALIDATOR" \
        --averaging-start "$AVERAGING_START" \
        --minimum-flow-throughs 20 \
        --consecutive-required 2 \
        --solver-pid "$SOLVER_PID" > log.urans.gate-watcher 2>&1 &
    WATCHER_PID=$!
    set +e
    wait "$SOLVER_PID"
    SOLVER_RC=$?
    set -e
    if kill -0 "$WATCHER_PID" 2>/dev/null; then
        kill "$WATCHER_PID" 2>/dev/null || true
    fi
    wait "$WATCHER_PID" 2>/dev/null || true
    if (( SOLVER_RC != 0 )); then
        echo "URANS averaging solver failed with rc=$SOLVER_RC" >&2
        exit "$SOLVER_RC"
    fi
    printf '0\n' > urans_average.solver.rc
fi

python3 - "$AVERAGING_START" "$MIN_AVERAGING_END" "$AVERAGING_HARD_END" <<'PY'
from decimal import Decimal
from pathlib import Path
import sys

start = Decimal(sys.argv[1])
minimum_end = Decimal(sys.argv[2])
hard_end = Decimal(sys.argv[3])
times = []
for path in Path("processor0").iterdir():
    if not path.is_dir():
        continue
    try:
        value = Decimal(path.name)
    except Exception:
        continue
    if value >= start:
        times.append((value, path.name))
times.sort()
if len(times) < 2 or not minimum_end <= times[-1][0] <= hard_end:
    raise SystemExit(
        f"averaging end is outside [{minimum_end}, {hard_end}]: {times[-2:]}"
    )
Path("urans_average.previous_time").write_text(times[-2][1] + "\n", encoding="ascii")
Path("urans_average.latest_time").write_text(times[-1][1] + "\n", encoding="ascii")
PY
PREVIOUS_TIME=$(tr -d '\r\n' < urans_average.previous_time)
LATEST_TIME=$(tr -d '\r\n' < urans_average.latest_time)

reconstructPar -time "$PREVIOUS_TIME,$LATEST_TIME" > log.urans.average.reconstruct 2>&1
postProcess -time "$LATEST_TIME" -func writeCellCentres > log.urans.average.writeCellCentres 2>&1
checkMesh -time "$LATEST_TIME" > log.urans.average.checkMesh 2>&1
grep -q 'Mesh OK' log.urans.average.checkMesh || { echo "URANS checkMesh did not report Mesh OK" >&2; exit 3; }
rm -rf VTK
foamToVTK -time "$LATEST_TIME" -no-point-data -ascii > log.urans.average.foamToVTK 2>&1

SAMPLE_COUNT=$(grep -c '^Time = ' log.urans.average)
VALIDATION_REPORT="$CASE_DIR/urans_time_mean_validation_${LATEST_TIME}.json"
python3 "$VALIDATOR" \
    --case "$CASE_DIR" \
    --field-time "$LATEST_TIME" \
    --previous-time "$PREVIOUS_TIME" \
    --averaging-start "$AVERAGING_START" \
    --sample-count "$SAMPLE_COUNT" \
    --output "$VALIDATION_REPORT"

EXPORT_DIR="$CASE_DIR/step30_export_${LATEST_TIME}_time_mean"
python3 stage_step30_artifacts.py \
    --case "$CASE_DIR" \
    --time "$LATEST_TIME" \
    --output "$EXPORT_DIR" \
    --solver-log "$CASE_DIR/log.urans.average" \
    --mesh-log "$CASE_DIR/log.urans.average.checkMesh" \
    --application pimpleFoam \
    --mode urans \
    --field-statistic time_mean \
    --averaging-start "$AVERAGING_START" \
    --averaging-end "$LATEST_TIME" \
    --averaging-samples "$SAMPLE_COUNT" \
    --validation-report "$VALIDATION_REPORT" \
    --steady-assessment "$STEADY_ASSESSMENT"
printf '0\n' > urans_time_mean.driver.rc

echo "Validated URANS time-mean source bundle: $EXPORT_DIR"
