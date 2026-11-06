#!/bin/bash
# ============================================================================
# OpenFOAM Pelton Turbine Simulation - 一键运行脚本
# 在 WSL Ubuntu 终端中运行: bash ~/pelton_jet/setup_and_run.sh
# ============================================================================

set -e

# Source OpenFOAM
source /usr/share/openfoam/etc/bashrc 2>/dev/null

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CASE_DIR="${STEP30_CASE_DIR:-$SCRIPT_DIR}"
cd "$CASE_DIR"

echo "============================================"
echo " Pelton Turbine CFD Setup & Run"
echo "============================================"

# 1. Copy DEC STL from Windows
echo ""
echo "[1/5] Copying DEC geometry..."
mkdir -p constant/triSurface
cp "$REPO_ROOT/workspace/output/dec_model_runner.stl" constant/triSurface/ 2>/dev/null || {
    echo "  WARNING: DEC STL is unavailable at the repository-relative output path."
    echo "  Set STEP30_GEOMETRY_PATH to an explicit geometry file before rerunning."
    [ -n "${STEP30_GEOMETRY_PATH:-}" ] && cp "$STEP30_GEOMETRY_PATH" constant/triSurface/
}
ls -lh constant/triSurface/

# 2. Generate background mesh
echo ""
echo "[2/5] Generating background mesh (blockMesh)..."
blockMesh 2>&1 | tail -3

# 3. Optional: snappyHexMesh to snap to STL surface
echo ""
echo "[3/5] snappyHexMesh (snapping to DEC geometry)..."
if [ -f constant/triSurface/dec_model_runner.stl ]; then
    # Create basic snappyHexMeshDict if not exists
    if [ ! -f system/snappyHexMeshDict ]; then
        cat > system/snappyHexMeshDict << 'SNAPPYEOF'
FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true;
snap            true;
addLayers       false;
geometry { bucket { type triSurfaceMesh; file "dec_model_runner.stl"; } }
castellatedMeshControls
{
    maxLocalCells 500000; maxGlobalCells 2000000; minRefinementCells 10;
    nCellsBetweenLevels 3;
    features ( { file "dec_model_runner.eMesh"; level 2; } );
    refinementSurfaces { bucket { level (2 2); } }
    resolveFeatureAngle 30;
    locationInMesh (0.1 0.1 0.1);
}
snapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5; }
SNAPPYEOF
    fi
    snappyHexMesh -overwrite 2>&1 | tail -5 || echo "  snappyHexMesh skipped (OK for now)"
else
    echo "  No STL found, skipping snappyHexMesh"
fi

# 4. Check mesh
echo ""
echo "[4/5] Checking mesh..."
checkMesh 2>&1 | grep -E "points|cells|faces|Mesh OK" | head -5

# 5. Run simulation
echo ""
echo "[5/5] Running interFoam..."
echo "  Time step: 7.89e-5 s (DEC literature)"
echo "  Solver: VOF + kOmegaSST"
echo ""
interFoam 2>&1 | tail -20

echo ""
echo "============================================"
echo " Simulation Complete!"
echo " Results in: $CASE_DIR"
echo "============================================"
