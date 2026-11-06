# Authoritative open-boundary Pelton bucket SI case

This is the isolated replacement case for the rejected legacy full-runner/cropped-domain steady result. Legacy cases and Step30 artifacts are not overwritten.

## Geometry and units

- Geometry: complete target bucket selected in COMSOL 6.4 from the authoritative full-runner Parasolid
- Local metre STL SHA256: `181eb659091f187e1286ca27b689d654a83b5522b28d6f39aceaa4ebc3fc38ca`
- OpenFOAM coordinates: m
- Step30 internal coordinates: mm (`coordinate_scale_to_mm=1000`)
- `U`: m/s
- `p`: m2/s2 kinematic pressure
- `wallShearStress`: m2/s2 kinematic wall shear stress

The STL is already in metres and must not be scaled again. The historical millimetre-valued background vertices are converted exactly once by `blockMeshDict: scale 0.001`.

## Physical inputs

- Jet speed: 104.8 m/s
- Jet diameter: 0.24 m
- Water kinematic viscosity: `1e-6 m2/s`
- Reynolds number: `2.5152e7`
- Turbulence intensity: 5%
- Turbulence length scale: `0.07D = 0.0168 m`
- `k = 41.1864 m2/s2`
- `omega = 697.44025 s-1`

The non-jet inlet remainder, outlet, and sides are pressure openings. This is a single-phase submerged-jet approximation, not a VOF free jet.

## Validated mesh

The production v6 mesh uses `mergePatchFaces false`, level-3 bucket refinement, `maxBoundarySkewness 4`, and `minDeterminant 0.01`.

- Cells: 837,972
- Bucket faces: 218,988
- Bucket topology: closed, singly connected
- Maximum non-orthogonality: 45.270658
- Maximum skewness: 2.7861559
- Standard `checkMesh`: `Mesh OK`

Topology v3 contains 437,976 collision triangles with zero duplicate or inverted triangles. Its OpenFOAM face-area-vector conservation error is below `1e-10` globally.

## Run

OpenFOAM v2412 commands must already be available:

```bash
NPROCS=16 ./setup_and_run.sh
```

`NPROCS=1` is also accepted. Parallel OpenFOAM must resolve a real MPI `libPstream.so`, not the dummy library.

The runner:

1. verifies the authoritative STL hash;
2. rebuilds and validates the mesh;
3. initializes with `potentialFoam`;
4. runs a conservative SIMPLE startup stage;
5. applies strict `residualControl` with pressure threshold `1e-4`;
6. refuses production staging unless all SIMPLE criteria are satisfied;
7. emits a source bundle v2 with explicit `simpleFoam/steady_rans/snapshot/U` semantics.

The validated v6 investigation passed the pressure gate but did not reach a steady fixed point by 6000 iterations: `Uz` and `k` residuals remained above threshold, 5900-to-6000 field relative L2 changes remained 1.1-2.5%, and turbulence bounding persisted. That steady field is retained as diagnostic evidence and is not a production Step30 snapshot. The production path therefore uses a separate `pimpleFoam` URANS case and an explicitly identified `UMean` artifact.

## URANS time-mean production

`run_urans_time_mean.sh` starts from a reconstructed `0.002 s` field in an isolated copy of this case. It also requires a `steady-rans-fixed-point-assessment-v1` JSON document whose `accepted_as_steady_fixed_point` value is false. Prepare the physical-time restart from the rejected steady field without modifying that source case:

```bash
SOURCE_CASE=/path/to/rejected-steady-case SOURCE_TIME=6000 ./prepare_urans_restart.sh
```

The preparation step requires byte-identical `boundary/faces/neighbour/owner/points` files, refuses to replace an existing `0.002` directory, and creates the current and backward-history fields required by the transient scheme. The URANS runner then uses the validated 32-way decomposition and is restartable at completed stage boundaries:

```bash
NPROCS=32 \
STEADY_ASSESSMENT=/path/to/steady_fixed_point_assessment.json \
./run_urans_time_mean.sh
```

The fixed physical-time stages are:

- burn-in from `0.002` to `0.038056848 s`;
- stabilization from `0.038056848` to `0.0452682176 s`;
- extended burn-in from `0.0452682176` to `0.1209875984 s` after observed outlet/side-flux redistribution;
- clean continuous time averaging from `0.1209875984 s`, with a 20-flow-through minimum and an 80-flow-through hard cap.

All stages use adaptive timesteps with `maxCo=0.45`. Starting at 20 flow-through times, `watch_urans_acceptance.py` evaluates complete write checkpoints and requests a clean `nextWrite` stop only after two consecutive accepted checkpoints. Production staging remains fail-closed: `step30_urans_validation.py` rechecks the final checkpoint for Courant number, turbulence bounding, non-finite values, averaging duration, successive mean-field L2 change, pressure/viscous/total bucket-force stability, flux stability and imbalance, probe stability, and field sanity before a source bundle is created. The resulting semantics are explicitly `pimpleFoam / urans / time_mean / UMean`; no URANS mean is labelled as a steady `U` snapshot.
