# FEM S0/S1 Readiness Contract

Status: `blocked`

Current local tool check:

- Gmsh `4.15.2` is available.
- meshio `5.3.5` and CadQuery/OCP are available.
- CalculiX 2.19 is installed outside the repository and passed the bundled
  `freq_test` solver smoke case; executable and package hashes are recorded in
  `fem_tool_provenance_ccx219.json`.
- Direct Parasolid import is blocked locally: this Gmsh build has no Parasolid
  support.
- A previous GPU/COMSOL export produced a local single-bucket solid STEP that
  Gmsh can open as 1 volume and 30 surfaces; see `output/step30_fem_remote_export_v1/`
  and `fem_geometry_conversion_v2.json`.
- That STEP still lacks independently confirmed `ROOT_INTERFACE`,
  `WETTED_SURFACE`, and `AI5_04` selections, so real FEM remains blocked.
- Official DEC AI5-04 traces, report metadata, E/nu, 375 rpm, and source hashes
  are frozen in `output/step30_fem_official_evidence_v1.json`; the manifest is
  intentionally `blocked` because it does not invent CAD coordinates, gauge
  vectors, STEP scale, or material density.
- Surface catalog for manual review is frozen in
  `output/step30_fem_remote_export_v1/step_surface_catalog.json`; it is a
  geometry catalog only, not a named-selection package.
- The required submission format for the missing CAD/material evidence is
  `templates/step30-fem-selection-material-intake-v1.json`, validated by
  `step30_fem_selection_intake.py`.

This checklist defines the shortest admissible path to the first real structural
solver evidence. Completing the checklist does not validate erosion spatial
accuracy and does not open an official gate.

## Frozen Inputs

| Input | Current state | Required action |
|---|---|---|
| Authoritative bucket geometry | Local single-bucket STEP from prior GPU/COMSOL export exists and opens in Gmsh as one solid volume | Confirm source binding, then add independently confirmed ROOT/WETTED/AI5 selections; STL is CFD-only fallback |
| Geometry SHA-256 | Available through existing provenance | Freeze the exact STEP/STL bytes used by Gmsh |
| Material E and Poisson ratio | DEC supports 206 GPa and 0.3 | Bind DEC source pages and raw PDF SHA-256 |
| Material density | Engineering assumption only | Obtain design/material-certificate source before formal S1 |
| Rotation | 375 rpm for S1 | Freeze axis, origin, sign and units |
| AI5-04 measurement | 13 official raw traces and report metadata are hash-bound in `step30_fem_official_evidence_v1.json` | Bind CAD patch and gauge direction(s); use the trace only as independent validation, never calibration |
| AI5-04 position/directions | Physical semantics are confirmed (4# bucket root / water-divider extension line), but numeric CAD patch and directions are absent | Submit `AI5_04` selection, numeric patch center, coordinate frame, and one/two unit directions |
| Structural solver | CalculiX 2.19 installed and smoke-tested | Bind `fem_tool_provenance_ccx219.json` to every real case; smoke test is not project FEM evidence |

## Required Physical Groups

- `MATERIAL_DOMAIN`: all and only solid C3D10 elements.
- `ROOT_INTERFACE`: verified bucket/runner attachment interface.
- `WETTED_SURFACE`: wetted faces only; must exclude the root closure.
- `AI5_04`: local gauge patch outside the fully constrained node set.

Physical group tags may not be inferred from Gmsh tag numbers alone. They require
CAD/named-selection confirmation. `AI5_04` must not be calibrated from the same
measurement used for validation.

## Immutable Three-Mesh Package

Prepare exactly three mesh identities. The characteristic sizes are selected only
after the authoritative solid and physical groups are available; do not invent
absolute sizes from the STL.

| Mesh ID | Role | Required relation |
|---|---|---|
| `S0-C` | coarse qualification mesh | largest effective size |
| `S0-M` | medium qualification mesh | `h_C / h_M >= 1.2` |
| `S0-F` | fine qualification mesh | `h_M / h_F >= 1.2` |

For every mesh ID, freeze the following in a machine-readable manifest before a
solver run:

- source STEP/BREP path, SHA-256, coordinate units, and geometry volume;
- Gmsh executable version/SHA-256 plus `.geo` or API driver SHA-256;
- meshing options, target size, element order, optimization method, and command;
- `.msh` path/SHA-256, node count, C3D10 element count, effective size, mesh
  volume, minimum quality, and negative-Jacobian count;
- physical-group element/face/node counts and pairwise overlap counts;
- generated CalculiX `.inp` path/SHA-256 and case-provenance SHA-256.

The three mesh hashes, effective sizes, and element counts must be distinct.
Changing geometry, named selections, units, material data, rotation definition,
or the mesh driver invalidates all downstream S0/S1 identities.

## S0 Mesh Qualification

Generate coarse, medium and fine straight-sided C3D10 meshes using Gmsh and
Netgen optimization.

Gates:

- authoritative geometry and mesh SHA-256 bound;
- one solid material domain;
- required physical groups present and nonempty;
- root and wetted groups do not overlap;
- AI5-04 nodes are not fixed nodes;
- zero negative Jacobians;
- minimum element quality at least 0.01;
- mesh/geometry volume difference below 1% for pilot, target below 0.5%;
- adjacent effective mesh-size ratio at least 1.2.

Until these pass: `mesh_qualification_gate = blocked`.

S0 exit requires all three immutable manifests and every gate above to pass. A
generated mesh or INP alone is software/case-preparation evidence, not a real FEM
result.

## S1 Centrifugal 375 rpm

Use only centrifugal loading for the first real solver sanity check. Do not use
CFD pressure or the AI5 result for calibration.

Required evidence:

- immutable CalculiX INP and mesh files;
- `ccx` executable/version SHA-256;
- normal solver termination and finite displacements/stresses;
- root reaction force and moment balance;
- three-mesh convergence for AI5 projected stress and hotspot stress;
- root-boundary sensitivity case;
- AI5 projected strain/stress evaluated in the frozen gauge directions.

Run IDs are `S1-C`, `S1-M`, and `S1-F`, each bound one-to-one to `S0-C`, `S0-M`,
and `S0-F`. Each result package must contain the immutable input, normal solver
termination record, finite displacement/stress export, root reaction vector and
moment, AI5 projection inputs/outputs, and all file SHA-256 values required by
`step30_fem_results.py`.

S1 may start only when:

- S0 has passed for all three meshes;
- density has an admissible design/material source;
- rotation axis, origin, sign, speed, and units are frozen;
- one `ccx` executable/version/SHA-256 is frozen for all three runs;
- AI5-04 CAD location and gauge directions are independently frozen and were not
  used to tune material, boundary, or load parameters.

Interpretation:

- AI5 relative error <= 10%: engineering sanity check only;
- AI5 relative error <= 5% plus mesh/boundary gates: strong structural evidence;
- neither result proves erosion spatial error <= 5%.

Until a real solver result exists: `fem_gate = blocked`.

## Latest Stage A Verification (2026-08-10)

- Actual mounted snapshot: ``
- Full regression with isolated `vtk`/`meshio`/`gmsh` dependencies: `249 passed, 1 skipped`
- Required FEM module `py_compile`: passed
- Official evidence manifest validation: `blocked` because its frozen `official_package_root` points to the unavailable historical `B:` path
- Unfilled CAD/material selection package: `blocked`
- No S0 mesh qualification or S1 solver run was started; `fem_gate` remains `blocked`

S1 exit additionally requires the three real solver packages to pass
`step30_fem_results.py` and `step30_fem_validation.py`. A valid numerical result
above the preregistered AI5 or mesh-convergence threshold is `failed`; missing or
invalid identity, provenance, units, named selections, solver evidence, or hashes
is `blocked`.

## Deferred Pressure Cases

`S2_pmean_only_diagnostic` and `S3_combined_diagnostic` may start only after the
VTP quadrilateral triangulation, wall-normal orientation, root-face exclusion,
and conservative force/moment mapping gates pass. Existing `pMean` is a
stationary, submerged, single-bucket diagnostic load, not rotating-runner VOF
pressure and not a dynamic stress-amplitude load.

## Literature-Prior and Geometry-Audit Addendum (2026-08-10)

The first exploratory package is recorded in
`output/step30_literature_fem_provisional_manifest_v1.json`.

The literature-prior module is a comparison and uncertainty-propagation tool,
not an accuracy claim. It uses Oka, Finnie, Tabakoff, and Mansouri formulas on
the frozen particle-hit artifact and propagates density, hardness, and sediment
concentration priors. Its status is `provisional`; `fem_gate` remains
`blocked`.

The geometry audit confirms that the authoritative STEP is one solid with 30
boundary surfaces in Gmsh and that the STEP text declares metres. The COMSOL
export script also requests metres. However, the Gmsh-native coordinate/metric
scale is not reconciled by one uniform scale within the audit tolerance, so
the formal coordinate-unit gate is `blocked`. Gmsh masses are therefore
reported as native reader values, without relabelling them as SI volume or
area.

No ROOT_INTERFACE or WETTED_SURFACE physical selection is promoted. No AI5-04
numeric patch or gauge direction is invented. Raw surface tags, bounding-box
heuristics, and visual guesses remain inadmissible.
