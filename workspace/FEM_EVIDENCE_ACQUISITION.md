# Real FEM Evidence Acquisition

This document describes the exact evidence needed to move the structural FEM gate from `blocked` to S0 mesh preparation. The recovered STEP and local CalculiX executable are already available, but the following inputs must be independently frozen before a solver result can be called real FEM evidence.

## Already Frozen

The official DEC evidence manifest is:

`output/step30_fem_official_evidence_v1.json`

It binds 16 original DEC artifacts by SHA-256:

- the official dynamic-stress report;
- the official dynamic-stress data note;
- the official geometry-file note;
- 13 raw AI5-04 time-series files.

The manifest supports these claims:

- channel: `AI5-04`;
- physical semantics: 4# bucket root, on the water-divider extension line, identified as the simulated maximum-stress location;
- stress unit: `MPa`;
- sampling frequency: `1000 Hz`;
- material `E=206 GPa`, `nu=0.3`;
- S1 operating speed: `375 rpm`.

These claims do not provide a numeric CAD point, gauge direction vector, physical boundary selection, or material density.

## 1. CAD Selection Package

Use the recovered source geometry:

`output/step30_fem_remote_export_v1/output/single_bucket_target_authoritative.step`

Its current SHA-256 is:

`521fbaf7688ad1f0e063dedb08c360135026ce60e5e371b4a196294b18cfe21d`

In COMSOL, NX, Ansys, or another CAD tool that can retain topology, create and export a named-selection report for all four selections:

- `ROOT_INTERFACE`: the bucket/runner attachment interface to be fixed in S1;
- `WETTED_SURFACE`: wetted bucket faces only, excluding the root closure;
- `AI5_04`: the strain-gauge patch/face set;
- `MATERIAL_DOMAIN`: the complete solid volume.

The submission must contain:

- the source STEP path and SHA-256;
- coordinate units (`m` or `mm`) and the coordinate frame;
- entity dimension and entity tags for every selection;
- a CAD selection export or log and a screenshot/report that independently identifies the physical meaning;
- explicit confirmation that `ROOT_INTERFACE` and `WETTED_SURFACE` do not overlap;
- explicit confirmation that the AI5 patch is outside the fixed root set;
- the exported selection artifact's SHA-256.

Do not submit a list copied from `step_surface_catalog.json` without independent physical confirmation. That catalog deliberately labels raw surface tags as identifiers only.

## 2. AI5-04 Patch and Gauge Direction

The official report identifies the location semantically, but the FEM projection requires numeric data. Supply:

- AI5 patch center coordinates in the STEP frame;
- patch face IDs or node IDs;
- coordinate frame name and units;
- one or two unit gauge directions, depending on the instrument interpretation;
- the direction of each strain-gauge axis, not only the sensor name;
- evidence that the directions are not fitted to an FEM result.

Use this exact machine-readable structure in `templates/step30-fem-selection-material-intake-v1.json`:

```json
"AI5_04": {
  "status": "passed",
  "confirmation_method": "independent_cad_review",
  "independent_confirmation": true,
  "entity_dimension": 2,
  "entity_tags": [101, 102],
  "patch_center": {
    "coordinates": [0.0, 0.0, 0.0],
    "units": "m"
  },
  "direction_frame": "step_global",
  "gauge_directions": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
  "outside_fixed_root": true,
  "evidence_files": [
    {"path": "absolute/path/to/cad-selection-report.json", "sha256": "...", "bytes": 1234}
  ]
}
```

The example values are placeholders and must not be used as project inputs.

## 3. Material Density Package

The official DEC package found here supports `E=206 GPa` and `nu=0.3`, but it does not contain an admissible runner-material density source. Submit either:

- a material certificate bound to grade and heat/batch; or
- an official design/material record that explicitly states density.

Required fields:

- material grade;
- heat or batch identifier;
- density in `kg/m3`;
- source kind: `material_certificate` or `official_design_record`;
- source file path, byte count, and SHA-256.

Do not use:

- water density `1000 kg/m3`;
- the synthetic/default `7850 kg/m3` value;
- a generic web material table without grade/heat identity;
- a material trial artifact that is marked synthetic or lacks a certificate.

## 4. STEP Unit Probe

The COMSOL export script declares SI meters, while some catalog magnitudes look like millimeter-scale coordinates. Before meshing, perform an independent unit probe:

1. Open the exact STEP whose SHA-256 is frozen above.
2. Record bounding box, volume, and one known dimensional reference from the CAD/report.
3. State whether the solver input is interpreted as `m` or `mm`.
4. Hash the probe report.

The unit probe must agree with the selection package and the Gmsh mesh manifest.

## 5. Automatic Validation

Fill the template and run:

```text
python step30_fem_selection_intake.py --package path/to/completed-package.json
```

The command returns `ready_for_s0` only if all required fields, source hashes, selections, non-overlap constraints, AI5 directions, and density source pass. Otherwise it returns `blocked`.

After `ready_for_s0`, the next steps are:

1. generate immutable S0-C/M/F C3D10 meshes;
2. qualify groups, volume, quality, boundary ownership, and refinement ratios;
3. generate pressure-free S1-C/M/F cases at exactly 375 rpm;
4. run the same CalculiX executable for all three meshes;
5. parse native solver outputs and validate reactions, AI5 tensor projections, and convergence.

A generated mesh or CalculiX input deck alone is preparation evidence. It is not a real FEM result.
