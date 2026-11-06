# Step30 Participant Supplemental Data Package

Created on 2026-08-30. All artifacts in this package are stored under the H:
drive workspace. No local numerical simulation was run while assembling the
package.

## What is now available

- An open-access 2024 review with page-bound Pelton region, erodent hardness,
  test-rig and model-table evidence.
- An open 2021 JAFM paper with 0.1, 0.5 and 1.0 mm particle-size cases and
  Pelton distribution-part transport/erosion behavior.
- An IIT Roorkee thesis extract covering 13Cr-4Ni, 3D scanning, mass loss and
  laboratory erosion-velocity examples.
- A technical presentation documenting 13Cr-4Ni constitutive characterization
  context and a prototype-scale Pelton numerical case.
- An open Metals paper with 13Cr4Ni forged/WAAM hardness and tensile-property
  ranges, retained only as a same-family material prior.
- SHA-256 and byte-count bindings for every downloaded source.

The machine-readable register is
`step30_supplemental_data_manifest_v1.json`.

The frozen participant-only replay defaults are in
`step30_participant_default_scenarios_v1.json`. They make explicit the
density, hardness, quartz, particle-size, velocity, angle and exposure-mass
scenarios already used by the project. They are not measured DEC data.

## Data-use boundaries

The package supports a participant-built reference and preregistered model
comparison. It does not create official data, measured DEC wear truth, a
ZG0Cr13Ni4Mo batch certificate, or a formal FEM material card.

The following are allowed as conditional competition inputs:

1. qualitative region priors for splitter, cutout and near-outlet-like faces;
2. particle-size scenario cases and velocity sensitivity ranges;
3. same-family 13Cr-4Ni material priors;
4. secondary quantitative relations copied from the cited review, provided
   they remain tagged as secondary and stay in the leave-one-model-out
   comparison;
5. a synthetic participant baseline/worn depth envelope.

The following remain acquisition tasks:

- DEC-specific D10/D50/D90, mineral fraction and particle shape distribution;
- the exact ZG0Cr13Ni4Mo heat/batch certificate, heat treatment, hardness and
  density;
- an independent baseline/worn surface pair or registered sparse depth field;
- primary source PDFs for the Rai 2020 and Liu 2019 equations before promoting
  their coefficients beyond sensitivity-prior status.

## Remote validation command

Run the validator and any numerical replay on the DCU or user server only. The
local machine should only copy, inspect and hash artifacts under H:. The
existing official gates must remain unchanged:

```text
official_accuracy_gate=false
official_reference_gate=false
fem_gate=blocked
```

The package is intentionally fail-closed: a missing page locator, source hash,
or original text downgrades a claim rather than improving the score.
