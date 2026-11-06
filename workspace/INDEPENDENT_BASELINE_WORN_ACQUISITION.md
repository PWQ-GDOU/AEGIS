# Independent Baseline/Worn Acquisition

Status: `blocked`

This is the delivery checklist for the real erosion spatial truth. The
literature package in
`output/step30_literature_reference_distribution_v1.json` is only a
`literature_reference` prior. It does not replace this acquisition.

## Required package

For each independent asset/service episode, provide:

- baseline scan of the unworn or pre-service bucket;
- worn scan of the same bucket after the frozen service episode;
- scanner serial number and calibration certificate;
- acquisition timestamps and service start/end timestamps;
- common coordinate frame, fiducials, fixture identity, and coordinate units;
- immutable source SHA-256 and byte count for every raw file;
- scan uncertainty or point-spacing report;
- registration transform, coverage, residuals, and rejection thresholds;
- topology-v3 parent-face mapping and the exact source mesh hash.

Use at least three independent asset/service-episode groups for a pilot and
five groups for a formal candidate. Split assignment is identity-only and must
be frozen before prediction reveal.

## Acceptance boundary

The measured chain is:

```text
baseline/worn raw files
  -> registration
  -> surface difference or sparse thickness
  -> topology-v3 parent-face mapping
  -> preregistered independent-test mask
  -> reveal receipt
  -> independent replay
```

Do not classify literature figures, synthetic fixtures, DEC AI5 traces, CFD
fields, or guessed CAD regions as measured wear. Missing files, units,
identity, hashes, topology, or coverage remain `blocked`; a valid measured
number above the frozen 5% threshold is `failed`.

The shipped intake template is
`templates/step30-independent-baseline-worn-acquisition-v1.json`, and the
machine-readable scan manifest is
`templates/step30-scan-pair-manifest-v1.json`.
