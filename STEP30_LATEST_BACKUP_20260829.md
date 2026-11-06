# Step30 Latest Backup, 2026-08-29

This branch preserves the latest Step30 scripts, auditable JSON evidence, and
the DCU-generated v5 particle confirmation package without changing the
existing project history or formal gate conclusions.

## Scope

- Source scripts and focused tests for particle tracking, topology-connected
  surface estimation, literature priors, material pilot protocol, measured
  baseline/worn intake, geometry audit, and exploratory FEM.
- Versioned JSON reports and manifests that are small enough to review in Git.
- DCU-generated v5 confirmation report, five topology-aggregate fields, and
  their remote SHA-256 manifest under `workspace/output/remote_results/`.
- Large binary data use Git LFS. Generated caches, solver logs, installers,
  and unrelated source archives are excluded.

## Evidence status

The v5 particle result is a numerically stable coarse risk map. It does not
open `spatial_accuracy_status`, `official_accuracy_gate`, `official_reference_gate`,
or `fem_gate`. A baseline/worn scan pair or equivalent independent measured
truth remains required for physical spatial accuracy.

## Credential policy

Local session context and connection material are intentionally excluded. The
repository history previously contained sensitive session material; access
tokens and server passwords must be revoked and rotated outside this commit.
