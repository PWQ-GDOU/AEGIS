# Repository Path Portability Policy

All operational project references are repository-relative so the project can be
cloned to any drive or server directory without source edits.

- Local files use relative paths such as `workspace/output/...` and
  `DecPelton2024/...`.
- Compute-server provenance uses the relative `remote_artifacts/...` namespace.
  It identifies the artifact's source without claiming it is a local path.
- Historical external locations use `historical_external/...` or
  `external_workspaces/...`; they are provenance labels, not execution targets.
- Remote submission scripts discover their working directory through `$HOME`,
  `/PWQ`, and environment variables at runtime. These are deployment probes,
  not persisted project artifact paths.
- URLs, DOIs, JSON Schema identifiers, and regular expressions are not file
  paths and are intentionally retained.

`workspace/tools/normalize_repo_paths.py` performs the normalization pass for
UTF-8 text artifacts. It excludes test fixtures that deliberately contain
absolute paths to verify fail-closed validation behavior.
