#!/usr/bin/env bash
# Intentionally fail-closed: no validated GPU particle backend is registered.
set -euo pipefail

: "${STEP30_EXECUTION_TARGET:?must be user_gpu}"
[[ "${STEP30_EXECUTION_TARGET}" == "user_gpu" ]] || {
  echo "ERROR: this workload is only valid for user_gpu" >&2
  exit 64
}

echo "BLOCKED: the current tracker is CPU/NumPy/SciPy and has no validated GPU backend." >&2
echo "Provide and validate a GPU backend before submitting any user_gpu numerical job." >&2
exit 78

