#!/usr/bin/env bash
# Step30 remote-only data audit entry point. No numerical work runs locally.
set -euo pipefail

: "${STEP30_EXECUTION_TARGET:?set dcu_cpu or user_gpu}"
: "${STEP30_REMOTE_EXECUTION:?set to 1 on the remote host}"
: "${STEP30_JOB_ID:?set a fresh audit job id}"
[[ "${STEP30_REMOTE_EXECUTION}" == "1" ]] || { echo "SAFETY_REFUSAL: remote execution flag missing" >&2; exit 64; }
[[ "${STEP30_EXECUTION_TARGET}" == "dcu_cpu" || "${STEP30_EXECUTION_TARGET}" == "user_gpu" ]] || { echo "SAFETY_REFUSAL: unknown execution target" >&2; exit 64; }

workspace_root="${STEP30_REMOTE_ROOT:-$(pwd)}"
cd "${workspace_root}/workspace"
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export TEMP="${workspace_root}/workspace/output/.audit_tmp_${STEP30_JOB_ID}"
export TMP="${TEMP}"
export TMPDIR="${TEMP}"
export MPLCONFIGDIR="${TEMP}/mplconfig"
mkdir -p "${TEMP}" "${MPLCONFIGDIR}"

run_root="${STEP30_RUN_ROOT:-${workspace_root}/workspace/output/step30_runs}"
output_path="${STEP30_AUDIT_OUTPUT:-${workspace_root}/workspace/output/step30_particle_data_audit_${STEP30_JOB_ID}.json}"
mkdir -p "$(dirname "${output_path}")"
if [[ -e "${output_path}" ]]; then
  echo "SAFETY_REFUSAL: audit output already exists: ${output_path}" >&2
  exit 64
fi

python3 step30_particle_data_audit_v1.py \
  --run-root "${run_root}" \
  --output "${output_path}"

sha256sum "${output_path}" > "${output_path}.sha256"
printf 'job_id=%s\nexecution_target=%s\nrun_root=%s\naudit_output=%s\n' \
  "${STEP30_JOB_ID}" "${STEP30_EXECUTION_TARGET}" "${run_root}" "${output_path}"
