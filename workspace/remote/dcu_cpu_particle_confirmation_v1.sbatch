#!/usr/bin/env bash
# Step30 server-only workload. Submit this file on the DCU server.
set -euo pipefail

: "${STEP30_EXECUTION_TARGET:?must be dcu_cpu}"
[[ "${STEP30_EXECUTION_TARGET}" == "dcu_cpu" ]] || {
  echo "ERROR: this workload is only valid for dcu_cpu" >&2
  exit 64
}
: "${STEP30_REMOTE_EXECUTION:=1}"
export STEP30_REMOTE_EXECUTION=1

: "${STEP30_REMOTE_ROOT:?remote project root is required}"
: "${STEP30_JOB_ID:?scheduler job id is required}"

cd "${STEP30_REMOTE_ROOT}/workspace"
export PYTHONHASHSEED=0
export STEP30_EXECUTION_TARGET=dcu_cpu

mkdir -p "${STEP30_REMOTE_ROOT}/results/${STEP30_JOB_ID}"
set +e
python3 step30_particle_confirmation_parallel_v1.py \
  --execution-target dcu_cpu \
  --workers "${SLURM_CPUS_PER_TASK:-1}" \
  --seed 57721 \
  --seed 104729 \
  --seed 130363 \
  --seed 161027 \
  --seed 196613 \
  --topology "${STEP30_REMOTE_ROOT}/workspace/output/step30_acceptance_0.8684190921/bucket_topology_v3_0.8684190921.npz" \
  --native-si-directory "${STEP30_REMOTE_ROOT}/workspace/output/native_bucket_si_urans_0.8684190921" \
  --reference-transform "${STEP30_REMOTE_ROOT}/workspace/output/authoritative_bucket_reference_transform_v1.json" \
  --discovery-report "${STEP30_REMOTE_ROOT}/workspace/output/step30_particle_convergence_5k_10k_20k_topology_aggregate_v5_discovery_r2.json"
status=$?
set -e

result_root="${STEP30_REMOTE_ROOT}/results/${STEP30_JOB_ID}"
cp -a "${STEP30_REMOTE_ROOT}/workspace/output/step30_v5_confirmation_parallel_v1" "${result_root}/"
cp -a "${STEP30_REMOTE_ROOT}/workspace/output/step30_runs" "${result_root}/"
printf 'job_id=%s\nexecution_target=dcu_cpu\nexit_code=%s\n' "${STEP30_JOB_ID}" "${status}" > "${result_root}/execution_status.txt"
(
  cd "${result_root}"
  find . -type f -print0 | sort -z | xargs -0 sha256sum > output_sha256sums.txt
)
exit "${status}"
