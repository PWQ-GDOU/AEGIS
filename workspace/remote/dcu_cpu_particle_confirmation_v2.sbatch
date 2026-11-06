#!/usr/bin/env bash
# Step30 v2 safety wrapper. Numerical work is server-side only.
set -euo pipefail

: "${STEP30_EXECUTION_TARGET:?must be dcu_cpu}"
[[ "${STEP30_EXECUTION_TARGET}" == "dcu_cpu" ]] || {
  echo "SAFETY_REFUSAL: this workload is only valid for dcu_cpu" >&2
  exit 64
}
: "${STEP30_REMOTE_ROOT:?remote project root is required}"
: "${STEP30_JOB_ID:?scheduler job id is required}"

cd "${STEP30_REMOTE_ROOT}/workspace"
export PYTHONHASHSEED=0
export STEP30_EXECUTION_TARGET=dcu_cpu
export STEP30_REMOTE_EXECUTION=1

state_root="${STEP30_REMOTE_ROOT}/workspace/output/step30_v2_state/${STEP30_JOB_ID}"
result_root="${STEP30_REMOTE_ROOT}/results/${STEP30_JOB_ID}"
if [[ -e "${result_root}" ]]; then
  echo "SAFETY_REFUSAL: result root already exists; use a fresh job id" >&2
  exit 64
fi
mkdir -p "${result_root}"

set +e
python3 step30_particle_confirmation_parallel_v2.py \
  --execution-target dcu_cpu \
  --workers "${SLURM_CPUS_PER_TASK:-${PBS_NP:-1}}" \
  --seed 510001 \
  --seed 510003 \
  --seed 510009 \
  --seed 510021 \
  --seed 510023 \
  --parcels "${STEP30_PARCELS:-60000}" \
  --max-runtime-s "${STEP30_MAX_RUNTIME_S:-7200}" \
  --min-free-gb "${STEP30_MIN_FREE_GB:-10}" \
  --state-root "${state_root}" \
  --topology "${STEP30_REMOTE_ROOT}/workspace/output/step30_acceptance_0.8684190921/bucket_topology_v3_0.8684190921.npz" \
  --native-si-directory "${STEP30_REMOTE_ROOT}/workspace/output/native_bucket_si_urans_0.8684190921" \
  --reference-transform "${STEP30_REMOTE_ROOT}/workspace/output/authoritative_bucket_reference_transform_v1.json" \
  --discovery-report "${STEP30_REMOTE_ROOT}/workspace/output/step30_particle_convergence_5k_10k_20k_topology_aggregate_v5_discovery_r2.json" \
  > "${result_root}/runner.stdout.log" \
  2> "${result_root}/runner.stderr.log"
status=$?
set -e

if [[ -d "${state_root}" ]]; then
  cp -a "${state_root}" "${result_root}/state"
fi
if [[ -d "${STEP30_REMOTE_ROOT}/workspace/output/step30_runs" ]]; then
  cp -a "${STEP30_REMOTE_ROOT}/workspace/output/step30_runs" "${result_root}/"
fi
printf 'job_id=%s\nexecution_target=dcu_cpu\nexit_code=%s\nstate_root=%s\n' \
  "${STEP30_JOB_ID}" "${status}" "${state_root}" > "${result_root}/execution_status.txt"
(
  cd "${result_root}"
  find . -type f -print0 | sort -z | xargs -0 sha256sum > output_sha256sums.txt
)
exit "${status}"
