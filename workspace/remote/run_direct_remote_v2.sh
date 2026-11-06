#!/usr/bin/env bash
# Explicit direct-remote fallback for hosts without Slurm/PBS.
set -euo pipefail

: "${STEP30_REMOTE_ROOT:?remote project root is required}"
: "${STEP30_JOB_ID:?fresh job id is required}"
: "${STEP30_ALLOW_DIRECT_REMOTE:?set this explicitly to 1}"
[[ "${STEP30_ALLOW_DIRECT_REMOTE}" == "1" ]] || {
  echo "SAFETY_REFUSAL: STEP30_ALLOW_DIRECT_REMOTE must equal 1" >&2
  exit 64
}
[[ "${STEP30_JOB_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "SAFETY_REFUSAL: invalid job id" >&2
  exit 64
}

cd "${STEP30_REMOTE_ROOT}/workspace"
export STEP30_EXECUTION_TARGET=dcu_cpu
export STEP30_REMOTE_EXECUTION=1

state_root="${STEP30_REMOTE_ROOT}/workspace/output/step30_v2_state/${STEP30_JOB_ID}"
result_root="${STEP30_REMOTE_ROOT}/results/${STEP30_JOB_ID}"
if [[ -e "${result_root}" ]]; then
  echo "SAFETY_REFUSAL: result root already exists; use a fresh job id" >&2
  exit 64
fi
mkdir -p "${result_root}"

nohup python3 step30_particle_confirmation_parallel_v2.py \
  --execution-target dcu_cpu \
  --allow-direct-remote \
  --workers "${STEP30_WORKERS:-5}" \
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
  2> "${result_root}/runner.stderr.log" < /dev/null &
pid=$!
printf '%s\n' "${pid}" > "${result_root}/runner.pid"
printf 'job_id=%s\nexecution_target=dcu_cpu\nmode=direct_remote\npid=%s\n' "${STEP30_JOB_ID}" "${pid}" > "${result_root}/execution_status.txt"
printf '%s\n' "${result_root}"
