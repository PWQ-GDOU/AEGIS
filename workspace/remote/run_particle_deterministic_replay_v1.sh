#!/usr/bin/env bash
# Step30 remote-only deterministic replay smoke test.
# It creates two isolated workspaces and never changes historical artifacts.
set -euo pipefail

: "${STEP30_REMOTE_ROOT:?set the remote project root}"
: "${STEP30_JOB_ID:?set a fresh replay job id}"
: "${STEP30_EXECUTION_TARGET:?set dcu_cpu or user_gpu}"
: "${STEP30_REMOTE_EXECUTION:?set to 1 on the remote host}"
[[ "${STEP30_REMOTE_EXECUTION}" == "1" ]] || { echo "SAFETY_REFUSAL: remote flag missing" >&2; exit 64; }
[[ "${STEP30_EXECUTION_TARGET}" == "dcu_cpu" ]] || { echo "SAFETY_REFUSAL: replay is registered for dcu_cpu only" >&2; exit 64; }
[[ "${STEP30_JOB_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "SAFETY_REFUSAL: invalid job id" >&2; exit 64; }

source_root="${STEP30_REMOTE_ROOT}/workspace"
replay_root="${STEP30_REMOTE_ROOT}/workspace/output/step30_deterministic_replay/${STEP30_JOB_ID}"
report="${replay_root}/replay_report.json"
if [[ -e "${replay_root}" ]]; then
  echo "SAFETY_REFUSAL: replay root already exists: ${replay_root}" >&2
  exit 64
fi
mkdir -p "${replay_root}"

for replica in A B; do
  root="${replay_root}/replica_${replica}"
  mkdir -p "${root}/output"
  for source in step30_lagrangian_particle_tracking.py step30_cfd_topology.py step30_artifact_identity.py; do
    cp -p "${source_root}/${source}" "${root}/${source}"
  done
  # Inputs are shared read-only through symlinks; all generated outputs stay in
  # the replica-specific output tree.
  ln -s "${source_root}/output/step30_acceptance_0.8684190921" "${root}/output/step30_acceptance_0.8684190921"
  ln -s "${source_root}/output/native_bucket_si_urans_0.8684190921" "${root}/output/native_bucket_si_urans_0.8684190921"
  ln -s "${source_root}/output/authoritative_bucket_reference_transform_v1.json" "${root}/output/authoritative_bucket_reference_transform_v1.json"
  ln -s "${source_root}/output/vertices_full_gpr.npz" "${root}/output/vertices_full_gpr.npz"
done

run_replica() {
  local replica="$1"
  local root="${replay_root}/replica_${replica}"
  local state="${replay_root}/state_${replica}"
  mkdir -p "${state}/pycache" "${state}/tmp" "${state}/mplconfig"
  cd "${root}"
  export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
  export PYTHONPYCACHEPREFIX="${state}/pycache"
  export TEMP="${state}/tmp" TMP="${state}/tmp" TMPDIR="${state}/tmp" MPLCONFIGDIR="${state}/mplconfig"
  export STEP30_EXECUTION_TARGET=dcu_cpu STEP30_REMOTE_EXECUTION=1
  remote_artifacts/PWQ/step30_cpu_v1/.venv/bin/python step30_lagrangian_particle_tracking.py \
    --cfd native_si \
    --collision-backend cfd_topology \
    --collision-policy strict_v2 \
    --topology output/step30_acceptance_0.8684190921/bucket_topology_v3_0.8684190921.npz \
    --native-si-directory output/native_bucket_si_urans_0.8684190921 \
    --field-time 0.8684190921 \
    --analytic-reference-transform authoritative_bucket_reference_transform_v1.json \
    --immutable-output \
    --parcels "${STEP30_REPLAY_PARCELS:-1000}" \
    --seed "${STEP30_REPLAY_SEED:-57721}" \
    --max-impacts 3 \
    --mass-flow nominal \
    --psd-sampling mass_stratified_v1 \
    --max-time 0.05 \
    --dt 1e-5 \
    --particle-cfl 0.25 \
    --strict-max-substep 1e-5 \
    --cfd-interpolation idw_k8_v1 \
    --particle-integrator midpoint_exponential_v2 \
    --max-cfd-speed 500 \
    > "${state}/stdout.log" 2> "${state}/stderr.log"
}

run_replica A & pid_a=$!
run_replica B & pid_b=$!
set +e
wait "${pid_a}"; status_a=$?
wait "${pid_b}"; status_b=$?
set -e

find_replica_run() {
  find "${replay_root}/replica_$1/output/step30_runs" -type f -name particle_run_manifest.json -print 2>/dev/null | head -n 1 | xargs -r dirname
}
run_a="$(find_replica_run A)"
run_b="$(find_replica_run B)"
if [[ -z "${run_a}" || -z "${run_b}" ]]; then
  echo "replay output missing" >&2
  exit 3
fi

set +e
cmp -s "${run_a}/particle_hits.npz" "${run_b}/particle_hits.npz"; npz_equal=$?
cmp -s "${run_a}/summary.json" "${run_b}/summary.json"; summary_equal=$?
cmp -s "${run_a}/particle_run_manifest.json" "${run_b}/particle_run_manifest.json"; manifest_equal=$?
set -e

python3 - "${report}" "${run_a}" "${run_b}" "${status_a}" "${status_b}" "${npz_equal}" "${summary_equal}" "${manifest_equal}" <<'PY'
import hashlib, json, sys
from pathlib import Path

report_path, run_a, run_b, status_a, status_b, npz_equal, summary_equal, manifest_equal = sys.argv[1:]
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
summary_a = load(Path(run_a) / 'summary.json')
summary_b = load(Path(run_b) / 'summary.json')
report = {
    'version': 'step30-particle-deterministic-replay-v1',
    'status': 'passed' if status_a == status_b == '0' and npz_equal == summary_equal == manifest_equal == '0' else 'failed',
    'execution_target': 'dcu_cpu',
    'run_a': run_a,
    'run_b': run_b,
    'process_status': {'A': int(status_a), 'B': int(status_b)},
    'byte_equal': {'particle_hits_npz': npz_equal == '0', 'summary_json': summary_equal == '0', 'particle_run_manifest_json': manifest_equal == '0'},
    'sha256': {name: {'A': digest(Path(run_a) / name), 'B': digest(Path(run_b) / name)} for name in ('particle_hits.npz', 'summary.json', 'particle_run_manifest.json')},
    'summary_differences': {
        'tracking': summary_a.get('tracking') != summary_b.get('tracking'),
        'face_erosion': summary_a.get('face_erosion') != summary_b.get('face_erosion'),
        'validation': summary_a.get('validation') != summary_b.get('validation'),
    },
    'interpretation': 'Byte-identical replay supports deterministic implementation at this smoke scale; it does not prove spatial convergence or physical accuracy.'
}
Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

sha256sum "${report}" > "${report}.sha256"
printf 'job_id=%s\nstatus=%s\nreport=%s\n' "${STEP30_JOB_ID}" "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${report}")" "${report}"
exit 0
