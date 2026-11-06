#!/usr/bin/env bash
# Step30 remote-only performance A/B for strict-v2 near-miss diagnostics.
set -euo pipefail

: "${STEP30_REMOTE_ROOT:?set the remote project root}"
: "${STEP30_JOB_ID:?set a fresh performance A/B job id}"
: "${STEP30_EXECUTION_TARGET:?set dcu_cpu or user_gpu}"
: "${STEP30_REMOTE_EXECUTION:?set to 1 on the remote host}"
[[ "${STEP30_REMOTE_EXECUTION}" == "1" ]] || { echo "SAFETY_REFUSAL: remote flag missing" >&2; exit 64; }
[[ "${STEP30_EXECUTION_TARGET}" == "dcu_cpu" ]] || { echo "SAFETY_REFUSAL: A/B is registered for dcu_cpu only" >&2; exit 64; }
[[ "${STEP30_JOB_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "SAFETY_REFUSAL: invalid job id" >&2; exit 64; }

source_root="${STEP30_REMOTE_ROOT}/workspace"
ab_root="${source_root}/output/step30_particle_performance_ab/${STEP30_JOB_ID}"
report="${ab_root}/performance_ab_report.json"
if [[ -e "${ab_root}" ]]; then
  echo "SAFETY_REFUSAL: A/B root already exists: ${ab_root}" >&2
  exit 64
fi
mkdir -p "${ab_root}"

for variant in baseline optimized; do
  root="${ab_root}/${variant}"
  mkdir -p "${root}/output"
  for source in step30_lagrangian_particle_tracking.py step30_cfd_topology.py step30_artifact_identity.py; do
    cp -p "${source_root}/${source}" "${root}/${source}"
  done
  ln -s "${source_root}/output/step30_acceptance_0.8684190921" "${root}/output/step30_acceptance_0.8684190921"
  ln -s "${source_root}/output/native_bucket_si_urans_0.8684190921" "${root}/output/native_bucket_si_urans_0.8684190921"
  ln -s "${source_root}/output/authoritative_bucket_reference_transform_v1.json" "${root}/output/authoritative_bucket_reference_transform_v1.json"
  ln -s "${source_root}/output/vertices_full_gpr.npz" "${root}/output/vertices_full_gpr.npz"
done

run_variant() {
  local variant="$1"
  local root="${ab_root}/${variant}"
  local state="${ab_root}/state_${variant}"
  mkdir -p "${state}/pycache" "${state}/tmp" "${state}/mplconfig"
  cd "${root}"
  export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
  export PYTHONPYCACHEPREFIX="${state}/pycache"
  export TEMP="${state}/tmp" TMP="${state}/tmp" TMPDIR="${state}/tmp" MPLCONFIGDIR="${state}/mplconfig"
  export STEP30_EXECUTION_TARGET=dcu_cpu STEP30_REMOTE_EXECUTION=1
  local extra=()
  if [[ "${variant}" == "optimized" ]]; then
    extra+=(--disable-strict-near-miss-distance-diagnostics)
  fi
  /usr/bin/time -f '%e' -o "${state}/wall_time_seconds.txt" \
    remote_artifacts/PWQ/step30_cpu_v1/.venv/bin/python step30_lagrangian_particle_tracking.py \
      --cfd native_si --collision-backend cfd_topology --collision-policy strict_v2 \
      --topology output/step30_acceptance_0.8684190921/bucket_topology_v3_0.8684190921.npz \
      --native-si-directory output/native_bucket_si_urans_0.8684190921 --field-time 0.8684190921 \
      --analytic-reference-transform authoritative_bucket_reference_transform_v1.json --immutable-output \
      --parcels "${STEP30_AB_PARCELS:-1000}" --seed "${STEP30_AB_SEED:-57721}" --max-impacts 3 \
      --mass-flow nominal --psd-sampling mass_stratified_v1 --max-time 0.05 --dt 1e-5 \
      --particle-cfl 0.25 --strict-max-substep 1e-5 --cfd-interpolation idw_k8_v1 \
      --particle-integrator midpoint_exponential_v2 --max-cfd-speed 500 "${extra[@]}" \
      > "${state}/stdout.log" 2> "${state}/stderr.log"
}

# Sequential execution avoids CPU contention in the wall-clock comparison.
run_variant baseline
run_variant optimized

find_run() {
  find "${ab_root}/$1/output/step30_runs" -type f -name particle_run_manifest.json -print 2>/dev/null | head -n 1 | xargs -r dirname
}
run_baseline="$(find_run baseline)"
run_optimized="$(find_run optimized)"
if [[ -z "${run_baseline}" || -z "${run_optimized}" ]]; then
  echo "A/B output missing" >&2
  exit 3
fi

set +e
cmp -s "${run_baseline}/particle_hits.npz" "${run_optimized}/particle_hits.npz"; npz_equal=$?
set -e

python3 - "${report}" "${run_baseline}" "${run_optimized}" "${npz_equal}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

report_path, run_baseline, run_optimized, npz_equal = sys.argv[1:]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def tracked(summary):
    tracking = summary.get('tracking', {})
    face = summary.get('face_erosion', {})
    return {
        'n_impacts': tracking.get('n_impacts'),
        'intersection_hits': tracking.get('intersection_hits'),
        'fallback_hits': tracking.get('fallback_hits'),
        'hit_parcels': tracking.get('hit_parcels'),
        'total_injection_mass_kg': tracking.get('total_injection_mass_kg'),
        'impact_mass_kg': tracking.get('impact_mass_kg'),
        'substep_count': tracking.get('substep_count'),
        'termination_counts': tracking.get('termination_counts'),
        'face_erosion_total': face.get('total_erosion_mass_kg'),
    }

base_dir = Path(run_baseline)
opt_dir = Path(run_optimized)
base_summary = load(base_dir / 'summary.json')
opt_summary = load(opt_dir / 'summary.json')
base_tracking = tracked(base_summary)
opt_tracking = tracked(opt_summary)
physics_equal = base_tracking == opt_tracking
root = Path(report_path).parent
base_time = float((root / 'state_baseline' / 'wall_time_seconds.txt').read_text().strip())
opt_time = float((root / 'state_optimized' / 'wall_time_seconds.txt').read_text().strip())
report = {
    'version': 'step30-particle-performance-ab-v1',
    'status': 'passed' if npz_equal == '0' and physics_equal else 'failed',
    'execution_target': 'dcu_cpu',
    'job_id': root.name,
    'variants': {
        'baseline': {
            'strict_near_miss_distance_diagnostics': True,
            'run_dir': str(base_dir),
            'wall_time_seconds': base_time,
            'particle_hits_sha256': digest(base_dir / 'particle_hits.npz'),
            'tracking': base_tracking,
            'near_miss_diagnostics': base_summary.get('tracking', {}).get('near_miss_diagnostics'),
        },
        'optimized': {
            'strict_near_miss_distance_diagnostics': False,
            'run_dir': str(opt_dir),
            'wall_time_seconds': opt_time,
            'particle_hits_sha256': digest(opt_dir / 'particle_hits.npz'),
            'tracking': opt_tracking,
            'near_miss_diagnostics': opt_summary.get('tracking', {}).get('near_miss_diagnostics'),
        },
    },
    'comparisons': {
        'particle_hits_npz_byte_equal': npz_equal == '0',
        'accepted_physics_summary_equal': physics_equal,
        'tracking_difference': {
            key: {'baseline': base_tracking.get(key), 'optimized': opt_tracking.get(key)}
            for key in sorted(set(base_tracking) | set(opt_tracking))
            if base_tracking.get(key) != opt_tracking.get(key)
        },
        'wall_time_speedup': (base_time / opt_time) if opt_time > 0 else None,
        'wall_time_reduction_fraction': ((base_time - opt_time) / base_time) if base_time > 0 else None,
    },
    'interpretation': (
        'Performance-only candidate: accepted collision outputs are unchanged; '
        'near-miss distance bookkeeping is omitted in optimized mode. '
        'This does not prove particle spatial convergence or physical accuracy.'
    ),
}
Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

sha256sum "${report}" > "${report}.sha256"
printf 'job_id=%s\nstatus=%s\nreport=%s\n' "${STEP30_JOB_ID}" \
  "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${report}")" "${report}"
