#!/usr/bin/env bash
# Remote-only bootstrap. This creates directories and a Python venv; it does
# not run CFD, particle, FEM, solver, GPU, or numerical validation code.
set -euo pipefail

workspace_name="${STEP30_REMOTE_WORKSPACE_NAME:-step30_cpu_v1}"
if [[ ! "$workspace_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "invalid workspace name" >&2
  exit 64
fi

choose_pwq_root() {
  local candidate
  for candidate in /PWQ "$HOME/PWQ" "/home/$USER/PWQ"; do
    if [[ -d "$candidate" && -w "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
    if [[ ! -e "$candidate" ]] && mkdir -p "$candidate" 2>/dev/null; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

pwq_root="$(choose_pwq_root)" || {
  echo "no writable PWQ directory found" >&2
  exit 73
}
remote_root="$pwq_root/$workspace_name"
mkdir -p "$remote_root"

if command -v sbatch >/dev/null 2>&1; then
  scheduler="slurm"
  scheduler_probe="$(command -v sbatch)"
elif command -v qsub >/dev/null 2>&1; then
  scheduler="pbs"
  scheduler_probe="$(command -v qsub)"
else
  echo "no Slurm or PBS submission command found" >&2
  exit 69
fi

python3 -m venv "$remote_root/.venv"
cat > "$remote_root/remote_environment.json" <<EOF
{
  "schema": "step30-remote-environment-v1",
  "host": "$(hostname)",
  "user": "$(id -un)",
  "workspace": "$remote_root",
  "venv": "$remote_root/.venv",
  "scheduler": "$scheduler",
  "scheduler_probe": "$scheduler_probe",
  "python": "$remote_root/.venv/bin/python",
  "numerical_job_submitted": false
}
EOF
printf '%s\n' "$remote_root"

