#!/usr/bin/env bash
# Remote-only bootstrap for the GPU host. It creates a workspace and venv
# without installing packages or launching a numerical workload.
set -euo pipefail

workspace_name="${STEP30_REMOTE_WORKSPACE_NAME:-step30_gpu_v1}"
if [[ ! "$workspace_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "invalid workspace name" >&2
  exit 64
fi

remote_root="${STEP30_REMOTE_ROOT:-$HOME/PWQ/$workspace_name}"
mkdir -p "$remote_root"
python3 -m venv "$remote_root/.venv"
cat > "$remote_root/remote_environment.json" <<EOF
{
  "schema": "step30-remote-environment-v1",
  "host": "$(hostname)",
  "user": "$(id -un)",
  "workspace": "$remote_root",
  "venv": "$remote_root/.venv",
  "python": "$remote_root/.venv/bin/python",
  "gpu_backend_validated": false,
  "numerical_job_submitted": false
}
EOF
printf '%s\n' "$remote_root"

