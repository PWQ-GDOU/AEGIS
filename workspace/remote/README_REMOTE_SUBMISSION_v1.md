# Step30 Remote Submission Kit v1

Effective 2026-08-28, all CFD, particle tracking, FEM, solver, GPU execution,
and numerical validation must run on an approved server. The workstation may
only prepare, transfer, monitor, and archive files.

## Approved targets

- `dcu_cpu`: multi-core CPU jobs on the DCU server.
- `user_gpu`: GPU jobs on the user server.

The current particle tracker is a CPU/NumPy/SciPy implementation. It must be
submitted to `dcu_cpu`; setting `user_gpu` does not make it GPU accelerated.
The GPU template is intentionally fail-closed until a separately validated GPU
backend is supplied.

## Package contents

- `remote_job_manifest_v1.json`: fill in server, scheduler, paths, and code/input
  identities before submission.
- `dcu_cpu_particle_confirmation_v1.sbatch`: Slurm template for the existing
  strict-v2 topology confirmation workload.
- `user_gpu_backend_gate_v1.sbatch`: fail-closed GPU submission template.
- `submit_remote_job.ps1`: optional transfer/submission wrapper. It refuses
  local execution and refuses paths outside the H: project tree.
- `retrieve_remote_job.ps1`: retrieves only the server-side result package and
  writes a local SHA-256 inventory under `workspace/output/remote_returns`.
- `result_manifest_schema_v1.json`: required provenance for returned results.

## Required operator inputs

Before a real submission, record the actual values in the manifest:

1. DCU and user server hostnames or scheduler endpoints.
2. SSH account and authentication method, without storing secrets in this repo.
3. Scheduler type and partition/queue name.
4. Remote project root and runtime module/container. The CPU layout is
   `PWQ/step30_cpu_v1/workspace` with `PWQ/step30_cpu_v1/.venv`.
5. Remote code archive hash and all input artifact hashes.
6. Job ID, stdout/stderr paths, exit code, and output hashes after completion.

The historical host in `output/step30_fem_remote_export_v1` is not treated as
current connectivity information.

## Submission boundary

Do not run `python`, `pytest`, a solver, or a GPU program from the workstation
for this project. The PowerShell wrapper only performs manifest validation,
file transfer, remote scheduler submission, and result retrieval; it never
executes a numerical command locally.

Returned artifacts must remain under the H: project directory and must retain
their server-side manifest, scheduler job ID, remote code hash, input hashes,
stdout/stderr, and output hashes. Existing v3/v4/v5 reports are immutable
historical artifacts and must not be overwritten.

The Windows control workspace is `external_workspaces/step30_remote_control_v1`.
Submission scripts set local temporary paths and SSH host-key files there;
they do not use C: for newly created project state.
