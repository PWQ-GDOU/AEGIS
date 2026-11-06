# Step30 Remote Safety Runbook v2

This runbook applies to new particle discovery and confirmation jobs. The
historical v1 runner and its outputs are retained unchanged.

## Safety guarantees

- Numerical entry points require a non-Windows remote host and explicit
  `STEP30_REMOTE_EXECUTION=1`. Slurm/PBS is required by default; the only
  exception is an explicit `--allow-direct-remote` together with
  `STEP30_ALLOW_DIRECT_REMOTE=1` on a host where scheduler tools are absent.
- `dcu_cpu` is the only enabled target for the current NumPy/SciPy tracker.
  `user_gpu` remains fail-closed until a separately validated equivalent
  backend is registered.
- A fresh state root must not already exist. Resume is allowed only
  with `--resume`, the matching v2 status schema, unchanged input inventory,
  and unchanged parcel/seed selection.
- State root and artifact root must be distinct and non-nested.
- The runner checks scheduler CPU capacity before starting and refuses a worker
  count above that capacity.
- The runner checks a configurable minimum free space before starting. All
  temporary, Matplotlib, and Python bytecode paths are under the state root.
- The runner rechecks free space during every heartbeat interval and
  terminates the process group if the disk guard is crossed.
- The runner refuses to start when any Step30 tracker or runner process is
  already active in the shared workspace, including historical v1 jobs.
- Topology, native-SI directory contents, transform, discovery report, and
  tracker source are hashed before execution. An optional expected inventory
  can make that hash check fail closed against a transferred manifest.
- Each seed has separate stdout and stderr logs, a process-group timeout, and
  periodic heartbeat updates in `run_status.json`. Timeout termination uses
  SIGTERM followed by SIGKILL if required.
- A matching artifact not recorded in the current state is rejected rather
  than silently reused. This prevents mixing runs from different jobs.

## Operational boundary

Do not run the v2 entry point from Windows. The expected path is the DCU
Slurm wrapper `remote/dcu_cpu_particle_confirmation_v2.sbatch`; where no
scheduler exists, use `remote/run_direct_remote_v2.sh` only with explicit
direct-remote opt-in. Use a fresh
`STEP30_JOB_ID` and a fresh remote project root for each independent run.

The state status is operational provenance, not a convergence or accuracy
claim. It does not alter `official_accuracy_gate`, `official_reference_gate`,
or `fem_gate`.
