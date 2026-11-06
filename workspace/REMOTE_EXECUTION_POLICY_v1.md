# Step30 Remote Execution Policy

Effective 2026-08-28, all CFD, particle, FEM, solver, and GPU calculations are server-only.

- `dcu_cpu`: approved target for multi-core CPU execution on the DCU server.
- `user_gpu`: approved target for GPU execution on the user server.
- The local workstation must not run simulations or numerical validation.
- New local control files, temporary files, SSH host-key files, and returned
  packages must be stored under `external_workspaces/step30_remote_control_v1`; C: is
  not an allowed project workspace or newly-created runtime location.
- Tracking and confirmation entry points require `STEP30_EXECUTION_TARGET` to be explicitly set to an approved value.
- Numerical entry points additionally require `STEP30_REMOTE_EXECUTION=1` on a non-Windows server; Windows execution is rejected even when a target variable is set.
- The current particle tracker is CPU/NumPy/SciPy. Setting `user_gpu` does not make it GPU-accelerated; a separately validated GPU backend is required.
- Existing local results remain historical artifacts and are not reclassified as server-generated results.
