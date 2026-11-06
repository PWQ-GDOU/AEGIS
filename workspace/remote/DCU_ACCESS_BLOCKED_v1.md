# DCU Access Status v1

Recorded: 2026-08-28 (Asia/Shanghai)

## Observed state

- Tailscale TCP connectivity to `100.121.154.58:22` is available.
- SSH account attempted: `mgr`.
- Authentication result: `Permission denied (publickey,password)`.
- No scheduler probe was performed after authentication because login did not
  complete.

## Required next action

Provide one of the following through the normal SSH credential flow:

- authorize the existing public key for `mgr`, or
- provide an interactive password entry when the SSH prompt is opened.

Credentials must not be written into the repository, manifest, shell history,
or chat logs.

## Planned remote setup after login

1. Detect `sbatch`/`squeue`/`sinfo` versus `qsub`/`qstat`.
2. Select the installed scheduler; Slurm is preferred if both exist.
3. Create a concise work area under the server's `PWQ` directory.
4. Create `.venv` inside that remote work area with the system Python.
5. Transfer the H-drive project package without running numerical code locally.
6. Submit only the DCU CPU workload and retain job ID, logs, and hashes.

No CPU, CFD, particle, FEM, solver, or numerical validation job has been
submitted to either server as of this record.

