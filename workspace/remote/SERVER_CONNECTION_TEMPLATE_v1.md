# Step30 Server Connection Record v1

Fill this record outside the repository or commit only non-secret values. Do
not store passwords, private keys, tokens, or scheduler credentials here.

## DCU CPU

- Host: `100.121.154.58` (Tailscale)
- SSH user: `mgr`
- Scheduler: `slurm` / `pbs`
- Partition or queue:
- CPU cores per job:
- Remote project root: under `PWQ`, concise workspace name such as `step30_cpu_v1`
- Python/module/container:
- Expected wall time:
- Filesystem quota:
- Transfer method approved by administrator:

## User GPU

- Host: `100.92.47.51` (Tailscale)
- SSH user: `user`
- Scheduler: none detected during non-computational probe
- Partition or queue:
- GPU model/count: 4 x NVIDIA GeForce RTX 2080 Ti, 11264 MiB each
- CUDA version: record after backend setup; driver `580.159.03`
- PyTorch/Numba/CuPy version:
- Remote project root:
- Validated GPU backend commit/hash:
- Expected wall time:

## Required evidence returned for every job

- Scheduler job ID and server hostname.
- Remote code archive or checkout hash.
- Runtime/container/module digest.
- Input file and directory-manifest hashes.
- Exact command line and environment target.
- Start/end timestamps and exit code.
- stdout/stderr files and SHA-256 hashes.
- Output files and SHA-256 hashes.
- `official_accuracy_gate=false` and `fem_gate=blocked` unless a separate
  evidence contract explicitly changes them.
