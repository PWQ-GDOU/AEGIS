"""Fail-closed safety and provenance helpers for remote Step30 workloads."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Any, Iterable, Mapping


APPROVED_TARGETS = {"dcu_cpu", "user_gpu"}
REMOTE_EXECUTION_ENV = "STEP30_REMOTE_EXECUTION"


class RemoteSafetyError(RuntimeError):
    """Raised when a workload cannot be proven safe to start or resume."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise RemoteSafetyError(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_manifest_hash(directory: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash a directory by sorted relative paths, sizes, and file SHA-256 values."""
    if not directory.is_dir():
        raise RemoteSafetyError(f"required directory is missing: {directory}")
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise RemoteSafetyError(f"input directory contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        rows.append({"relative_path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    # H = SHA256(canonical_json(sorted(relative_path, bytes, file_sha256)))
    manifest_hash = canonical_sha256(rows)
    return manifest_hash, rows


def ensure_remote_server(*, target: str, require_scheduler: bool = True, allow_direct_remote: bool = False) -> dict[str, Any]:
    normalized_target = target.strip().lower()
    if normalized_target not in APPROVED_TARGETS:
        raise RemoteSafetyError(f"unsupported execution target: {target!r}")
    if os.name == "nt" or os.environ.get(REMOTE_EXECUTION_ENV) != "1":
        raise RemoteSafetyError(
            "numerical workload refused: require a non-Windows host with "
            f"{REMOTE_EXECUTION_ENV}=1"
        )
    scheduler_job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID")
    if require_scheduler and not scheduler_job_id:
        if not (allow_direct_remote and os.environ.get("STEP30_ALLOW_DIRECT_REMOTE") == "1"):
            raise RemoteSafetyError("numerical workload refused: no Slurm/PBS job identity")
        scheduler = "direct_remote"
    else:
        scheduler = "slurm" if os.environ.get("SLURM_JOB_ID") else "pbs" if os.environ.get("PBS_JOBID") else None
    return {
        "execution_target": normalized_target,
        "os_name": os.name,
        "remote_execution": os.environ.get(REMOTE_EXECUTION_ENV),
        "scheduler": scheduler,
        "scheduler_job_id": scheduler_job_id,
    }


def scheduler_cpu_capacity(*, allow_direct_remote: bool = False) -> int:
    raw = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("PBS_NP") or os.environ.get("SLURM_CPUS_ON_NODE")
    if not raw:
        if not (allow_direct_remote and os.environ.get("STEP30_ALLOW_DIRECT_REMOTE") == "1"):
            raise RemoteSafetyError("cannot verify CPU capacity: scheduler CPU variable is missing")
        # Direct mode is an explicit exception. Restrict to this process' CPU affinity.
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            value = os.cpu_count() or 0
            if value <= 0:
                raise RemoteSafetyError("cannot determine direct-remote CPU capacity")
            return value
    try:
        value = int(str(raw).split("(", 1)[0])
    except ValueError as exc:
        raise RemoteSafetyError(f"invalid scheduler CPU count: {raw!r}") from exc
    if value <= 0:
        raise RemoteSafetyError(f"scheduler CPU count must be positive: {value}")
    return value


def validate_workers(workers: int, seed_count: int, *, allow_direct_remote: bool = False) -> dict[str, int]:
    if workers <= 0:
        raise RemoteSafetyError("workers must be positive")
    if seed_count <= 0:
        raise RemoteSafetyError("at least one seed is required")
    capacity = scheduler_cpu_capacity(allow_direct_remote=allow_direct_remote)
    if workers > capacity:
        raise RemoteSafetyError(f"workers={workers} exceeds scheduler CPU capacity={capacity}")
    effective = min(workers, seed_count, capacity)
    return {"requested": workers, "effective": effective, "scheduler_cpu_capacity": capacity}


def acquire_direct_remote_lock(path: Path) -> Any:
    """Acquire a non-blocking host-wide lock for explicit direct execution."""
    try:
        import fcntl
    except ImportError as exc:
        raise RemoteSafetyError("direct-remote mode requires POSIX file locking") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RemoteSafetyError(f"another direct-remote workload holds the lock: {path}") from exc
    return handle


def release_direct_remote_lock(handle: Any) -> None:
    with suppress(Exception):
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    with suppress(Exception):
        handle.close()


def active_step30_processes() -> list[dict[str, Any]]:
    """Return active Step30 runner/tracker processes on Linux, excluding self."""
    if os.name == "nt":
        return []
    current_pid = os.getpid()
    rows: list[dict[str, Any]] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return rows
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == current_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
            arguments = [part.decode("utf-8", errors="replace") for part in raw.split(b"\x00") if part]
        except (OSError, ValueError):
            continue
        command = " ".join(arguments)
        basenames = {Path(argument).name for argument in arguments}
        if not basenames.intersection({
            "step30_lagrangian_particle_tracking.py",
            "step30_particle_confirmation_parallel_v1.py",
            "step30_particle_confirmation_parallel_v2.py",
        }):
            continue
        rows.append({"pid": int(entry.name), "command": command})
    return sorted(rows, key=lambda item: item["pid"])


def ensure_distinct_paths(paths: Mapping[str, Path]) -> None:
    resolved = {label: path.resolve() for label, path in paths.items()}
    for label, path in resolved.items():
        for other_label, other in resolved.items():
            if label >= other_label:
                continue
            if path == other or path in other.parents or other in path.parents:
                raise RemoteSafetyError(f"paths must be distinct and non-nested: {label}={path}, {other_label}={other}")


def validate_state_root(path: Path, *, resume: bool) -> Path:
    resolved = path.resolve()
    if path.is_symlink():
        raise RemoteSafetyError(f"state root must not be a symlink: {path}")
    if resolved.exists():
        if not resolved.is_dir():
            raise RemoteSafetyError(f"state root is not a directory: {resolved}")
        if resume:
            if not (resolved / "run_status.json").is_file():
                raise RemoteSafetyError("resume requires state root/run_status.json")
        else:
            raise RemoteSafetyError(f"refusing to reuse existing state root: {resolved}")
    else:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        try:
            resolved.mkdir()
        except FileExistsError as exc:
            raise RemoteSafetyError(f"state root appeared during creation: {resolved}") from exc
    return resolved


def check_disk_space(path: Path, minimum_free_gb: float) -> dict[str, int | float]:
    if minimum_free_gb <= 0:
        raise RemoteSafetyError("minimum free disk space must be positive")
    anchor = path if path.exists() else path.parent
    usage = shutil.disk_usage(anchor)
    required_bytes = int(minimum_free_gb * 1024**3)
    # safe_to_start := free_bytes >= required_bytes
    if usage.free < required_bytes:
        raise RemoteSafetyError(
            f"insufficient free disk space at {anchor}: {usage.free} < {required_bytes} bytes"
        )
    return {
        "anchor": str(anchor.resolve()),
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
        "minimum_free_gb": float(minimum_free_gb),
        "required_bytes": required_bytes,
    }


def build_input_inventory(entries: Iterable[tuple[str, Path, str]]) -> dict[str, Any]:
    """Build immutable input inventory; kind is ``file`` or ``directory``."""
    inventory: list[dict[str, Any]] = []
    for label, path, kind in entries:
        resolved = path.resolve()
        if kind == "file":
            inventory.append({"label": label, "kind": kind, "path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)})
        elif kind == "directory":
            digest, files = directory_manifest_hash(resolved)
            inventory.append({"label": label, "kind": kind, "path": str(resolved), "file_count": len(files), "sha256": digest, "files": files})
        else:
            raise RemoteSafetyError(f"unsupported inventory kind: {kind}")
    return {"version": "step30-remote-input-inventory-v1", "entries": inventory, "inventory_sha256": canonical_sha256(inventory)}


def validate_expected_inventory(actual: Mapping[str, Any], expected_path: Path | None) -> None:
    if expected_path is None:
        return
    if not expected_path.is_file():
        raise RemoteSafetyError(f"expected input inventory is missing: {expected_path}")
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    expected_entries = expected.get("entries") if isinstance(expected, dict) else None
    if expected_entries != actual.get("entries"):
        raise RemoteSafetyError("input inventory hash preflight failed: expected entries differ")
