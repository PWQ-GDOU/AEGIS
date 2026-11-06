"""Fail-closed, server-only parallel particle runner with safe resume support.

This v2 entry point is separate from the historical v1 runner. It adds
operational safety and provenance controls without changing the particle
model or reclassifying old results.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from step30_remote_safety_v1 import (
    RemoteSafetyError,
    build_input_inventory,
    canonical_sha256,
    check_disk_space,
    acquire_direct_remote_lock,
    active_step30_processes,
    ensure_distinct_paths,
    ensure_remote_server,
    release_direct_remote_lock,
    sha256_file,
    validate_expected_inventory,
    validate_state_root,
    validate_workers,
)


WORKSPACE = Path(__file__).resolve().parent
OUTPUT = WORKSPACE / "output"
DEFAULT_SEEDS = (510001, 510003, 510009, 510021, 510023)
DEFAULT_PARCELS = 60_000
FIELD_TIME = "0.8684190921"
POLICY = "strict_v2"
BACKEND = "cfd_topology"
REQUIRED = ("particle_hits.npz", "summary.json", "validation.png", "particle_run_manifest.json")
HEARTBEAT_S = 30.0
DEFAULT_MAX_RUNTIME_S = 7200.0
DEFAULT_MIN_FREE_GB = 10.0
MAX_RUNTIME_S = 86_400.0
MAX_PARCELS = 1_000_000
_ACTIVE_CHILDREN: set[subprocess.Popen[str]] = set()
_ACTIVE_CHILDREN_LOCK = threading.Lock()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RemoteSafetyError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass


def complete_run(artifact_root: Path, seed: int, n_parcels: int) -> dict[str, Any] | None:
    policy_root = artifact_root / POLICY
    if not policy_root.is_dir():
        return None
    for run_dir in sorted(policy_root.iterdir(), key=lambda item: item.name):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        if any(not (run_dir / name).is_file() for name in REQUIRED):
            continue
        try:
            summary = load_json(run_dir / "summary.json")
            parameters = summary.get("parameters") or {}
            cfd = summary.get("cfd") or {}
            tracking = summary.get("tracking") or {}
            if int(parameters.get("n_parcels", -1)) != n_parcels or int(parameters.get("seed", -1)) != seed:
                continue
            if parameters.get("collision_policy") != POLICY or parameters.get("collision_backend") != BACKEND:
                continue
            if str(cfd.get("field_time")) != FIELD_TIME or int(tracking.get("fallback_hits", -1)) != 0:
                continue
            manifest = load_json(run_dir / "particle_run_manifest.json")
            outputs = manifest.get("outputs") or {}
            for key, filename in (("particle_hits", "particle_hits.npz"), ("summary", "summary.json"), ("validation", "validation.png")):
                record = outputs.get(key) or {}
                if record.get("file") != filename or record.get("sha256") != sha256_file(run_dir / filename):
                    return None
            return {
                "status": "completed",
                "run_directory": str(run_dir.resolve()),
                "particle_hits_sha256": sha256_file(run_dir / "particle_hits.npz"),
                "summary_sha256": sha256_file(run_dir / "summary.json"),
                "validation_sha256": sha256_file(run_dir / "validation.png"),
                "manifest_sha256": sha256_file(run_dir / "particle_run_manifest.json"),
                "fallback_hits": 0,
            }
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return None


def environment(state_root: Path, target: str, *, allow_direct_remote: bool) -> dict[str, str]:
    ensure_remote_server(target=target, require_scheduler=True, allow_direct_remote=allow_direct_remote)
    temp = state_root / "tmp"
    mpl = state_root / "mplconfig"
    cache = state_root / "pycache"
    for path in (temp, mpl, cache):
        path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "TEMP": str(temp),
        "TMP": str(temp),
        "TMPDIR": str(temp),
        "MPLCONFIGDIR": str(mpl),
        "PYTHONPYCACHEPREFIX": str(cache),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "STEP30_EXECUTION_TARGET": target,
        "STEP30_REMOTE_EXECUTION": "1",
    })
    return env


def _terminate_process_group(process: subprocess.Popen[str]) -> str:
    if process.poll() is not None:
        return "already_exited"
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
        return "sigterm"
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)
        return "sigkill"


def _register_child(process: subprocess.Popen[str]) -> None:
    with _ACTIVE_CHILDREN_LOCK:
        _ACTIVE_CHILDREN.add(process)


def _unregister_child(process: subprocess.Popen[str]) -> None:
    with _ACTIVE_CHILDREN_LOCK:
        _ACTIVE_CHILDREN.discard(process)


def _handle_shutdown(signum: int, _frame: Any) -> None:
    with _ACTIVE_CHILDREN_LOCK:
        children = list(_ACTIVE_CHILDREN)
    for process in children:
        _terminate_process_group(process)
    raise SystemExit(128 + signum)


def run_seed(
    seed: int,
    artifact_root: Path,
    topology: Path,
    native_si: Path,
    transform: Path,
    n_parcels: int,
    max_runtime_s: float,
    min_free_gb: float,
    state_root: Path,
    target: str,
    allow_direct_remote: bool,
    update_seed: Callable[[int, dict[str, Any]], None],
    allow_recorded_run: str | None = None,
) -> dict[str, Any]:
    existing = complete_run(artifact_root, seed, n_parcels)
    if existing and allow_recorded_run and existing.get("run_directory") == allow_recorded_run:
        existing["seed"] = seed
        existing["reuse_reason"] = "recorded_resume_artifact"
        return existing
    if existing:
        return {"seed": seed, "status": "failed", "failure_reason": "matching artifact exists but is not recorded by this state"}

    stdout_log = state_root / f"seed_{seed}.stdout.log"
    stderr_log = state_root / f"seed_{seed}.stderr.log"
    command = [
        sys.executable,
        "step30_lagrangian_particle_tracking.py",
        "--cfd", "native_si",
        "--collision-backend", BACKEND,
        "--collision-policy", POLICY,
        "--topology", str(topology),
        "--native-si-directory", str(native_si),
        "--field-time", FIELD_TIME,
        "--analytic-reference-transform", str(transform),
        "--immutable-output",
        "--parcels", str(n_parcels),
        "--seed", str(seed),
        "--max-impacts", "3",
        "--mass-flow", "nominal",
        "--psd-sampling", "mass_stratified_v1",
        "--max-time", "0.05",
        "--dt", "1e-5",
        "--particle-cfl", "0.25",
        "--strict-max-substep", "1e-5",
        "--cfd-interpolation", "idw_k8_v1",
        "--particle-integrator", "midpoint_exponential_v2",
        "--max-cfd-speed", "500",
    ]
    started = time.time()
    update_seed(seed, {"status": "starting", "started_epoch": started, "command": command})
    env = environment(state_root, target, allow_direct_remote=allow_direct_remote)
    with stdout_log.open("w", encoding="utf-8", newline="\n") as stdout, stderr_log.open("w", encoding="utf-8", newline="\n") as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(WORKSPACE),
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            text=True,
        )
        _register_child(process)
        try:
            update_seed(seed, {"status": "running", "pid": process.pid, "heartbeat_epoch": time.time()})
            timed_out = False
            disk_guarded = False
            while process.poll() is None:
                elapsed_s = time.time() - started
                # elapsed_s = wall_clock_now - started_epoch
                if elapsed_s > max_runtime_s:
                    timed_out = True
                    termination = _terminate_process_group(process)
                    update_seed(seed, {"status": "timeout", "elapsed_s": elapsed_s, "termination": termination, "heartbeat_epoch": time.time()})
                    break
                try:
                    check_disk_space(state_root, min_free_gb)
                except RemoteSafetyError as exc:
                    disk_guarded = True
                    termination = _terminate_process_group(process)
                    update_seed(seed, {"status": "disk_guard", "elapsed_s": elapsed_s, "termination": termination, "failure_reason": str(exc), "heartbeat_epoch": time.time()})
                    break
                update_seed(seed, {"status": "running", "elapsed_s": elapsed_s, "heartbeat_epoch": time.time()})
                time.sleep(min(HEARTBEAT_S, max(0.1, max_runtime_s - elapsed_s)))
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                _terminate_process_group(process)
            raise
        finally:
            _unregister_child(process)
    finished = time.time()
    result: dict[str, Any] = {
        "seed": seed,
        "status": "failed",
        "returncode": returncode,
        "stdout_log": str(stdout_log.resolve()),
        "stderr_log": str(stderr_log.resolve()),
        "started_epoch": started,
        "finished_epoch": finished,
        "elapsed_s": finished - started,
        "command": command,
        "stdout_sha256": sha256_file(stdout_log),
        "stderr_sha256": sha256_file(stderr_log),
    }
    if timed_out:
        result["failure_reason"] = "hard runtime limit exceeded"
        return result
    if disk_guarded:
        result["failure_reason"] = "runtime disk guard terminated the process"
        return result
    accepted = complete_run(artifact_root, seed, n_parcels)
    if returncode == 0 and accepted:
        result.update(accepted)
    else:
        result["failure_reason"] = "tracker returned nonzero" if returncode else "immutable four-piece output missing or invalid"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--topology", type=Path, default=OUTPUT / "step30_acceptance_0.8684190921" / "bucket_topology_v3_0.8684190921.npz")
    parser.add_argument("--native-si-directory", type=Path, default=OUTPUT / "native_bucket_si_urans_0.8684190921")
    parser.add_argument("--reference-transform", type=Path, default=OUTPUT / "authoritative_bucket_reference_transform_v1.json")
    parser.add_argument("--discovery-report", type=Path, default=OUTPUT / "step30_particle_convergence_5k_10k_20k_topology_aggregate_v5_discovery_r2.json")
    parser.add_argument("--execution-target", choices=("dcu_cpu", "user_gpu"), required=True)
    parser.add_argument("--parcels", type=int, default=DEFAULT_PARCELS)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--max-runtime-s", type=float, default=DEFAULT_MAX_RUNTIME_S)
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--expected-input-inventory", type=Path)
    parser.add_argument("--allow-direct-remote", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if args.parcels <= 0 or not 0 < args.max_runtime_s <= MAX_RUNTIME_S:
        raise RemoteSafetyError(f"parcels must be positive and max-runtime-s must be in (0, {MAX_RUNTIME_S}]")
    if args.parcels > MAX_PARCELS:
        raise RemoteSafetyError(f"parcels exceeds safety cap {MAX_PARCELS}")
    if args.execution_target == "user_gpu":
        raise RemoteSafetyError("user_gpu is blocked until a separately validated GPU backend is registered")
    if args.allow_direct_remote and os.environ.get("STEP30_ALLOW_DIRECT_REMOTE") != "1":
        raise RemoteSafetyError("--allow-direct-remote requires STEP30_ALLOW_DIRECT_REMOTE=1")
    seeds = list(dict.fromkeys(args.seeds or DEFAULT_SEEDS))
    worker_info = validate_workers(int(args.workers), len(seeds), allow_direct_remote=bool(args.allow_direct_remote))

    topology = args.topology.resolve()
    native_si = args.native_si_directory.resolve()
    transform = args.reference_transform.resolve()
    discovery_report = args.discovery_report.resolve()
    tracker = WORKSPACE / "step30_lagrangian_particle_tracking.py"
    safety_module = WORKSPACE / "step30_remote_safety_v1.py"
    artifact_set_id = str((load_json(discovery_report).get("identity") or {}).get("artifact_set_id", ""))
    if not artifact_set_id:
        raise RemoteSafetyError("discovery report has no artifact_set_id")
    artifact_root = (OUTPUT / "step30_runs" / artifact_set_id).resolve()
    active_processes = active_step30_processes()
    if active_processes:
        raise RemoteSafetyError(f"active Step30 process detected in shared workspace: {active_processes}")
    state_root = validate_state_root(args.state_root, resume=bool(args.resume))
    ensure_distinct_paths({"state_root": state_root, "artifact_root": artifact_root})
    for path, label in ((topology, "topology"), (native_si, "native SI directory"), (transform, "reference transform"), (discovery_report, "discovery report"), (tracker, "tracker"), (safety_module, "safety module")):
        if not path.exists():
            raise RemoteSafetyError(f"{label} does not exist: {path}")
    disk = check_disk_space(state_root, float(args.min_free_gb))
    input_inventory = build_input_inventory([
        ("topology", topology, "file"),
        ("native_si", native_si, "directory"),
        ("reference_transform", transform, "file"),
        ("discovery_report", discovery_report, "file"),
        ("tracker", tracker, "file"),
        ("safety_module", safety_module, "file"),
    ])
    validate_expected_inventory(input_inventory, args.expected_input_inventory)
    manifest_path = native_si / "native_si_manifest_0.8684190921.json"
    if str(load_json(manifest_path).get("field_time")) != FIELD_TIME:
        raise RemoteSafetyError(f"native SI field time mismatch: {manifest_path}")

    lock = threading.Lock()
    status_path = state_root / "run_status.json"
    if args.resume:
        state = load_json(status_path)
        if state.get("version") != "step30-particle-confirmation-parallel-v2":
            raise RemoteSafetyError("resume state version mismatch")
        if state.get("status") == "completed":
            raise RemoteSafetyError("completed state cannot be resumed")
        if state.get("artifact_set_id") != artifact_set_id:
            raise RemoteSafetyError("resume refused: artifact_set_id changed")
        if state.get("input_inventory", {}).get("inventory_sha256") != input_inventory["inventory_sha256"]:
            raise RemoteSafetyError("resume refused: input inventory changed")
        if state.get("expected", {}).get("n_parcels") != args.parcels or state.get("requested_seeds") != seeds:
            raise RemoteSafetyError("resume refused: parcels or seeds changed")
        if state.get("expected", {}).get("max_runtime_s") != args.max_runtime_s or state.get("expected", {}).get("min_free_gb") != args.min_free_gb:
            raise RemoteSafetyError("resume refused: runtime or disk guard changed")
        if bool(state.get("direct_remote_opt_in")) != bool(args.allow_direct_remote):
            raise RemoteSafetyError("resume refused: direct-remote mode changed")
        expected_code_hashes = {
            "runner": sha256_file(Path(__file__)),
            "safety_module": sha256_file(safety_module),
            "tracker": sha256_file(tracker),
        }
        if state.get("code_hashes") != expected_code_hashes:
            raise RemoteSafetyError("resume refused: code hash changed")
    else:
        scheduler = ensure_remote_server(target=args.execution_target, require_scheduler=True, allow_direct_remote=bool(args.allow_direct_remote))
        state: dict[str, Any] = {
            "version": "step30-particle-confirmation-parallel-v2",
            "status": "running",
            "workspace": str(WORKSPACE.resolve()),
            "artifact_set_id": artifact_set_id,
            "requested_seeds": seeds,
            "workers": worker_info,
            "scheduler": scheduler,
            "direct_remote_opt_in": bool(args.allow_direct_remote),
            "active_process_snapshot": active_processes,
            "disk_preflight": disk,
            "input_inventory": input_inventory,
            "code_hashes": {"runner": sha256_file(Path(__file__)), "safety_module": sha256_file(safety_module), "tracker": sha256_file(tracker)},
            "expected": {"n_parcels": args.parcels, "field_time": FIELD_TIME, "collision_policy": POLICY, "collision_backend": BACKEND, "required_files": list(REQUIRED), "max_runtime_s": args.max_runtime_s, "min_free_gb": args.min_free_gb},
            "seeds": {},
            "created_epoch": time.time(),
        }
        write_json(status_path, state)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    direct_lock = None
    if args.allow_direct_remote:
        try:
            direct_lock = acquire_direct_remote_lock(state_root.parent / ".step30_direct_remote.lock")
        except RemoteSafetyError as exc:
            state["status"] = "safety_refused"
            state["failure_reason"] = str(exc)
            state["finished_epoch"] = time.time()
            write_json(status_path, state)
            raise

    def update_seed(seed: int, patch: dict[str, Any]) -> None:
        with lock:
            state.setdefault("seeds", {}).setdefault(str(seed), {}).update(patch)
            state["heartbeat_epoch"] = time.time()
            write_json(status_path, state)

    recorded = {str(seed): (row.get("run_directory") if isinstance(row, dict) else None) for seed, row in state.get("seeds", {}).items()}
    pending = [seed for seed in seeds if state.get("seeds", {}).get(str(seed), {}).get("status") != "completed"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_info["effective"]) as pool:
        futures = {
            pool.submit(run_seed, seed, artifact_root, topology, native_si, transform, args.parcels, args.max_runtime_s, float(args.min_free_gb), state_root, args.execution_target, bool(args.allow_direct_remote), update_seed, recorded.get(str(seed))): seed
            for seed in pending
        }
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"seed": seed, "status": "failed", "failure_reason": f"runner exception: {type(exc).__name__}: {exc}"}
            update_seed(seed, result)

    completed = sum(value.get("status") == "completed" for value in state.get("seeds", {}).values())
    state["completed_seed_count"] = completed
    state["status"] = "completed" if completed == len(seeds) else "incomplete"
    state["finished_epoch"] = time.time()
    state["state_sha256"] = canonical_sha256({key: value for key, value in state.items() if key != "state_sha256"})
    write_json(status_path, state)
    if direct_lock is not None:
        release_direct_remote_lock(direct_lock)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if state["status"] == "completed" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RemoteSafetyError as exc:
        print(f"SAFETY_REFUSAL: {exc}", file=sys.stderr)
        raise SystemExit(64)
