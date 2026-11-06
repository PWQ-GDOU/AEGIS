"""Controlled parallel launcher for the five v5 particle confirmations."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parent
OUTPUT = WORKSPACE / "output"
STATE_ROOT = OUTPUT / "step30_v5_confirmation_parallel_v1"
STATUS_PATH = STATE_ROOT / "run_status.json"
DEFAULT_SEEDS = (57721, 104729, 130363, 161027, 196613)
N_PARCELS = 20_000
FIELD_TIME = "0.8684190921"
POLICY = "strict_v2"
BACKEND = "cfd_topology"
EXECUTION_TARGET_ENV = "STEP30_EXECUTION_TARGET"
EXECUTION_TARGETS = {"dcu_cpu", "user_gpu"}
REMOTE_EXECUTION_MODE_ENV = "STEP30_REMOTE_EXECUTION"
REQUIRED = ("particle_hits.npz", "summary.json", "validation.png", "particle_run_manifest.json")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def hpath(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.drive.upper() == "H:":
        return path
    if os.name != "nt" and os.environ.get(REMOTE_EXECUTION_MODE_ENV) == "1":
        return path
    raise RuntimeError(
        f"{label} must remain on H: drive locally, or use a non-Windows remote "
        f"server with {REMOTE_EXECUTION_MODE_ENV}=1: {path}"
    )

def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value

def complete_run(artifact_root: Path, seed: int, n_parcels: int) -> dict[str, Any] | None:
    policy_root = artifact_root / POLICY
    if not policy_root.is_dir():
        return None
    for run_dir in policy_root.iterdir():
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
                if record.get("file") != filename or record.get("sha256") != sha256(run_dir / filename):
                    return None
            return {"status": "completed", "run_directory": str(run_dir.resolve()), "particle_hits_sha256": sha256(run_dir / "particle_hits.npz"), "summary_sha256": sha256(run_dir / "summary.json"), "validation_sha256": sha256(run_dir / "validation.png"), "manifest_sha256": sha256(run_dir / "particle_run_manifest.json"), "fallback_hits": 0}
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return None

def environment() -> dict[str, str]:
    target = os.environ.get(EXECUTION_TARGET_ENV, "").strip().lower()
    if target not in EXECUTION_TARGETS:
        raise RuntimeError(f"server-only runner requires {EXECUTION_TARGET_ENV}=dcu_cpu or user_gpu")
    if os.name == "nt" or os.environ.get(REMOTE_EXECUTION_MODE_ENV) != "1":
        raise RuntimeError(
            f"server-only runner requires a non-Windows remote host with "
            f"{REMOTE_EXECUTION_MODE_ENV}=1"
        )
    temp = hpath(STATE_ROOT / "tmp", "TEMP")
    mpl = hpath(STATE_ROOT / "mplconfig", "MPLCONFIGDIR")
    cache = hpath(STATE_ROOT / "pycache", "PYTHONPYCACHEPREFIX")
    for path in (temp, mpl, cache):
        path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"TEMP": str(temp), "TMP": str(temp), "TMPDIR": str(temp), "MPLCONFIGDIR": str(mpl), "PYTHONPYCACHEPREFIX": str(cache), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", EXECUTION_TARGET_ENV: target, REMOTE_EXECUTION_MODE_ENV: "1"})
    return env

def run_seed(seed: int, artifact_root: Path, topology: Path, native_si: Path, transform: Path, n_parcels: int) -> dict[str, Any]:
    existing = complete_run(artifact_root, seed, n_parcels)
    if existing:
        existing["seed"] = seed
        return existing
    log = STATE_ROOT / f"seed_{seed}.log"
    command = [sys.executable, "step30_lagrangian_particle_tracking.py", "--cfd", "native_si", "--collision-backend", BACKEND, "--collision-policy", POLICY, "--topology", str(topology), "--native-si-directory", str(native_si), "--field-time", FIELD_TIME, "--analytic-reference-transform", str(transform), "--immutable-output", "--parcels", str(N_PARCELS), "--seed", str(seed), "--max-impacts", "3", "--mass-flow", "nominal", "--psd-sampling", "mass_stratified_v1", "--max-time", "0.05", "--dt", "1e-5", "--particle-cfl", "0.25", "--strict-max-substep", "1e-5", "--cfd-interpolation", "idw_k8_v1", "--particle-integrator", "midpoint_exponential_v2", "--max-cfd-speed", "500"]
    started = time.time()
    with log.open("w", encoding="utf-8", newline="\n") as stream:
        process = subprocess.run(command, cwd=str(WORKSPACE), env=environment(), stdout=stream, stderr=subprocess.STDOUT, check=False)
    finished = time.time()
    result: dict[str, Any] = {"seed": seed, "status": "failed", "returncode": process.returncode, "log": str(log.resolve()), "started_epoch": started, "finished_epoch": finished, "elapsed_s": finished - started, "command": command}
    accepted = complete_run(artifact_root, seed, n_parcels)
    if process.returncode == 0 and accepted:
        result.update(accepted)
    else:
        result["failure_reason"] = "tracker returned nonzero" if process.returncode else "immutable four-piece output missing or invalid"
    return result

def main(argv: list[str] | None = None) -> int:
    global N_PARCELS, STATE_ROOT, STATUS_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--topology", type=Path, default=OUTPUT / "step30_acceptance_0.8684190921" / "bucket_topology_v3_0.8684190921.npz")
    parser.add_argument("--native-si-directory", type=Path, default=OUTPUT / "native_bucket_si_urans_0.8684190921")
    parser.add_argument("--reference-transform", type=Path, default=OUTPUT / "authoritative_bucket_reference_transform_v1.json")
    parser.add_argument("--discovery-report", type=Path, default=OUTPUT / "step30_particle_convergence_5k_10k_20k_topology_aggregate_v5_discovery_r2.json")
    parser.add_argument("--execution-target", choices=sorted(EXECUTION_TARGETS), required=True)
    parser.add_argument("--parcels", type=int, default=N_PARCELS)
    parser.add_argument("--state-root", type=Path, default=STATE_ROOT)
    args = parser.parse_args(argv)
    if args.parcels <= 0:
        raise ValueError("--parcels must be positive")
    N_PARCELS = int(args.parcels)
    STATE_ROOT = args.state_root.resolve()
    STATUS_PATH = STATE_ROOT / "run_status.json"
    os.environ[EXECUTION_TARGET_ENV] = args.execution_target
    seeds = list(dict.fromkeys(args.seeds or DEFAULT_SEEDS))
    topology = hpath(args.topology, "topology")
    native_si = hpath(args.native_si_directory, "native SI directory")
    transform = hpath(args.reference_transform, "reference transform")
    discovery_report = hpath(args.discovery_report, "discovery report")
    for path in (topology, native_si, transform, discovery_report):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = native_si / "native_si_manifest_0.8684190921.json"
    if str(load_json(manifest).get("field_time")) != FIELD_TIME:
        raise ValueError(f"native SI field time mismatch: {manifest}")
    discovery = load_json(discovery_report)
    artifact_set_id = str((discovery.get("identity") or {}).get("artifact_set_id", ""))
    if not artifact_set_id:
        raise ValueError("discovery report has no artifact_set_id")
    artifact_root = hpath(OUTPUT / "step30_runs" / artifact_set_id, "artifact root")
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"version": "step30-particle-confirmation-parallel-v1", "status": "running", "workspace": str(WORKSPACE), "artifact_set_id": artifact_set_id, "requested_seeds": seeds, "workers": max(1, min(int(args.workers), len(seeds))), "expected": {"n_parcels": N_PARCELS, "field_time": FIELD_TIME, "collision_policy": POLICY, "collision_backend": BACKEND, "required_files": list(REQUIRED)}, "seeds": {}}
    write_json(STATUS_PATH, state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=state["workers"]) as pool:
        futures = {pool.submit(run_seed, seed, artifact_root, topology, native_si, transform, N_PARCELS): seed for seed in seeds}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            state["seeds"][str(result["seed"])] = result
            write_json(STATUS_PATH, state)
    completed = sum(value.get("status") == "completed" for value in state["seeds"].values())
    state["completed_seed_count"] = completed
    state["status"] = "completed" if completed == len(seeds) else "incomplete"
    state["updated_epoch"] = time.time()
    write_json(STATUS_PATH, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if state["status"] == "completed" else 3

if __name__ == "__main__":
    raise SystemExit(main())

