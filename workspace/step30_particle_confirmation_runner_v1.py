"""Resumable, H-drive-only runner for the pre-registered v5 confirmation runs.

This wrapper deliberately keeps orchestration separate from the particle
tracker. It records an auditable status file before and after every seed,
preserves the tracker log on failure, and accepts a run only when the
immutable tracker directory contains all four required artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


WORKSPACE = Path(__file__).resolve().parent
OUTPUT = WORKSPACE / "output"
STATE_ROOT = OUTPUT / "step30_v5_confirmation_runs_v1"
STATUS_PATH = STATE_ROOT / "run_status.json"
DEFAULT_SEEDS = (57721, 104729, 130363, 161027, 196613)
EXPECTED_N = 20_000
EXPECTED_FIELD_TIME = "0.8684190921"
EXPECTED_POLICY = "strict_v2"
EXPECTED_BACKEND = "cfd_topology"
EXECUTION_TARGET_ENV = "STEP30_EXECUTION_TARGET"
EXECUTION_TARGETS = {"dcu_cpu", "user_gpu"}
REMOTE_EXECUTION_MODE_ENV = "STEP30_REMOTE_EXECUTION"
REQUIRED_FILES = ("particle_hits.npz", "summary.json", "validation.png", "particle_run_manifest.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _assert_h_drive(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() == "H:":
        return resolved
    if os.name != "nt" and os.environ.get(REMOTE_EXECUTION_MODE_ENV) == "1":
        return resolved
    raise RuntimeError(
        f"{label} must remain on H: drive locally, or use a non-Windows remote "
        f"server with {REMOTE_EXECUTION_MODE_ENV}=1: {resolved}"
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _candidate_runs(artifact_root: Path) -> Iterable[Path]:
    policy_root = artifact_root / EXPECTED_POLICY
    if not policy_root.is_dir():
        return ()
    return (item for item in policy_root.iterdir() if item.is_dir() and not item.name.startswith("."))


def _inspect_complete_run(artifact_root: Path, seed: int) -> dict[str, Any] | None:
    for run_dir in _candidate_runs(artifact_root):
        if any(not (run_dir / name).is_file() for name in REQUIRED_FILES):
            continue
        try:
            summary = _load_json(run_dir / "summary.json")
            parameters = summary.get("parameters") or {}
            tracking = summary.get("tracking") or {}
            if int(parameters.get("n_parcels", -1)) != EXPECTED_N:
                continue
            if int(parameters.get("seed", -1)) != seed:
                continue
            if parameters.get("collision_policy") != EXPECTED_POLICY:
                continue
            if parameters.get("collision_backend") != EXPECTED_BACKEND:
                continue
            cfd = summary.get("cfd") or {}
            if str(cfd.get("field_time")) != EXPECTED_FIELD_TIME:
                continue
            if int(tracking.get("fallback_hits", -1)) != 0:
                continue
            manifest = _load_json(run_dir / "particle_run_manifest.json")
            outputs = manifest.get("outputs") or {}
            for key, filename in {
                "particle_hits": "particle_hits.npz",
                "summary": "summary.json",
                "validation": "validation.png",
            }.items():
                record = outputs.get(key) or {}
                if record.get("file") != filename or record.get("sha256") != _sha256(run_dir / filename):
                    raise ValueError(f"manifest hash mismatch for {run_dir / filename}")
            return {
                "status": "completed",
                "run_directory": str(run_dir.resolve()),
                "particle_hits_sha256": _sha256(run_dir / "particle_hits.npz"),
                "summary_sha256": _sha256(run_dir / "summary.json"),
                "manifest_sha256": _sha256(run_dir / "particle_run_manifest.json"),
                "validation_sha256": _sha256(run_dir / "validation.png"),
                "fallback_hits": int(tracking["fallback_hits"]),
            }
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return None


def _base_environment() -> dict[str, str]:
    target = os.environ.get(EXECUTION_TARGET_ENV, "").strip().lower()
    if target not in EXECUTION_TARGETS:
        raise RuntimeError(f"server-only runner requires {EXECUTION_TARGET_ENV}=dcu_cpu or user_gpu")
    if os.name == "nt" or os.environ.get(REMOTE_EXECUTION_MODE_ENV) != "1":
        raise RuntimeError(
            f"server-only runner requires a non-Windows remote host with "
            f"{REMOTE_EXECUTION_MODE_ENV}=1"
        )
    temp_root = _assert_h_drive(STATE_ROOT / "tmp", "temporary directory")
    mpl_root = _assert_h_drive(STATE_ROOT / "mplconfig", "matplotlib directory")
    pycache_root = _assert_h_drive(STATE_ROOT / "pycache", "Python cache directory")
    for path in (temp_root, mpl_root, pycache_root):
        path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "TEMP": str(temp_root),
            "TMP": str(temp_root),
            "TMPDIR": str(temp_root),
            "MPLCONFIGDIR": str(mpl_root),
            "PYTHONPYCACHEPREFIX": str(pycache_root),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            EXECUTION_TARGET_ENV: target,
            REMOTE_EXECUTION_MODE_ENV: "1",
        }
    )
    return env


def _validate_inputs(topology: Path, native_si: Path, transform: Path) -> None:
    for path, label in ((topology, "topology"), (native_si, "native SI directory"), (transform, "reference transform")):
        _assert_h_drive(path, label)
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")
    manifest = native_si / "native_si_manifest_0.8684190921.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing native SI manifest: {manifest}")
    document = _load_json(manifest)
    if str(document.get("field_time")) != EXPECTED_FIELD_TIME:
        raise ValueError(f"native SI manifest field time mismatch: {manifest}")


def _load_state() -> dict[str, Any]:
    if not STATUS_PATH.is_file():
        return {
            "version": "step30-particle-confirmation-runner-v1",
            "workspace": str(WORKSPACE),
            "status": "not_started",
            "seeds": {},
        }
    return _load_json(STATUS_PATH)


def run_confirmation(
    seeds: Iterable[int],
    *,
    topology: Path,
    native_si: Path,
    transform: Path,
    discovery_report: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    _assert_h_drive(WORKSPACE, "workspace")
    _assert_h_drive(OUTPUT, "output")
    topology = topology.resolve()
    native_si = native_si.resolve()
    transform = transform.resolve()
    discovery_report = discovery_report.resolve()
    _validate_inputs(topology, native_si, transform)
    if not discovery_report.is_file():
        raise FileNotFoundError(f"missing discovery report: {discovery_report}")
    discovery = _load_json(discovery_report)
    artifact_set_id = str((discovery.get("identity") or {}).get("artifact_set_id", ""))
    if not artifact_set_id:
        raise ValueError("discovery report has no artifact_set_id")
    artifact_root = OUTPUT / "step30_runs" / artifact_set_id
    _assert_h_drive(artifact_root, "immutable artifact root")
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    state.update(
        {
            "version": "step30-particle-confirmation-runner-v1",
            "workspace": str(WORKSPACE),
            "artifact_set_id": artifact_set_id,
            "topology": str(topology),
            "native_si_directory": str(native_si),
            "reference_transform": str(transform),
            "discovery_report": str(discovery_report),
            "expected": {
                "n_parcels": EXPECTED_N,
                "field_time": EXPECTED_FIELD_TIME,
                "collision_policy": EXPECTED_POLICY,
                "collision_backend": EXPECTED_BACKEND,
                "required_files": list(REQUIRED_FILES),
            },
        }
    )
    seed_list = list(dict.fromkeys(int(seed) for seed in seeds))
    for seed in seed_list:
        key = str(seed)
        existing = _inspect_complete_run(artifact_root, seed)
        if existing:
            state.setdefault("seeds", {})[key] = existing
            _json_write(STATUS_PATH, state)
            continue
        if dry_run:
            state.setdefault("seeds", {})[key] = {"status": "pending", "seed": seed}
            _json_write(STATUS_PATH, state)
            continue

        log_path = STATE_ROOT / f"seed_{seed}.log"
        started = time.time()
        record = {
            "status": "running",
            "seed": seed,
            "log": str(log_path),
            "started_epoch": started,
            "command": [
                sys.executable,
                "step30_lagrangian_particle_tracking.py",
                "--cfd", "native_si",
                "--collision-backend", EXPECTED_BACKEND,
                "--collision-policy", EXPECTED_POLICY,
                "--topology", str(topology),
                "--native-si-directory", str(native_si),
                "--field-time", EXPECTED_FIELD_TIME,
                "--analytic-reference-transform", str(transform),
                "--immutable-output",
                "--parcels", str(EXPECTED_N),
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
            ],
        }
        state.setdefault("seeds", {})[key] = record
        state["status"] = "running"
        _json_write(STATUS_PATH, state)
        env = _base_environment()
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            process = subprocess.run(
                record["command"],
                cwd=str(WORKSPACE),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record["returncode"] = process.returncode
        record["finished_epoch"] = time.time()
        record["elapsed_s"] = record["finished_epoch"] - started
        complete = _inspect_complete_run(artifact_root, seed)
        if process.returncode == 0 and complete:
            record.update(complete)
        else:
            record.update(
                {
                    "status": "failed",
                    "failure_reason": (
                        "tracker returned nonzero" if process.returncode != 0 else "missing or invalid immutable output"
                    ),
                }
            )
        state["seeds"][key] = record
        state["status"] = "running"
        _json_write(STATUS_PATH, state)

    completed = [value for value in state.get("seeds", {}).values() if value.get("status") == "completed"]
    state["status"] = "completed" if len(completed) == len(seed_list) and seed_list else "incomplete"
    state["completed_seed_count"] = len(completed)
    state["requested_seeds"] = seed_list
    state["updated_epoch"] = time.time()
    _json_write(STATUS_PATH, state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--topology", type=Path, default=OUTPUT / "step30_acceptance_0.8684190921" / "bucket_topology_v3_0.8684190921.npz")
    parser.add_argument("--native-si-directory", type=Path, default=OUTPUT / "native_bucket_si_urans_0.8684190921")
    parser.add_argument("--reference-transform", type=Path, default=OUTPUT / "authoritative_bucket_reference_transform_v1.json")
    parser.add_argument("--discovery-report", type=Path, default=OUTPUT / "step30_particle_convergence_5k_10k_20k_topology_aggregate_v5_discovery_r2.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execution-target", choices=sorted(EXECUTION_TARGETS), required=True)
    args = parser.parse_args(argv)
    os.environ[EXECUTION_TARGET_ENV] = args.execution_target
    result = run_confirmation(
        args.seeds or DEFAULT_SEEDS,
        topology=args.topology,
        native_si=args.native_si_directory,
        transform=args.reference_transform,
        discovery_report=args.discovery_report,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
