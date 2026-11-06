"""Candidate CuPy kernels for the non-collision particle update.

The kernels mirror the strict tracker formulas for drag response and
Schiller-Naumann response-time correction.  They do not implement CFD
interpolation, topology collision detection, impact response, or erosion
aggregation, so this module is a candidate acceleration component only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any


TARGET_ENV = "STEP30_EXECUTION_TARGET"
REMOTE_ENV = "STEP30_REMOTE_EXECUTION"
TARGET = "user_gpu"
VERSION = "step30-gpu-particle-kernel-candidate-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def drag_state_numpy(position_mm, velocity_ms, fluid_velocity_ms, tau_eff_s, dt_s, gravity_ms2=None):
    """Reference implementation copied from the strict tracker contract."""
    import numpy as np

    tau = np.maximum(np.asarray(tau_eff_s, dtype=np.float64), 1.0e-12)
    decay = np.exp(-dt_s / tau)[..., None]
    fluid = np.asarray(fluid_velocity_ms, dtype=np.float64)
    velocity = np.asarray(velocity_ms, dtype=np.float64)
    equilibrium = fluid if gravity_ms2 is None else fluid + np.asarray(gravity_ms2) * tau[..., None]
    new_velocity = equilibrium + (velocity - equilibrium) * decay
    displacement_m = equilibrium * dt_s + (velocity - equilibrium) * tau[..., None] * (1.0 - decay)
    return np.asarray(position_mm, dtype=np.float64) + displacement_m * 1000.0, new_velocity


def effective_tau_numpy(particle_tau_s, diameter_m, particle_velocity_ms, fluid_velocity_ms):
    import numpy as np

    slip = np.linalg.norm(np.asarray(fluid_velocity_ms) - np.asarray(particle_velocity_ms), axis=-1)
    reynolds = 1000.0 * slip * np.asarray(diameter_m) / (1000.0e-6) + 1.0e-12
    return np.maximum(np.asarray(particle_tau_s) / (1.0 + 0.15 * reynolds**0.687), 1.0e-12)


def drag_state_cupy(position_mm, velocity_ms, fluid_velocity_ms, tau_eff_s, dt_s, gravity_ms2=None):
    import cupy as cp

    tau = cp.maximum(cp.asarray(tau_eff_s, dtype=cp.float64), 1.0e-12)
    decay = cp.exp(-dt_s / tau)[:, None]
    fluid = cp.asarray(fluid_velocity_ms, dtype=cp.float64)
    velocity = cp.asarray(velocity_ms, dtype=cp.float64)
    equilibrium = fluid if gravity_ms2 is None else fluid + cp.asarray(gravity_ms2) * tau[:, None]
    new_velocity = equilibrium + (velocity - equilibrium) * decay
    displacement_m = equilibrium * dt_s + (velocity - equilibrium) * tau[:, None] * (1.0 - decay)
    return cp.asarray(position_mm, dtype=cp.float64) + displacement_m * 1000.0, new_velocity


def effective_tau_cupy(particle_tau_s, diameter_m, particle_velocity_ms, fluid_velocity_ms):
    import cupy as cp

    slip = cp.linalg.norm(cp.asarray(fluid_velocity_ms) - cp.asarray(particle_velocity_ms), axis=1)
    reynolds = 1000.0 * slip * cp.asarray(diameter_m) / (1000.0e-6) + 1.0e-12
    return cp.maximum(cp.asarray(particle_tau_s) / (1.0 + 0.15 * reynolds**0.687), 1.0e-12)


def run_candidate(output: Path, n: int = 200_000) -> dict[str, Any]:
    if os.name == "nt" or os.environ.get(REMOTE_ENV) != "1":
        raise RuntimeError(f"candidate kernel is server-only; set {REMOTE_ENV}=1")
    if os.environ.get(TARGET_ENV, "").strip().lower() != TARGET:
        raise RuntimeError(f"candidate kernel requires {TARGET_ENV}={TARGET}")
    if n <= 0:
        raise ValueError("n must be positive")

    import cupy as cp
    import numpy as np

    source = Path(__file__).resolve()
    rng = np.random.default_rng(20260831)
    position = rng.normal(0.0, 20.0, size=(n, 3)).astype(np.float64)
    velocity = rng.normal(0.0, 100.0, size=(n, 3)).astype(np.float64)
    fluid = rng.normal(0.0, 100.0, size=(n, 3)).astype(np.float64)
    particle_tau = rng.uniform(1.0e-5, 2.0e-3, size=n).astype(np.float64)
    diameter = rng.uniform(1.0e-6, 2.3e-4, size=n).astype(np.float64)
    gravity = np.asarray([0.0, 0.0, -9.81], dtype=np.float64)
    dt = 1.0e-5

    tau_cpu = effective_tau_numpy(particle_tau, diameter, velocity, fluid)
    pos_cpu, vel_cpu = drag_state_numpy(position, velocity, fluid, tau_cpu, dt, gravity)
    device = cp.cuda.Device(0)
    device.use()
    started = time.time()
    tau_gpu = effective_tau_cupy(particle_tau, diameter, velocity, fluid)
    pos_gpu, vel_gpu = drag_state_cupy(position, velocity, fluid, tau_gpu, dt, gravity)
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.time() - started
    tau_gpu_host = cp.asnumpy(tau_gpu)
    pos_gpu_host = cp.asnumpy(pos_gpu)
    vel_gpu_host = cp.asnumpy(vel_gpu)

    arrays = {
        "tau": (tau_cpu, tau_gpu_host),
        "position": (pos_cpu, pos_gpu_host),
        "velocity": (vel_cpu, vel_gpu_host),
    }
    errors = {
        name: {
            "max_abs": float(np.max(np.abs(cpu - gpu))),
            "max_rel": float(np.max(np.abs(cpu - gpu) / np.maximum(np.abs(cpu), 1.0e-30))),
        }
        for name, (cpu, gpu) in arrays.items()
    }
    passed = all(row["max_abs"] <= 2.0e-10 for row in errors.values())
    digest = hashlib.sha256()
    for name in ("tau", "position", "velocity"):
        digest.update(np.ascontiguousarray(arrays[name][1]).tobytes())
    report = {
        "version": VERSION,
        "status": "passed" if passed else "failed",
        "candidate_only": True,
        "physics_backend_validated": False,
        "competition_evidence_eligible": False,
        "execution_target": TARGET,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "cupy": cp.__version__,
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "device": str(cp.cuda.runtime.getDeviceProperties(0).get("name", b"")).replace("b'", "").rstrip("'"),
        },
        "contract": {
            "integrator_formula": "strict_v2 drag_state + Schiller-Naumann tau correction",
            "n": n,
            "dt_s": dt,
            "gravity_ms2": gravity.tolist(),
            "random_seed": 20260831,
        },
        "errors": errors,
        "gpu_result_sha256": digest.hexdigest(),
        "gpu_elapsed_s": elapsed,
        "source": {"file": source.name, "sha256": _sha256(source)},
        "limitations": [
            "CFD interpolation is not ported.",
            "Topology collision detection and impact response are not ported.",
            "This candidate cannot replace strict_v2 or change a competition score.",
        ],
    }
    _json_write(output, report)
    return report


def _json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output") / "step30_gpu_particle_kernel_candidate_v1.json")
    parser.add_argument("--n", type=int, default=200_000)
    args = parser.parse_args(argv)
    report = run_candidate(args.output, args.n)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
