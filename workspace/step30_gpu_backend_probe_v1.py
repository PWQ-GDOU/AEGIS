"""Fail-closed user-GPU capability probe for Step30.

This probe is deliberately not a particle tracker.  It records only whether
the approved user GPU server can execute a deterministic CuPy workload.  A
passing probe never promotes the GPU path to a physics or competition result.
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
VERSION = "step30-gpu-backend-probe-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_probe(*, output: Path, elements: int = 1_000_000) -> dict[str, Any]:
    if os.name == "nt" or os.environ.get(REMOTE_ENV) != "1":
        raise RuntimeError(
            f"GPU probe is server-only; set {REMOTE_ENV}=1 on the approved server"
        )
    if os.environ.get(TARGET_ENV, "").strip().lower() != TARGET:
        raise RuntimeError(f"GPU probe requires {TARGET_ENV}={TARGET}")
    if elements <= 0:
        raise ValueError("elements must be positive")

    try:
        import cupy as cp
        import numpy as np
    except Exception as exc:  # pragma: no cover - exercised on unconfigured hosts
        raise RuntimeError(f"CuPy import failed: {exc}") from exc

    source = Path(__file__).resolve()
    device_count = int(cp.cuda.runtime.getDeviceCount())
    if device_count < 1:
        raise RuntimeError("no CUDA devices reported")

    device_rows: list[dict[str, Any]] = []
    host_results: list[np.ndarray] = []
    started = time.time()
    for device_id in range(device_count):
        device = cp.cuda.Device(device_id)
        device.use()
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        name = props.get("name", b"")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")

        # Fixed arithmetic, no RNG: suitable for a capability/determinism gate.
        x = cp.arange(1, elements + 1, dtype=cp.float64)
        y = (x * 0.125) + 3.0
        values = cp.asarray(
            [cp.sum(x), cp.sum(y), cp.sum(x * y), cp.max(y)], dtype=cp.float64
        )
        cp.cuda.runtime.deviceSynchronize()
        host = cp.asnumpy(values)
        host_results.append(host)
        device_rows.append(
            {
                "device_id": device_id,
                "name": str(name),
                "compute_capability": f"{device.compute_capability[0]}.{device.compute_capability[1]}",
                "total_memory_bytes": int(device.mem_info[1]),
                "free_memory_bytes_after_probe": int(device.mem_info[0]),
                "result": [float(value) for value in host],
                "result_sha256": hashlib.sha256(host.tobytes()).hexdigest(),
            }
        )
        del x, y, values

    reference = host_results[0]
    cross_device_equal = all(np.array_equal(reference, value) for value in host_results[1:])
    source_hash = _sha256(source)
    report = {
        "version": VERSION,
        "status": "passed" if cross_device_equal else "failed",
        "probe_only": True,
        "physics_backend_validated": False,
        "competition_evidence_eligible": False,
        "execution_target": TARGET,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "cupy": cp.__version__,
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "device_count": device_count,
        },
        "workload": {
            "elements": elements,
            "arithmetic": "fixed float64 arange/scale/sum/dot/max; no RNG",
            "elapsed_s": time.time() - started,
        },
        "devices": device_rows,
        "cross_device_equal": cross_device_equal,
        "source": {"file": source.name, "sha256": source_hash},
        "limitations": [
            "This does not execute particle tracking.",
            "This does not validate topology collision semantics.",
            "This does not authorize replacement of the CPU strict_v2 tracker.",
        ],
    }
    _json_write(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "step30_gpu_backend_probe_v1.json",
    )
    parser.add_argument("--elements", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    report = run_probe(output=args.output, elements=args.elements)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
