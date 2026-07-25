"""
step13_sensitivity.py
=====================
灰盒 Layer-4：工况敏感性分析

对四个参数做单因素扫描，输出全局最短 T_RUL 对各参数的响应曲线。
直接复用 step10 的物理模型（不重跑 GPR，用修正因子均值近似）。

扫描参数
--------
  C_s  : 含沙量   [kg/m³]   0.3×1.31 ~ 3×1.31
  H    : 水头      [m]       500 ~ 700
  Dp   : 粒径      [μm]      20 ~ 200
  Hv   : 材料硬度  [GPa]     2.0 ~ 4.0

输出
----
output/sensitivity_curves.json
output/sensitivity_dashboard.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 复用 step10 的物理模型和 step12 的退化参数
from step10_unified_pipeline import (
    OkaModel, RunParams,
    compute_jet_velocity, compute_bucket_velocity,
    compute_impact_velocity, compute_impact_angle,
    N_RATED, OUTPUT_DIR,
)
from step12_degradation import (
    thickness_loss_rate, compute_rul,
    H0_DEFAULT, HMIN_DEFAULT, CS_DEFAULT, HOURS_PER_YEAR,
)

WORKSPACE = Path(__file__).parent
OUT_JSON  = OUTPUT_DIR / "sensitivity_curves.json"
OUT_PNG   = OUTPUT_DIR / "sensitivity_dashboard.png"

# GPR 修正因子均值（来自 step11 gpr_summary.json）
def _load_mean_correction() -> float:
    p = OUTPUT_DIR / "gpr_summary.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return float(d["correction_factor"]["mean"])
    return 1.0   # 无 GPR 结果时退化为纯物理


def min_rul_for_params(
    xyz: np.ndarray,
    normals: np.ndarray,
    Vp_base: np.ndarray,
    D1_mm: float,
    params: RunParams,
    correction: float,
) -> float:
    """给定工况参数，返回全局最短 T_RUL [h]。"""
    V_jet    = compute_jet_velocity(params.H)
    V_bucket = compute_bucket_velocity(D1_mm, N_RATED)
    Vp       = compute_impact_velocity(xyz, V_jet, V_bucket)
    alpha    = compute_impact_angle(xyz, normals)

    oka    = OkaModel(Hv=params.Hv, Dp=params.Dp)
    E_oka  = oka.erosion_rate(Vp, alpha) * correction

    dhdt   = thickness_loss_rate(E_oka, Vp, C_s=params.C_s)
    rul    = compute_rul(dhdt, H0_DEFAULT, HMIN_DEFAULT)
    finite = np.isfinite(rul)
    return float(rul[finite].min()) if finite.any() else np.inf


def run_sensitivity() -> None:
    print("=" * 60)
    print("AEGIS Step-13 工况敏感性分析")
    print("=" * 60)

    # 加载基准几何（真机转轮）
    data    = np.load(str(OUTPUT_DIR / "vertices_full.npz"))
    xyz     = data["xyz"]
    normals = data["normals"]
    Vp_base = data["Vp"]
    D1_mm   = float(data["D1_mm"]) if "D1_mm" in data else 2630.0
    correction = _load_mean_correction()
    print(f"GPR 修正因子均值: {correction:.4f}", flush=True)

    base = RunParams()   # 基准工况

    # ------------------------------------------------------------------
    # 扫描定义
    # ------------------------------------------------------------------
    scans = {
        "C_s":  np.linspace(0.3 * 1.31, 3.0 * 1.31, 10),
        "H":    np.linspace(500, 700, 10),
        "Dp":   np.linspace(20, 200, 8),
        "Hv":   np.linspace(2.0, 4.0, 6),
    }
    labels = {
        "C_s": "Sediment conc. C_s [kg/m3]",
        "H":   "Head H [m]",
        "Dp":  "Particle size Dp [um]",
        "Hv":  "Material hardness Hv [GPa]",
    }

    results: dict[str, dict] = {}

    for param, values in scans.items():
        print(f"  扫描 {param} ({len(values)} 点) ...", flush=True)
        rul_list = []
        for v in values:
            p = RunParams(
                H   = v   if param == "H"   else base.H,
                C_s = v   if param == "C_s" else base.C_s,
                Dp  = v   if param == "Dp"  else base.Dp,
                Hv  = v   if param == "Hv"  else base.Hv,
            )
            rul_h = min_rul_for_params(xyz, normals, Vp_base, D1_mm, p, correction)
            rul_list.append(rul_h)   # 保留小时

        results[param] = {
            "values":    values.tolist(),
            "rul_h":     rul_list,
            "label":     labels[param],
            "base_value": getattr(base, param),
        }
        print(f"    T_RUL range: [{min(rul_list):.1f}, {max(rul_list):.1f}] h", flush=True)

    # ------------------------------------------------------------------
    # 保存 JSON
    # ------------------------------------------------------------------
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"→ {OUT_JSON.name}", flush=True)

    # ------------------------------------------------------------------
    # 绘图
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("AEGIS Operating-Condition Sensitivity - Global Min RUL",
                 fontsize=13)

    for ax, (param, res) in zip(axes.flat, results.items()):
        ax.plot(res["values"], res["rul_h"], "o-", linewidth=1.8, markersize=5)
        ax.axvline(res["base_value"], color="red", linestyle="--",
                   linewidth=1, label=f"base={res['base_value']}")
        ax.set_xlabel(res["label"], fontsize=10)
        ax.set_ylabel("Min T_RUL [h]", fontsize=10)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(str(OUT_PNG), dpi=150)
    plt.close()
    print(f"→ {OUT_PNG.name}")
    print("Step-13 敏感性分析完成。\n")


if __name__ == "__main__":
    run_sensitivity()
