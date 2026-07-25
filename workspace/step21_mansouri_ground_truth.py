"""
step21_mansouri_ground_truth.py
================================
Mansouri (2015) 侵蚀模型 — 合成真值生成器

公式来源: Mansouri et al., Wear 332-333 (2015) 1090-1097
  ER = k * Fs * Vp^n * f(α)
  f(α) = a1 * sin(α)^b1 * (1.5 - sin(α))^b2

不锈钢 316 拟合系数 (Table 3):
  k=2.57e-9  n=2.4  a1=228.5  b1=7.91  b2=12.81
  (k 量纲: [kg/kg] / ([m/s]^n), 原文 k=2.57 对应 Vp 单位 m/s, ER 单位 kg/kg)

注: 原文 k=2.57 是在 Vp[m/s] 量纲下拟合的无量纲侵蚀比 [kg/kg]。
    实际数量级: Vp~40m/s, ER~2.57*(40^2.4)*f(α) ≈ 2.57*5000*f(α)
    → 需乘以量纲修正因子使 ER 落在合理范围 [1e-8, 1e-4] kg/kg
    → 使用 k = 2.57e-9 (与 Oka 量级对齐, 文献常见做法)

输出:
  output/vertices_full_mansouri.npz  — 逐顶点 Mansouri 侵蚀率真值场
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "output"

# Mansouri (2015) Table 3 — Stainless Steel 316
K_MANSOURI = 2.57e-9   # 量纲修正后的 k [kg/kg / (m/s)^n]
N_VEL      = 2.4       # 速度指数
A1         = 228.5     # 角度函数系数
B1         = 7.91
B2         = 12.81
FS_SHARP   = 1.0       # 尖锐颗粒 Fs=1.0

OUT_NPZ  = OUTPUT_DIR / "vertices_full_mansouri.npz"
OUT_JSON = OUTPUT_DIR / "mansouri_summary.json"


def angle_function(alpha_rad: np.ndarray) -> np.ndarray:
    """f(α) = a1 * sin(α)^b1 * (1.5 - sin(α))^b2"""
    s = np.clip(np.sin(alpha_rad), 1e-6, 1.0)
    term2 = np.clip(1.5 - s, 1e-6, None)
    return A1 * (s ** B1) * (term2 ** B2)


def erosion_rate_mansouri(Vp: np.ndarray, alpha_deg: np.ndarray) -> np.ndarray:
    """
    ER [kg/kg] = k * Fs * Vp^n * f(α)
    alpha_deg: 冲击角 [°]
    Vp:        冲击速度 [m/s]
    """
    alpha_rad = np.deg2rad(np.clip(alpha_deg, 0.1, 89.9))
    fa = angle_function(alpha_rad)
    return K_MANSOURI * FS_SHARP * (Vp ** N_VEL) * fa


def run() -> dict:
    print("=" * 60)
    print("AEGIS Step-21  Mansouri (2015) 合成真值生成")
    print("=" * 60)

    # 加载 GPR 数据（含 Vp, alpha, xyz）
    gpr = np.load(str(OUTPUT_DIR / "vertices_full_gpr.npz"))
    xyz   = gpr["xyz"].astype(np.float64)
    Vp    = gpr["Vp"].astype(np.float64)
    alpha = gpr["alpha"].astype(np.float64)   # [°]
    E_oka = gpr["E_oka"].astype(np.float64)

    N = xyz.shape[0]
    print(f"  顶点数: {N:,}")
    print(f"  Vp  范围: [{Vp.min():.1f}, {Vp.max():.1f}] m/s")
    print(f"  α   范围: [{alpha.min():.1f}, {alpha.max():.1f}] °")

    # 计算 Mansouri 侵蚀率
    ER = erosion_rate_mansouri(Vp, alpha)

    # 统计
    print(f"\n  Mansouri ER 范围: [{ER.min():.3e}, {ER.max():.3e}] kg/kg")
    print(f"  Mansouri ER 中位: {np.median(ER):.3e}")
    print(f"  Oka      ER 中位: {np.median(E_oka):.3e}")

    # 归一化对比（用于 GPR 验证）
    ER_norm  = ER / (ER.max() + 1e-30)
    Oka_norm = E_oka / (E_oka.max() + 1e-30)

    from scipy.stats import spearmanr
    rho, pval = spearmanr(ER_norm, Oka_norm)
    print(f"\n  Mansouri vs Oka Spearman ρ = {rho:.4f}  (p={pval:.2e})")

    # 高危区（top 10%）重叠率
    thresh_m = np.percentile(ER, 90)
    thresh_o = np.percentile(E_oka, 90)
    hot_m = ER >= thresh_m
    hot_o = E_oka >= thresh_o
    overlap = np.sum(hot_m & hot_o) / np.sum(hot_m | hot_o)
    print(f"  Top-10% 高危区 IoU: {overlap:.4f}")

    # 保存
    np.savez_compressed(
        str(OUT_NPZ),
        xyz=xyz.astype(np.float32),
        Vp=Vp.astype(np.float32),
        alpha=alpha.astype(np.float32),
        ER_mansouri=ER.astype(np.float32),
        ER_mansouri_norm=ER_norm.astype(np.float32),
        E_oka=E_oka.astype(np.float32),
        E_oka_norm=Oka_norm.astype(np.float32),
    )
    print(f"\n  → {OUT_NPZ.name}")

    summary = {
        "model": "Mansouri (2015) Wear 332-333:1090-1097",
        "coefficients": {"k": K_MANSOURI, "n": N_VEL, "a1": A1, "b1": B1, "b2": B2, "Fs": FS_SHARP},
        "n_vertices": int(N),
        "ER_stats": {
            "min": float(ER.min()), "max": float(ER.max()),
            "median": float(np.median(ER)), "mean": float(ER.mean()),
            "p90": float(np.percentile(ER, 90)),
        },
        "validation": {
            "spearman_rho_vs_oka": round(float(rho), 4),
            "spearman_pval": float(pval),
            "top10pct_iou_vs_oka": round(float(overlap), 4),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {OUT_JSON.name}")
    print("\nStep-21 完成。\n")
    return summary


if __name__ == "__main__":
    run()
