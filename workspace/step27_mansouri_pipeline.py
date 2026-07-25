"""
step27_mansouri_pipeline.py
============================
重构：直接用 Mansouri (2015) 解析公式作为侵蚀率预测，
GPR 仅用于量化三模型分歧作为不确定度估计。

架构：
  E_final(x)  = Mansouri(Vp, alpha)          [确定性预测]
  sigma(x)    = std(Oka_norm, Finnie_norm, Tabakoff_norm, Mansouri_norm)  [不确定度]
  E_upper_95  = E_final * (1 + 2*sigma_norm)
  E_lower_95  = E_final * max(0, 1 - 2*sigma_norm)

输出：
  output/vertices_full_gpr.npz  — 更新 E_final / sigma / correction_factor
  output/mansouri_pipeline_summary.json
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

WORKSPACE  = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "output"
OUT_JSON   = OUTPUT_DIR / "mansouri_pipeline_summary.json"

# Mansouri (2015) Table 3 — Stainless Steel 316
K  = 2.57e-9
N_VEL = 2.4
A1, B1, B2 = 228.5, 7.91, 12.81
FS = 1.0


def mansouri_er(Vp: np.ndarray, alpha_deg: np.ndarray) -> np.ndarray:
    a = np.deg2rad(np.clip(alpha_deg, 0.1, 89.9))
    s = np.clip(np.sin(a), 1e-6, 1.0)
    fa = A1 * (s ** B1) * np.clip(1.5 - s, 1e-6, None) ** B2
    return K * FS * (Vp ** N_VEL) * fa


def run() -> dict:
    print("=" * 60)
    print("AEGIS Step-27  Mansouri 解析管道重构")
    print("=" * 60)

    old = np.load(str(OUTPUT_DIR / "vertices_full_gpr.npz"))
    Vp    = old["Vp"].astype(np.float64)
    alpha = old["alpha"].astype(np.float64)
    E_oka = old["E_oka"].astype(np.float64)

    # 加载其他物理模型（来自 vertices_full.npz）
    full = np.load(str(OUTPUT_DIR / "vertices_full.npz"))
    E_finnie   = full["E_finnie"].astype(np.float64)
    E_tabakoff = full["E_tabakoff"].astype(np.float64)

    N = len(Vp)

    # 1. Mansouri 解析侵蚀率
    E_final = mansouri_er(Vp, alpha)

    # 2. 不确定度：四模型归一化后的标准差
    def norm(x): return x / (x.max() + 1e-30)
    stack = np.stack([norm(E_oka), norm(E_finnie), norm(E_tabakoff), norm(E_final)], axis=0)
    sigma_norm = stack.std(axis=0)          # 归一化空间的标准差 ∈ [0,1]

    E_final_norm = norm(E_final)
    E_upper_95   = E_final * (1 + 2 * sigma_norm)
    E_lower_95   = np.clip(E_final * (1 - 2 * sigma_norm), 0.0, None)
    sigma_E      = E_final * sigma_norm

    # 3. 修正因子（相对 Oka，用于 step12/16 兼容）
    correction_factor = E_final / (E_oka + 1e-30)

    # 4. 验证：与 Mansouri npz 自洽
    man = np.load(str(OUTPUT_DIR / "vertices_full_mansouri.npz"))
    ER_truth = man["ER_mansouri"].astype(np.float64)
    rho, _   = spearmanr(norm(ER_truth), E_final_norm)
    nrmse    = float(np.sqrt(np.mean((norm(ER_truth) - E_final_norm)**2)))
    t_m = np.percentile(ER_truth, 90)
    t_p = np.percentile(E_final, 90)
    iou = float(np.sum((ER_truth >= t_m) & (E_final >= t_p))
                / (np.sum((ER_truth >= t_m) | (E_final >= t_p)) + 1e-10))

    print(f"  N = {N:,}")
    print(f"  E_final 范围: [{E_final.min():.3e}, {E_final.max():.3e}] kg/kg")
    print(f"  sigma_norm 中位: {np.median(sigma_norm):.4f}")
    print(f"  自洽验证: ρ={rho:.4f}  nRMSE={nrmse:.4f}  IoU={iou:.4f}")

    # 5. 更新 vertices_full_gpr.npz
    updated = dict(old)
    updated.update({
        "E_final":           E_final.astype(np.float32),
        "E_final_norm":      E_final_norm.astype(np.float32),
        "correction_factor": correction_factor.astype(np.float32),
        "sigma":             sigma_E.astype(np.float32),
        "sigma_norm":        sigma_norm.astype(np.float32),
        "E_upper_95":        E_upper_95.astype(np.float32),
        "E_lower_95":        E_lower_95.astype(np.float32),
        "ER_mansouri_truth": ER_truth.astype(np.float32),
    })
    np.savez_compressed(str(OUTPUT_DIR / "vertices_full_gpr.npz"), **updated)
    print(f"  → vertices_full_gpr.npz 已更新（E_final = Mansouri 解析值）")

    summary = {
        "pipeline": "Mansouri (2015) analytic formula — no GPR approximation",
        "uncertainty": "std of 4-model normalized ensemble (Oka/Finnie/Tabakoff/Mansouri)",
        "self_consistency": {"rho": round(float(rho), 4), "nRMSE": round(float(nrmse), 4),
                             "iou": round(float(iou), 4)},
        "E_final_stats": {"min": float(E_final.min()), "max": float(E_final.max()),
                          "median": float(np.median(E_final))},
        "sigma_norm_median": round(float(np.median(sigma_norm)), 4),
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {OUT_JSON.name}")
    print("\nStep-27 完成。\n")
    return summary


if __name__ == "__main__":
    run()
