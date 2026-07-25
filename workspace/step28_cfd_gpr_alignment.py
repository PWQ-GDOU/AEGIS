"""
step28_cfd_gpr_alignment.py
=============================
CFD (rebuild) runner wall pressure vs GPR-predicted erosion field
空间对齐与交叉验证。

对齐策略:
  - CFD runner 表面点 (38593) 与 GPR 顶点云 (154180) 均来自同一 STL，
    但 blockMesh 域可能裁剪了部分几何 → 各轴独立仿射对齐（scale + offset）
  - 对齐后用 KDTree 最近邻，将 CFD 压力场插值到 GPR 顶点
  - 用 |p - p_atm| 作为壁面载荷强度代理，与 GPR E_final 做 Spearman 相关

注：CFD 简化模型（层流、简化射流）不能作为精确 ground truth，
    此步骤的目的是"方向性验证"——检查高压区与高侵蚀预测区是否空间重合。
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

WORKSPACE  = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "output"
OUT_JSON   = OUTPUT_DIR / "cfd_gpr_alignment_summary.json"


def run() -> dict:
    print("=" * 60)
    print("AEGIS Step-28  CFD ↔ GPR 空间对齐验证")
    print("=" * 60)

    p_data = np.loadtxt(str(OUTPUT_DIR / "rebuild_p_runner.raw"), comments="#")
    cfd_xyz = p_data[:, :3]
    cfd_p   = p_data[:, 3]

    gpr = np.load(str(OUTPUT_DIR / "vertices_full_gpr.npz"))
    gpr_xyz = gpr["xyz"].astype(np.float64)
    E_final = gpr["E_final"].astype(np.float64)

    print(f"  CFD runner 点数: {len(cfd_xyz):,}")
    print(f"  GPR 顶点数:      {len(gpr_xyz):,}")

    # 各轴独立仿射对齐（scale + center）
    cfd_extent = cfd_xyz.max(axis=0) - cfd_xyz.min(axis=0)
    gpr_extent = gpr_xyz.max(axis=0) - gpr_xyz.min(axis=0)
    scale = gpr_extent / cfd_extent
    cfd_center = cfd_xyz.mean(axis=0)
    gpr_center = gpr_xyz.mean(axis=0)
    cfd_aligned = (cfd_xyz - cfd_center) * scale + gpr_center

    print(f"  各轴缩放: {scale}")

    # KDTree 最近邻：把 CFD 压力场映射到 GPR 顶点
    tree = cKDTree(cfd_aligned)
    dist, idx = tree.query(gpr_xyz, k=1)
    print(f"  最近邻距离: 中位={np.median(dist):.1f}mm  P90={np.percentile(dist,90):.1f}mm")

    p_on_gpr = cfd_p[idx]

    # 载荷强度代理（|p| 越大冲击越强）
    load_proxy = np.abs(p_on_gpr)

    # 只用距离合理的点（排除对齐失败的外点）
    valid = dist < np.percentile(dist, 95)
    print(f"  有效匹配点: {valid.sum():,} / {len(gpr_xyz):,}")

    rho, pval = spearmanr(load_proxy[valid], E_final[valid])
    print(f"\n  Spearman ρ(|p_CFD|, E_final_GPR) = {rho:.4f}  (p={pval:.2e})")

    # 高危区重叠
    t_load = np.percentile(load_proxy[valid], 90)
    t_E    = np.percentile(E_final[valid], 90)
    hot_load = load_proxy[valid] >= t_load
    hot_E    = E_final[valid] >= t_E
    iou = float(np.sum(hot_load & hot_E) / (np.sum(hot_load | hot_E) + 1e-10))
    print(f"  Top-10% 高危区 IoU: {iou:.4f}")

    summary = {
        "method": "anisotropic per-axis affine alignment + nearest-neighbor",
        "cfd_note": "laminar simpleFoam, simplified inlet, NOT precise ground truth — directional check only",
        "scale_factors": scale.tolist(),
        "n_cfd_points": int(len(cfd_xyz)),
        "n_gpr_points": int(len(gpr_xyz)),
        "n_valid_matches": int(valid.sum()),
        "median_nn_distance_mm": round(float(np.median(dist)), 1),
        "spearman_rho_abs_p_vs_E_final": round(float(rho), 4),
        "spearman_pval": float(pval),
        "top10pct_iou": round(iou, 4),
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  → {OUT_JSON.name}")
    print("\nStep-28 完成。\n")
    return summary


if __name__ == "__main__":
    run()
