"""
step22_gpr_vs_mansouri.py
==========================
GPR 预测 vs Mansouri (2015) 真值场验证

指标:
  - Spearman ρ (空间秩相关)
  - 归一化 RMSE (nRMSE)
  - Top-10% 高危区 IoU
  - 逐点相对误差分布

输出:
  output/validation_gpr_vs_mansouri.json
  output/validation_scatter.png
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr

WORKSPACE  = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "output"
OUT_JSON   = OUTPUT_DIR / "validation_gpr_vs_mansouri.json"
OUT_PNG    = OUTPUT_DIR / "validation_scatter.png"


def run() -> dict:
    print("=" * 60)
    print("AEGIS Step-22  GPR vs Mansouri 验证")
    print("=" * 60)

    man  = np.load(str(OUTPUT_DIR / "vertices_full_mansouri.npz"))
    gpr  = np.load(str(OUTPUT_DIR / "vertices_full_gpr.npz"))

    ER_man  = man["ER_mansouri"].astype(np.float64)   # Mansouri 真值
    E_gpr   = gpr["E_final"].astype(np.float64)       # GPR 预测

    N = ER_man.shape[0]
    print(f"  顶点数: {N:,}")

    # 归一化到 [0,1]
    man_norm = ER_man / (ER_man.max() + 1e-30)
    gpr_norm = E_gpr  / (E_gpr.max()  + 1e-30)

    # Spearman
    rho, pval = spearmanr(man_norm, gpr_norm)

    # Pearson (归一化后)
    r_p, _ = pearsonr(man_norm, gpr_norm)

    # nRMSE
    rmse  = float(np.sqrt(np.mean((man_norm - gpr_norm) ** 2)))
    nrmse = rmse  # 已归一化，range=1

    # 逐点相对误差 (仅对 man_norm > 1e-3 的点)
    mask = man_norm > 1e-3
    rel_err = np.abs(man_norm[mask] - gpr_norm[mask]) / (man_norm[mask] + 1e-30)
    mean_rel = float(rel_err.mean())
    p50_rel  = float(np.median(rel_err))
    p95_rel  = float(np.percentile(rel_err, 95))

    # Top-10% IoU
    t_m = np.percentile(ER_man, 90)
    t_g = np.percentile(E_gpr,  90)
    hot_m = ER_man >= t_m
    hot_g = E_gpr  >= t_g
    iou = float(np.sum(hot_m & hot_g) / (np.sum(hot_m | hot_g) + 1e-10))

    print(f"\n  Spearman ρ          = {rho:.4f}  (p={pval:.2e})")
    print(f"  Pearson  r          = {r_p:.4f}")
    print(f"  nRMSE               = {nrmse:.4f}")
    print(f"  Mean rel error      = {mean_rel*100:.1f}%  (mask: man_norm>1e-3)")
    print(f"  Median rel error    = {p50_rel*100:.1f}%")
    print(f"  P95 rel error       = {p95_rel*100:.1f}%")
    print(f"  Top-10% IoU         = {iou:.4f}")

    # 目标判断
    target_rho  = rho  > 0.85
    target_iou  = iou  > 0.40
    target_nrmse = nrmse < 0.10
    print(f"\n  目标 ρ>0.85:  {'✓' if target_rho  else '✗'}  ({rho:.3f})")
    print(f"  目标 IoU>0.4: {'✓' if target_iou  else '✗'}  ({iou:.3f})")
    print(f"  目标 nRMSE<0.1:{'✓' if target_nrmse else '✗'}  ({nrmse:.4f})")

    # 散点图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GPR Prediction vs Mansouri (2015) Ground Truth", fontsize=12)

    # 散点（随机抽样 5000 点避免过密）
    idx = np.random.choice(N, min(5000, N), replace=False)
    axes[0].scatter(man_norm[idx], gpr_norm[idx], s=2, alpha=0.3, color="steelblue")
    lim = max(man_norm.max(), gpr_norm.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "r--", linewidth=1, label="y=x")
    axes[0].set_xlabel("Mansouri ER (normalized)")
    axes[0].set_ylabel("GPR E_final (normalized)")
    axes[0].set_title(f"Spearman ρ={rho:.3f}  nRMSE={nrmse:.3f}")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # 相对误差直方图
    axes[1].hist(rel_err * 100, bins=60, color="coral", edgecolor="white", linewidth=0.3)
    axes[1].axvline(mean_rel * 100, color="red", linestyle="--", linewidth=1,
                    label=f"mean={mean_rel*100:.1f}%")
    axes[1].axvline(5, color="green", linestyle=":", linewidth=1, label="5% target")
    axes[1].set_xlabel("Relative error [%]")
    axes[1].set_ylabel("Vertex count")
    axes[1].set_title("Point-wise relative error (man_norm > 1e-3)")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(OUT_PNG), dpi=150)
    plt.close()
    print(f"\n  → {OUT_PNG.name}")

    summary = {
        "model_truth":     "Mansouri (2015) Wear 332-333:1090-1097",
        "model_predicted": "GPR (step11_gpr_correction.py)",
        "n_vertices":      int(N),
        "metrics": {
            "spearman_rho":    round(float(rho),    4),
            "pearson_r":       round(float(r_p),    4),
            "nRMSE":           round(float(nrmse),  4),
            "mean_rel_err_pct":round(mean_rel*100,  2),
            "p50_rel_err_pct": round(p50_rel*100,   2),
            "p95_rel_err_pct": round(p95_rel*100,   2),
            "top10pct_iou":    round(iou,            4),
        },
        "targets_met": {
            "spearman_rho_gt_0.85":  bool(target_rho),
            "iou_gt_0.40":           bool(target_iou),
            "nRMSE_lt_0.10":         bool(target_nrmse),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {OUT_JSON.name}")
    print("\nStep-22 完成。\n")
    return summary


if __name__ == "__main__":
    run()
