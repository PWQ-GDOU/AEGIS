"""
step24_gpr_physics_features.py
================================
用物理特征 (Vp, alpha) 替代 xyz 坐标训练 GPR，
直接拟合 Mansouri (2015) 的速度-角度响应曲线。

核心思路:
  Mansouri ER = k * Fs * Vp^n * f(alpha)
  → ER 完全由 (Vp, alpha) 决定，与 xyz 无关
  → GPR 用 (Vp, alpha) 作特征可以精确重建 Mansouri 曲面
  → 验证指标应接近 1.0（上界）

同时输出:
  - 用 (Vp, alpha, xyz_norm) 联合特征的版本（空间平滑 + 物理驱动）

输出:
  output/gpr_physics_summary.json
  output/gpr_physics_validation.png
"""

from __future__ import annotations
import json, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

WORKSPACE  = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "output"
OUT_JSON   = OUTPUT_DIR / "gpr_physics_summary.json"
OUT_PNG    = OUTPUT_DIR / "gpr_physics_validation.png"

N_INDUCING  = 2000
RANDOM_SEED = 42
N_RESTARTS  = 2


def train_and_eval(X_all, y_log, label, rng, n_inducing=N_INDUCING):
    """训练 GPR 并返回全量预测和指标。"""
    idx = rng.choice(len(X_all), size=min(n_inducing, len(X_all)), replace=False)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    kernel = (ConstantKernel(1.0, (1e-3, 10.0))
              * RBF(1.0, (0.1, 10.0))
              + WhiteKernel(0.01, (1e-6, 1.0)))
    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=N_RESTARTS,
        normalize_y=True, random_state=RANDOM_SEED, alpha=1e-8,
    )
    t0 = time.time()
    gpr.fit(X_scaled[idx], y_log[idx])
    print(f"  [{label}] 训练 {len(idx)} 点，耗时 {time.time()-t0:.1f}s，核: {gpr.kernel_}")

    BATCH = 20_000
    pred = np.zeros(len(X_all))
    for s in range(0, len(X_all), BATCH):
        e = min(s + BATCH, len(X_all))
        pred[s:e] = gpr.predict(X_scaled[s:e])
    return pred


def run() -> dict:
    print("=" * 60)
    print("AEGIS Step-24  物理特征 GPR 验证")
    print("=" * 60)
    rng = np.random.default_rng(RANDOM_SEED)

    man = np.load(str(OUTPUT_DIR / "vertices_full_mansouri.npz"))
    gpr_data = np.load(str(OUTPUT_DIR / "vertices_full_gpr.npz"))

    ER_man = man["ER_mansouri"].astype(np.float64)
    Vp     = gpr_data["Vp"].astype(np.float64)
    alpha  = gpr_data["alpha"].astype(np.float64)
    xyz    = gpr_data["xyz"].astype(np.float64)
    N = len(ER_man)

    ER_max  = ER_man.max()
    ER_norm = ER_man / (ER_max + 1e-30)
    eps     = 1e-8
    y_log   = np.log(np.clip(ER_norm, eps, 1.0))

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("GPR with Physics Features vs Mansouri Ground Truth", fontsize=11)

    configs = [
        ("Vp+alpha",      np.column_stack([Vp, alpha])),
        ("Vp+alpha+xyz",  np.column_stack([Vp, alpha, xyz])),
    ]

    for i, (label, X_all) in enumerate(configs):
        pred_log  = train_and_eval(X_all, y_log, label, rng)
        pred_norm = np.exp(pred_log)
        pred_norm = pred_norm / (pred_norm.max() + 1e-30)

        rho, _  = spearmanr(ER_norm, pred_norm)
        r_p, _  = pearsonr(ER_norm, pred_norm)
        nrmse   = float(np.sqrt(np.mean((ER_norm - pred_norm)**2)))
        mask    = ER_norm > 1e-3
        rel_err = np.abs(ER_norm[mask] - pred_norm[mask]) / (ER_norm[mask] + 1e-30)
        mean_rel = float(rel_err.mean())
        t_m = np.percentile(ER_man, 90)
        pred_abs = np.exp(pred_log) * ER_max
        t_p = np.percentile(pred_abs, 90)
        iou = float(np.sum((ER_man >= t_m) & (pred_abs >= t_p))
                    / (np.sum((ER_man >= t_m) | (pred_abs >= t_p)) + 1e-10))

        print(f"\n  [{label}]")
        print(f"    Spearman ρ   = {rho:.4f}")
        print(f"    Pearson  r   = {r_p:.4f}")
        print(f"    nRMSE        = {nrmse:.4f}")
        print(f"    Mean rel err = {mean_rel*100:.1f}%")
        print(f"    Top-10% IoU  = {iou:.4f}")
        print(f"    目标 ρ>0.85:   {'✓' if rho>0.85  else '✗'}")
        print(f"    目标 IoU>0.40: {'✓' if iou>0.40  else '✗'}")
        print(f"    目标 nRMSE<0.1:{'✓' if nrmse<0.10 else '✗'}")

        results[label] = {
            "spearman_rho": round(float(rho), 4),
            "pearson_r":    round(float(r_p), 4),
            "nRMSE":        round(float(nrmse), 4),
            "mean_rel_err_pct": round(mean_rel*100, 2),
            "top10pct_iou": round(iou, 4),
            "targets_met": {
                "rho_gt_0.85":   bool(rho > 0.85),
                "iou_gt_0.40":   bool(iou > 0.40),
                "nRMSE_lt_0.10": bool(nrmse < 0.10),
            },
        }

        idx_s = rng.choice(N, min(3000, N), replace=False)
        axes[i].scatter(ER_norm[idx_s], pred_norm[idx_s], s=2, alpha=0.3, color="steelblue")
        axes[i].plot([0, 1], [0, 1], "r--", lw=1, label="y=x")
        axes[i].set_xlabel("Mansouri ER (norm)")
        axes[i].set_ylabel("GPR pred (norm)")
        axes[i].set_title(f"{label}\nρ={rho:.3f}  nRMSE={nrmse:.3f}")
        axes[i].legend(fontsize=8); axes[i].grid(True, alpha=0.3)

    # 第3图：Mansouri 解析值 vs 直接计算（上界参考）
    from step21_mansouri_ground_truth import erosion_rate_mansouri
    ER_direct = erosion_rate_mansouri(Vp, alpha)
    ER_direct_norm = ER_direct / (ER_direct.max() + 1e-30)
    rho_direct, _ = spearmanr(ER_norm, ER_direct_norm)
    idx_s = rng.choice(N, min(3000, N), replace=False)
    axes[2].scatter(ER_norm[idx_s], ER_direct_norm[idx_s], s=2, alpha=0.3, color="green")
    axes[2].plot([0, 1], [0, 1], "r--", lw=1, label="y=x")
    axes[2].set_xlabel("Mansouri ER (norm, from npz)")
    axes[2].set_ylabel("Mansouri ER (norm, recomputed)")
    axes[2].set_title(f"Self-consistency check\nρ={rho_direct:.4f}")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
    print(f"\n  [自洽检验] Mansouri npz vs 重算: ρ={rho_direct:.4f}")
    results["self_consistency_rho"] = round(float(rho_direct), 4)

    plt.tight_layout()
    plt.savefig(str(OUT_PNG), dpi=150)
    plt.close()
    print(f"\n  → {OUT_PNG.name}")

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {OUT_JSON.name}")
    print("\nStep-24 完成。\n")
    return results


if __name__ == "__main__":
    run()
