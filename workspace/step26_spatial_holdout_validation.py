"""
step26_spatial_holdout_validation.py
=====================================
空间留出验证：按水斗区域划分训练/测试集，
验证 GPR(Vp, alpha) 在未见区域的空间泛化能力。

划分策略：
  - 按 z 坐标四分位数将水斗分为 4 个空间区域
  - 依次留出每个区域作测试集，其余 3 个区域训练
  - 报告每折的 Spearman ρ、nRMSE、Top-10% IoU

输出:
  output/spatial_holdout_results.json
  output/spatial_holdout_validation.png
"""

from __future__ import annotations
import json, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

WORKSPACE  = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "output"
OUT_JSON   = OUTPUT_DIR / "spatial_holdout_results.json"
OUT_PNG    = OUTPUT_DIR / "spatial_holdout_validation.png"

N_INDUCING  = 1500   # 每折训练点数（3/4 区域中采样）
RANDOM_SEED = 42
N_FOLDS     = 4


def run() -> dict:
    print("=" * 60)
    print("AEGIS Step-26  空间留出交叉验证")
    print("=" * 60)
    rng = np.random.default_rng(RANDOM_SEED)

    man     = np.load(str(OUTPUT_DIR / "vertices_full_mansouri.npz"))
    gpr_dat = np.load(str(OUTPUT_DIR / "vertices_full_gpr.npz"))

    ER_man = man["ER_mansouri"].astype(np.float64)
    Vp     = gpr_dat["Vp"].astype(np.float64)
    alpha  = gpr_dat["alpha"].astype(np.float64)
    xyz    = gpr_dat["xyz"].astype(np.float64)
    N = len(ER_man)

    ER_max  = ER_man.max()
    ER_norm = ER_man / (ER_max + 1e-30)
    eps     = 1e-8
    y_log   = np.log(np.clip(ER_norm, eps, 1.0))

    X_all  = np.column_stack([Vp, alpha])

    # 按 z 坐标四分位数划分空间区域
    z = xyz[:, 2]
    q = np.percentile(z, [25, 50, 75])
    region = np.zeros(N, dtype=int)
    region[z >= q[0]] = 1
    region[z >= q[1]] = 2
    region[z >= q[2]] = 3
    print(f"  区域分布: " + "  ".join(f"R{i}:{np.sum(region==i):,}" for i in range(4)))

    fold_results = []
    fig, axes = plt.subplots(1, N_FOLDS, figsize=(16, 4))
    fig.suptitle("Spatial Hold-out CV: GPR(Vp,α) vs Mansouri", fontsize=11)

    for fold in range(N_FOLDS):
        test_mask  = region == fold
        train_mask = ~test_mask

        X_train_all = X_all[train_mask]
        y_train_all = y_log[train_mask]

        # 从训练区域采样
        n_train = min(N_INDUCING, train_mask.sum())
        idx = rng.choice(len(X_train_all), size=n_train, replace=False)

        scaler    = StandardScaler()
        X_tr_sc   = scaler.fit_transform(X_train_all[idx])
        X_te_sc   = scaler.transform(X_all[test_mask])

        kernel = (ConstantKernel(1.0, (1e-3, 10.0))
                  * RBF(1.0, (0.1, 10.0))
                  + WhiteKernel(0.01, (1e-6, 1.0)))
        gpr = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=2,
            normalize_y=True, random_state=RANDOM_SEED, alpha=1e-8,
        )
        t0 = time.time()
        gpr.fit(X_tr_sc, y_train_all[idx])
        pred_log = gpr.predict(X_te_sc)
        elapsed  = time.time() - t0

        pred_norm = np.exp(pred_log)
        pred_norm = pred_norm / (pred_norm.max() + 1e-30)
        true_norm = ER_norm[test_mask]

        rho, _  = spearmanr(true_norm, pred_norm)
        nrmse   = float(np.sqrt(np.mean((true_norm - pred_norm)**2)))
        mask    = true_norm > 1e-3
        rel_err = np.abs(true_norm[mask] - pred_norm[mask]) / (true_norm[mask] + 1e-30)
        mean_rel = float(rel_err.mean()) if mask.any() else float("nan")
        t_m = np.percentile(ER_man[test_mask], 90)
        pred_abs = np.exp(pred_log) * ER_max
        t_p = np.percentile(pred_abs, 90)
        iou = float(np.sum((ER_man[test_mask] >= t_m) & (pred_abs >= t_p))
                    / (np.sum((ER_man[test_mask] >= t_m) | (pred_abs >= t_p)) + 1e-10))

        print(f"  Fold {fold} (R{fold} held out, n_test={test_mask.sum():,}): "
              f"ρ={rho:.4f}  nRMSE={nrmse:.4f}  rel={mean_rel*100:.1f}%  IoU={iou:.4f}  "
              f"({elapsed:.1f}s)")

        fold_results.append({
            "fold": fold, "n_test": int(test_mask.sum()),
            "spearman_rho": round(float(rho), 4),
            "nRMSE":        round(float(nrmse), 4),
            "mean_rel_err_pct": round(mean_rel*100, 2),
            "top10pct_iou": round(iou, 4),
            "targets_met": {
                "rho_gt_0.85":   bool(rho > 0.85),
                "nRMSE_lt_0.10": bool(nrmse < 0.10),
                "iou_gt_0.40":   bool(iou > 0.40),
            },
        })

        idx_s = rng.choice(test_mask.sum(), min(2000, test_mask.sum()), replace=False)
        axes[fold].scatter(true_norm[idx_s], pred_norm[idx_s], s=2, alpha=0.3, color="steelblue")
        axes[fold].plot([0, 1], [0, 1], "r--", lw=1)
        axes[fold].set_title(f"Fold {fold}  ρ={rho:.3f}\nnRMSE={nrmse:.3f}  IoU={iou:.3f}", fontsize=9)
        axes[fold].set_xlabel("True (norm)"); axes[fold].set_ylabel("Pred (norm)")
        axes[fold].grid(True, alpha=0.3)

    # 汇总
    rhos  = [r["spearman_rho"] for r in fold_results]
    nrmses = [r["nRMSE"] for r in fold_results]
    ious  = [r["top10pct_iou"] for r in fold_results]
    print(f"\n  汇总: ρ={np.mean(rhos):.4f}±{np.std(rhos):.4f}  "
          f"nRMSE={np.mean(nrmses):.4f}±{np.std(nrmses):.4f}  "
          f"IoU={np.mean(ious):.4f}±{np.std(ious):.4f}")
    all_met = all(r["targets_met"]["rho_gt_0.85"] for r in fold_results)
    print(f"  所有折 ρ>0.85: {'✓' if all_met else '✗'}")

    plt.tight_layout()
    plt.savefig(str(OUT_PNG), dpi=150)
    plt.close()
    print(f"  → {OUT_PNG.name}")

    summary = {
        "method": "4-fold spatial hold-out by z-coordinate quartile",
        "features": ["Vp [m/s]", "alpha [deg]"],
        "folds": fold_results,
        "aggregate": {
            "rho_mean":   round(float(np.mean(rhos)), 4),
            "rho_std":    round(float(np.std(rhos)), 4),
            "nRMSE_mean": round(float(np.mean(nrmses)), 4),
            "iou_mean":   round(float(np.mean(ious)), 4),
            "all_folds_rho_gt_0.85": bool(all_met),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {OUT_JSON.name}")
    print("\nStep-26 完成。\n")
    return summary


if __name__ == "__main__":
    run()
