"""
step23_gpr_mansouri_retrain.py
================================
以 Mansouri (2015) 真值为目标重新训练 GPR，
替代 step11 中以 Oka 融合值为目标的版本。

策略:
  - 标签: log(ER_mansouri_norm)  [归一化后取对数，保持非负约束]
  - 特征: xyz (标准化)
  - 分层采样: 按 Mansouri ER 分位数分层（替代 risk_level）
  - 预测: E_final_m = exp(GPR_pred) * ER_mansouri.max()  [还原量纲]

输出:
  output/vertices_full_gpr_m.npz   — 以 Mansouri 为基准的 GPR 预测场
  output/gpr_mansouri_model.pkl
  output/gpr_mansouri_summary.json
  output/gpr_mansouri_validation.png
"""

from __future__ import annotations
import json, time
from pathlib import Path

import joblib
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

OUT_NPZ   = OUTPUT_DIR / "vertices_full_gpr_m.npz"
OUT_MODEL = OUTPUT_DIR / "gpr_mansouri_model.pkl"
OUT_JSON  = OUTPUT_DIR / "gpr_mansouri_summary.json"
OUT_PNG   = OUTPUT_DIR / "gpr_mansouri_validation.png"

N_INDUCING  = 4000
RANDOM_SEED = 42
N_RESTARTS  = 3


def stratified_sample_by_quantile(values: np.ndarray, n_total: int,
                                   rng: np.random.Generator) -> np.ndarray:
    """按四分位数分层采样，高值区权重 4×。"""
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    levels = np.zeros(len(values), dtype=int)
    levels[values >= q25] = 1
    levels[values >= q50] = 2
    levels[values >= q75] = 3

    weights = np.array([1.0, 2.0, 3.0, 4.0])
    idx_by_lv = [np.where(levels == lv)[0] for lv in range(4)]
    counts_raw = np.array([len(i) for i in idx_by_lv], dtype=float)
    target = weights * counts_raw
    target = (target / target.sum() * n_total).astype(int)
    target[-1] += n_total - target.sum()

    selected = []
    for idx, cnt in zip(idx_by_lv, target):
        if len(idx) == 0: continue
        selected.append(rng.choice(idx, size=min(cnt, len(idx)), replace=False))
    return np.concatenate(selected)


def run() -> dict:
    print("=" * 60)
    print("AEGIS Step-23  GPR 以 Mansouri 真值重训练")
    print("=" * 60)
    rng = np.random.default_rng(RANDOM_SEED)

    # 加载数据
    man = np.load(str(OUTPUT_DIR / "vertices_full_mansouri.npz"))
    gpr = np.load(str(OUTPUT_DIR / "vertices_full_gpr.npz"))

    xyz      = gpr["xyz"].astype(np.float64)          # (N,3) mm
    ER_man   = man["ER_mansouri"].astype(np.float64)   # Mansouri 真值
    N = xyz.shape[0]
    print(f"  顶点数: {N:,}")

    # 标签: log(ER_norm)，ER_norm ∈ (0,1]
    ER_max  = ER_man.max()
    ER_norm = ER_man / (ER_max + 1e-30)
    eps     = 1e-8
    log_target = np.log(np.clip(ER_norm, eps, 1.0))   # ∈ [-18, 0]
    print(f"  log_target 范围: [{log_target.min():.2f}, {log_target.max():.2f}]")

    # 特征标准化
    scaler     = StandardScaler()
    xyz_scaled = scaler.fit_transform(xyz)

    # 分层采样
    train_idx = stratified_sample_by_quantile(ER_man, N_INDUCING, rng)
    X_train   = xyz_scaled[train_idx]
    y_train   = log_target[train_idx]
    print(f"  训练点: {len(train_idx):,}")

    # 构建核
    kernel = (ConstantKernel(1.0, (1e-3, 10.0))
              * RBF(0.3, (0.05, 5.0))
              + WhiteKernel(0.01, (1e-5, 0.5)))

    gpr_model = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=N_RESTARTS,
        normalize_y=True, random_state=RANDOM_SEED, alpha=1e-8,
    )
    t0 = time.time()
    gpr_model.fit(X_train, y_train)
    print(f"  训练耗时: {time.time()-t0:.1f}s")
    print(f"  核参数: {gpr_model.kernel_}")

    joblib.dump({"gpr": gpr_model, "scaler": scaler, "ER_max": ER_max}, str(OUT_MODEL))

    # 全量预测
    BATCH = 10_000
    log_pred  = np.zeros(N, dtype=np.float64)
    sigma_pred = np.zeros(N, dtype=np.float64)
    t0 = time.time()
    for s in range(0, N, BATCH):
        e = min(s + BATCH, N)
        mu, sig = gpr_model.predict(xyz_scaled[s:e], return_std=True)
        log_pred[s:e]  = mu
        sigma_pred[s:e] = sig
        if (s // BATCH) % 20 == 0:
            print(f"  预测 {100*e/N:.0f}%", flush=True)
    print(f"  预测耗时: {time.time()-t0:.1f}s")

    # 还原侵蚀率
    E_final_m = np.exp(log_pred) * ER_max   # [kg/kg]
    E_final_m = np.clip(E_final_m, 0.0, None)

    # 验证指标
    man_norm  = ER_norm
    pred_norm = E_final_m / (E_final_m.max() + 1e-30)

    rho, pval = spearmanr(man_norm, pred_norm)
    r_p, _    = pearsonr(man_norm, pred_norm)
    nrmse     = float(np.sqrt(np.mean((man_norm - pred_norm)**2)))

    mask    = man_norm > 1e-3
    rel_err = np.abs(man_norm[mask] - pred_norm[mask]) / (man_norm[mask] + 1e-30)
    mean_rel = float(rel_err.mean())

    t_m = np.percentile(ER_man, 90)
    t_p = np.percentile(E_final_m, 90)
    iou = float(np.sum((ER_man >= t_m) & (E_final_m >= t_p))
                / (np.sum((ER_man >= t_m) | (E_final_m >= t_p)) + 1e-10))

    print(f"\n  Spearman ρ   = {rho:.4f}  (p={pval:.2e})")
    print(f"  Pearson  r   = {r_p:.4f}")
    print(f"  nRMSE        = {nrmse:.4f}")
    print(f"  Mean rel err = {mean_rel*100:.1f}%")
    print(f"  Top-10% IoU  = {iou:.4f}")
    print(f"\n  目标 ρ>0.85:   {'✓' if rho>0.85  else '✗'}  ({rho:.3f})")
    print(f"  目标 IoU>0.40: {'✓' if iou>0.40  else '✗'}  ({iou:.3f})")
    print(f"  目标 nRMSE<0.1:{'✓' if nrmse<0.10 else '✗'}  ({nrmse:.4f})")

    # 保存 NPZ
    np.savez_compressed(
        str(OUT_NPZ),
        xyz=xyz.astype(np.float32),
        E_final_mansouri=E_final_m.astype(np.float32),
        E_final_norm=pred_norm.astype(np.float32),
        ER_mansouri_truth=ER_man.astype(np.float32),
        sigma=sigma_pred.astype(np.float32),
    )
    print(f"\n  → {OUT_NPZ.name}")

    # 散点图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GPR (Mansouri-trained) vs Mansouri Ground Truth", fontsize=12)
    idx = np.random.choice(N, min(5000, N), replace=False)
    axes[0].scatter(man_norm[idx], pred_norm[idx], s=2, alpha=0.3, color="steelblue")
    lim = 1.05
    axes[0].plot([0, lim], [0, lim], "r--", lw=1, label="y=x")
    axes[0].set_xlabel("Mansouri ER (normalized)")
    axes[0].set_ylabel("GPR prediction (normalized)")
    axes[0].set_title(f"ρ={rho:.3f}  nRMSE={nrmse:.3f}")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].hist(rel_err * 100, bins=60, color="coral", edgecolor="white", lw=0.3)
    axes[1].axvline(mean_rel*100, color="red", ls="--", lw=1, label=f"mean={mean_rel*100:.1f}%")
    axes[1].axvline(5, color="green", ls=":", lw=1, label="5% target")
    axes[1].set_xlabel("Relative error [%]")
    axes[1].set_ylabel("Vertex count")
    axes[1].set_title("Point-wise relative error (man_norm > 1e-3)")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(OUT_PNG), dpi=150)
    plt.close()
    print(f"  → {OUT_PNG.name}")

    summary = {
        "model_truth":     "Mansouri (2015)",
        "model_predicted": "GPR retrained on Mansouri targets (step23)",
        "n_vertices": int(N), "n_inducing": int(len(train_idx)),
        "metrics": {
            "spearman_rho":    round(float(rho), 4),
            "pearson_r":       round(float(r_p), 4),
            "nRMSE":           round(float(nrmse), 4),
            "mean_rel_err_pct":round(mean_rel*100, 2),
            "top10pct_iou":    round(iou, 4),
        },
        "targets_met": {
            "spearman_rho_gt_0.85": bool(rho > 0.85),
            "iou_gt_0.40":          bool(iou > 0.40),
            "nRMSE_lt_0.10":        bool(nrmse < 0.10),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {OUT_JSON.name}")
    print("\nStep-23 完成。\n")
    return summary


if __name__ == "__main__":
    run()
