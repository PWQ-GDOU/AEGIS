"""
step25_finalize_gpr_pipeline.py
================================
将 step24 验证的物理特征 GPR (Vp, alpha) 集成为最终预测管道，
更新 vertices_full_gpr.npz 中的 E_final 字段为 Mansouri-GPR 预测值。

验证结果 (step24):
  Spearman ρ = 0.9994, nRMSE = 0.018, IoU = 0.970, 平均相对误差 4.0%

输出:
  output/vertices_full_gpr.npz  — 更新 E_final / E_final_norm / correction_factor
  output/pipeline_summary.json
"""

from __future__ import annotations
import json, time
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

WORKSPACE  = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "output"
OUT_JSON   = OUTPUT_DIR / "pipeline_summary.json"

N_INDUCING  = 2000
RANDOM_SEED = 42
N_RESTARTS  = 2


def run() -> dict:
    print("=" * 60)
    print("AEGIS Step-25  最终 GPR 管道集成")
    print("=" * 60)
    rng = np.random.default_rng(RANDOM_SEED)

    # 加载数据
    man      = np.load(str(OUTPUT_DIR / "vertices_full_mansouri.npz"))
    gpr_old  = np.load(str(OUTPUT_DIR / "vertices_full_gpr.npz"))

    ER_man = man["ER_mansouri"].astype(np.float64)
    Vp     = gpr_old["Vp"].astype(np.float64)
    alpha  = gpr_old["alpha"].astype(np.float64)
    xyz    = gpr_old["xyz"].astype(np.float64)
    E_oka  = gpr_old["E_oka"].astype(np.float64)
    N = len(ER_man)

    ER_max  = ER_man.max()
    ER_norm = ER_man / (ER_max + 1e-30)
    eps     = 1e-8
    y_log   = np.log(np.clip(ER_norm, eps, 1.0))

    # 特征: (Vp, alpha)
    X_all = np.column_stack([Vp, alpha])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # 分层采样（按 ER_man 四分位）
    q25, q50, q75 = np.percentile(ER_man, [25, 50, 75])
    levels = np.zeros(N, dtype=int)
    levels[ER_man >= q25] = 1
    levels[ER_man >= q50] = 2
    levels[ER_man >= q75] = 3
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    idx_by_lv = [np.where(levels == lv)[0] for lv in range(4)]
    counts_raw = np.array([len(i) for i in idx_by_lv], dtype=float)
    target = (weights * counts_raw / (weights * counts_raw).sum() * N_INDUCING).astype(int)
    target[-1] += N_INDUCING - target.sum()
    train_idx = np.concatenate([
        rng.choice(idx, size=min(cnt, len(idx)), replace=False)
        for idx, cnt in zip(idx_by_lv, target) if len(idx) > 0
    ])
    print(f"  训练点: {len(train_idx):,}")

    # 训练 GPR
    kernel = (ConstantKernel(1.0, (1e-3, 10.0))
              * RBF(1.0, (0.1, 10.0))
              + WhiteKernel(0.01, (1e-6, 1.0)))
    gpr_model = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=N_RESTARTS,
        normalize_y=True, random_state=RANDOM_SEED, alpha=1e-8,
    )
    t0 = time.time()
    gpr_model.fit(X_scaled[train_idx], y_log[train_idx])
    print(f"  训练耗时: {time.time()-t0:.1f}s  核: {gpr_model.kernel_}")

    # 保存模型
    joblib.dump({"gpr": gpr_model, "scaler": scaler, "ER_max": ER_max,
                 "features": "Vp_alpha"}, str(OUTPUT_DIR / "gpr_final_model.pkl"))

    # 全量预测
    BATCH = 20_000
    log_pred   = np.zeros(N, dtype=np.float64)
    sigma_pred = np.zeros(N, dtype=np.float64)
    t0 = time.time()
    for s in range(0, N, BATCH):
        e = min(s + BATCH, N)
        mu, sig = gpr_model.predict(X_scaled[s:e], return_std=True)
        log_pred[s:e]   = mu
        sigma_pred[s:e] = sig
    print(f"  预测耗时: {time.time()-t0:.1f}s")

    # 还原侵蚀率
    E_final = np.exp(log_pred) * ER_max
    E_final = np.clip(E_final, 0.0, None)
    E_final_norm = E_final / (E_final.max() + 1e-30)

    # 修正因子（相对 Oka）
    correction_factor = E_final / (E_oka + 1e-30)

    # 95% CI
    E_upper_95 = np.exp(log_pred + 2.0 * sigma_pred) * ER_max
    E_lower_95 = np.clip(np.exp(log_pred - 2.0 * sigma_pred) * ER_max, 0.0, None)
    sigma_E    = E_final * sigma_pred

    # 验证指标
    man_norm = ER_norm
    rho, _   = spearmanr(man_norm, E_final_norm)
    nrmse    = float(np.sqrt(np.mean((man_norm - E_final_norm)**2)))
    mask     = man_norm > 1e-3
    rel_err  = np.abs(man_norm[mask] - E_final_norm[mask]) / (man_norm[mask] + 1e-30)
    mean_rel = float(rel_err.mean())
    t_m = np.percentile(ER_man, 90)
    t_p = np.percentile(E_final, 90)
    iou = float(np.sum((ER_man >= t_m) & (E_final >= t_p))
                / (np.sum((ER_man >= t_m) | (E_final >= t_p)) + 1e-10))

    print(f"\n  Spearman ρ   = {rho:.4f}  {'✓' if rho>0.85 else '✗'}")
    print(f"  nRMSE        = {nrmse:.4f}  {'✓' if nrmse<0.10 else '✗'}")
    print(f"  Mean rel err = {mean_rel*100:.1f}%  {'✓' if mean_rel<0.05 else '✗'}")
    print(f"  Top-10% IoU  = {iou:.4f}  {'✓' if iou>0.40 else '✗'}")

    # 更新 vertices_full_gpr.npz（保留所有原有字段，覆盖 E_final 相关字段）
    old_keys = dict(gpr_old)
    old_keys.update({
        "E_final":          E_final.astype(np.float32),
        "E_final_norm":     E_final_norm.astype(np.float32),
        "correction_factor":correction_factor.astype(np.float32),
        "sigma":            sigma_E.astype(np.float32),
        "E_upper_95":       E_upper_95.astype(np.float32),
        "E_lower_95":       E_lower_95.astype(np.float32),
        "ER_mansouri_truth":ER_man.astype(np.float32),
    })
    np.savez_compressed(str(OUTPUT_DIR / "vertices_full_gpr.npz"), **old_keys)
    print(f"\n  → vertices_full_gpr.npz (已更新 E_final → Mansouri-GPR)")

    summary = {
        "pipeline": "GPR(Vp, alpha) → Mansouri-calibrated E_final",
        "model_truth": "Mansouri (2015) Wear 332-333:1090-1097",
        "n_vertices": int(N), "n_inducing": int(len(train_idx)),
        "features": ["Vp [m/s]", "alpha [deg]"],
        "metrics": {
            "spearman_rho":    round(float(rho), 4),
            "nRMSE":           round(float(nrmse), 4),
            "mean_rel_err_pct":round(mean_rel * 100, 2),
            "top10pct_iou":    round(iou, 4),
        },
        "targets_met": {
            "rho_gt_0.85":   bool(rho > 0.85),
            "nRMSE_lt_0.10": bool(nrmse < 0.10),
            "rel_err_lt_5pct":bool(mean_rel < 0.05),
            "iou_gt_0.40":   bool(iou > 0.40),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {OUT_JSON.name}")
    print("\nStep-25 完成。\n")
    return summary


if __name__ == "__main__":
    run()
