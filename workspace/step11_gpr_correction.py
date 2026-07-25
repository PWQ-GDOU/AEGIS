"""
step11_gpr_correction.py
========================
灰盒 Layer-2：GPR 残差修正模块

设计原则
--------
无实测磨损数据时，采用"物理先验均值 GPR"：
  - 白盒基线：E_phys(x) = Oka 模型输出（来自 step10）
  - 残差标签：f(x) = log( E_ensemble(x) / E_oka(x) )
      E_ensemble = 三模型百分位秩加权融合（量化模型间分歧）
  - GPR 学习平滑的空间修正场 f(x)
  - 最终侵蚀率：E_final(x) = E_oka(x) · exp(f_GPR(x))  [非负约束]
  - 不确定度：σ(x) 来自 GPR 后验标准差

稀疏近似
--------
真机转轮顶点数 ~154 K，全矩阵 GP 不可行。
采用分层随机子采样（按风险等级）取 N_INDUCING 个训练点，
用 sklearn GaussianProcessRegressor 训练，再对全体顶点预测。

输出
----
output/vertices_full_gpr.npz   — full_runner 顶点 + GPR 修正字段
output/gpr_model.pkl           — 训练好的 GPR（joblib）
output/gpr_summary.json        — 训练摘要与验证指标
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
WORKSPACE  = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "output"

NPZ_FULL   = OUTPUT_DIR / "vertices_full.npz"
OUT_NPZ    = OUTPUT_DIR / "vertices_full_gpr.npz"
OUT_MODEL  = OUTPUT_DIR / "gpr_model.pkl"
OUT_JSON   = OUTPUT_DIR / "gpr_summary.json"

# ---------------------------------------------------------------------------
# 超参数
# ---------------------------------------------------------------------------
N_INDUCING: int   = 3000    # 训练点数（稀疏近似）；可通过命令行覆盖
RANDOM_SEED: int  = 42
N_RESTARTS: int   = 3       # GPR 超参数优化重启次数

# 三模型融合权重（Oka 最物理可靠，权重最高）
W_OKA:      float = 0.60
W_FINNIE:   float = 0.25
W_TABAKOFF: float = 0.15

# 修正因子的截断范围（log 空间），防止外推飞掉
LOG_CORR_MIN: float = -1.5   # exp(-1.5) ≈ 0.22 —— 最多将 Oka 下调 78%
LOG_CORR_MAX: float =  1.5   # exp( 1.5) ≈ 4.48 —— 最多将 Oka 上调 348%


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def percentile_rank(arr: np.ndarray) -> np.ndarray:
    """将数组转换为百分位秩 ∈ [0, 1]（处理平局用平均秩）。"""
    from scipy.stats import rankdata
    return rankdata(arr, method="average") / len(arr)


def build_ensemble(
    E_oka:      np.ndarray,
    E_finnie:   np.ndarray,
    E_tabakoff: np.ndarray,
) -> np.ndarray:
    """
    三模型百分位秩加权融合，返回归一化融合侵蚀率 ∈ [0, 1]。

    百分位秩融合优点：
      1. 消除三模型量纲差异（无需手动归一化系数）
      2. 保留空间排序信息
      3. 融合结果可直接与 Oka 归一化结果比较
    """
    r_oka      = percentile_rank(E_oka)
    r_finnie   = percentile_rank(E_finnie)
    r_tabakoff = percentile_rank(E_tabakoff)

    ensemble = W_OKA * r_oka + W_FINNIE * r_finnie + W_TABAKOFF * r_tabakoff
    return ensemble  # ∈ [0, 1]


def stratified_sample(
    risk_level: np.ndarray,
    n_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    按风险等级分层随机采样，确保高风险区样本充足。

    分层比例：极高:高:中:低 = 4:3:2:1
    """
    weights = np.array([1.0, 2.0, 3.0, 4.0])  # 等级 0/1/2/3 的相对权重

    indices_by_level = [np.where(risk_level == lv)[0] for lv in range(4)]
    counts_raw = np.array([len(idx) for idx in indices_by_level], dtype=float)

    # 目标采样数 = 分层权重 × 可用数量（取两者较小）
    target = weights * counts_raw
    target = target / target.sum() * n_total
    target = np.clip(target.astype(int), 1, None)
    # 修正总数
    diff = n_total - target.sum()
    target[-1] += diff   # 微调到极高风险区

    selected = []
    for lv, (idx, cnt) in enumerate(zip(indices_by_level, target)):
        if len(idx) == 0:
            continue
        cnt = min(cnt, len(idx))
        chosen = rng.choice(idx, size=cnt, replace=False)
        selected.append(chosen)

    return np.concatenate(selected)


# ---------------------------------------------------------------------------
# GPR 核构建
# ---------------------------------------------------------------------------

def build_kernel(xyz_scaled: np.ndarray) -> "Kernel":
    """
    构建 GPR 核。

    核 = C(振幅) × RBF(长度尺度) + WhiteNoise(噪声)

    初始长度尺度设为标准化坐标空间中的中等距离（0.3），
    边界宽松以允许超参数优化找到最优值。
    """
    amplitude = ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-3, 10.0)
    )
    rbf = RBF(
        length_scale=0.3,
        length_scale_bounds=(0.05, 5.0)
    )
    noise = WhiteKernel(
        noise_level=0.01,
        noise_level_bounds=(1e-5, 0.5)
    )
    return amplitude * rbf + noise


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def run_gpr(n_inducing: int = N_INDUCING) -> dict:
    """
    完整 GPR 修正流程，返回结果字典。

    参数
    ----
    n_inducing : 训练点数（稀疏近似）
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # -----------------------------------------------------------------------
    # 1. 加载 step10 输出
    # -----------------------------------------------------------------------
    print("[GPR] 加载 vertices_full.npz ...", flush=True)
    data = np.load(str(NPZ_FULL))

    xyz        = data["xyz"]           # (N, 3) mm
    E_oka      = data["E_oka"]         # (N,) mm³/kg
    E_finnie   = data["E_finnie"]      # (N,)
    E_tabakoff = data["E_tabakoff"]    # (N,)
    E_oka_norm = data["E_oka_norm"]    # (N,) [0,1]
    risk_level = data["risk_level"]    # (N,) int8

    N = xyz.shape[0]
    print(f"[GPR] 顶点总数: {N:,}", flush=True)

    # -----------------------------------------------------------------------
    # 2. 构建融合标签 & 对数修正目标
    # -----------------------------------------------------------------------
    print("[GPR] 计算三模型融合标签 ...", flush=True)
    ensemble_rank = build_ensemble(E_oka, E_finnie, E_tabakoff)   # [0,1]
    oka_rank      = percentile_rank(E_oka)                         # [0,1]

    # 对数修正目标（在百分位秩空间）
    # f(x) = log(ensemble_rank / oka_rank)，加小量防 log(0)
    eps = 1e-6
    log_target = np.log(
        np.clip(ensemble_rank, eps, None) / np.clip(oka_rank, eps, None)
    )
    log_target = np.clip(log_target, LOG_CORR_MIN, LOG_CORR_MAX)

    print(f"[GPR] 对数修正目标范围: [{log_target.min():.3f}, {log_target.max():.3f}]  "
          f"均值: {log_target.mean():.3f}  标准差: {log_target.std():.3f}", flush=True)

    # -----------------------------------------------------------------------
    # 3. 特征标准化（GPR 核对尺度敏感）
    # -----------------------------------------------------------------------
    scaler = StandardScaler()
    xyz_scaled = scaler.fit_transform(xyz)   # (N, 3)，均值0方差1

    # -----------------------------------------------------------------------
    # 4. 分层采样训练集
    # -----------------------------------------------------------------------
    n_inducing = min(n_inducing, N)
    print(f"[GPR] 分层采样 {n_inducing} 个训练点（风险加权）...", flush=True)
    train_idx = stratified_sample(risk_level, n_inducing, rng)

    X_train = xyz_scaled[train_idx]      # (n_inducing, 3)
    y_train = log_target[train_idx]      # (n_inducing,)

    print(f"[GPR] 训练集风险分布: "
          + "  ".join(
              f"L{lv}:{np.sum(risk_level[train_idx]==lv)}"
              for lv in range(4)
          ), flush=True)

    # -----------------------------------------------------------------------
    # 5. 训练 GPR
    # -----------------------------------------------------------------------
    print(f"[GPR] 训练 GaussianProcessRegressor (n_restarts={N_RESTARTS}) ...",
          flush=True)
    kernel = build_kernel(xyz_scaled)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=N_RESTARTS,
        normalize_y=True,       # 内部归一化 y，提高数值稳定性
        random_state=RANDOM_SEED,
        alpha=1e-8,             # 数值稳定项（不额外加噪声）
    )

    t0 = time.time()
    gpr.fit(X_train, y_train)
    t_train = time.time() - t0
    print(f"[GPR] 训练完成，耗时 {t_train:.1f} s", flush=True)
    print(f"[GPR] 最优核参数: {gpr.kernel_}", flush=True)
    print(f"[GPR] 对数边际似然: {gpr.log_marginal_likelihood_value_:.3f}", flush=True)

    # 保存模型
    joblib.dump({"gpr": gpr, "scaler": scaler}, str(OUT_MODEL))
    print(f"[GPR] 模型已保存: {OUT_MODEL.name}", flush=True)

    # -----------------------------------------------------------------------
    # 6. 全量预测（分批避免内存溢出）
    # -----------------------------------------------------------------------
    BATCH = 10_000
    log_corr_pred = np.zeros(N, dtype=np.float64)
    sigma_pred    = np.zeros(N, dtype=np.float64)

    print(f"[GPR] 全量预测 {N:,} 顶点（批大小={BATCH}）...", flush=True)
    t0 = time.time()
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        mu, sigma = gpr.predict(xyz_scaled[start:end], return_std=True)
        log_corr_pred[start:end] = mu
        sigma_pred[start:end]    = sigma
        if (start // BATCH) % 20 == 0:
            pct = 100.0 * end / N
            print(f"  {pct:.0f}%  ({end:,}/{N:,})", flush=True)

    t_pred = time.time() - t0
    print(f"[GPR] 预测完成，耗时 {t_pred:.1f} s", flush=True)

    # 截断修正因子
    log_corr_pred = np.clip(log_corr_pred, LOG_CORR_MIN, LOG_CORR_MAX)

    # -----------------------------------------------------------------------
    # 7. 计算最终侵蚀率（在原始 mm³/kg 空间）
    # -----------------------------------------------------------------------
    correction_factor = np.exp(log_corr_pred)          # 乘性修正，始终 > 0
    E_final           = E_oka * correction_factor       # (N,) mm³/kg

    # GPR 不确定度传播到 E 空间（Delta 法：σ_E ≈ E_final × σ_logcorr）
    sigma_E = E_final * sigma_pred

    # 95% 置信区间（log 空间对称 → E 空间不对称，取保守近似）
    E_upper_95 = E_oka * np.exp(log_corr_pred + 2.0 * sigma_pred)
    E_lower_95 = E_oka * np.exp(log_corr_pred - 2.0 * sigma_pred)
    E_lower_95 = np.clip(E_lower_95, 0.0, None)

    # 归一化最终侵蚀率
    E_final_norm = E_final / (E_final.max() + 1e-30)

    # -----------------------------------------------------------------------
    # 8. 交叉验证（保留集 = 未入训练集的点）
    # -----------------------------------------------------------------------
    print("[GPR] 计算保留集验证指标 ...", flush=True)
    all_idx  = np.arange(N)
    train_set = set(train_idx.tolist())
    holdout_idx = np.array([i for i in all_idx if i not in train_set])

    # 在保留集上比较 GPR 预测与真实对数修正目标
    ho_pred  = log_corr_pred[holdout_idx]
    ho_true  = log_target[holdout_idx]
    mae_holdout  = float(np.mean(np.abs(ho_pred - ho_true)))
    rmse_holdout = float(np.sqrt(np.mean((ho_pred - ho_true) ** 2)))
    rho_holdout, _ = spearmanr(ho_pred, ho_true)

    # 最终侵蚀率 vs Oka 的空间 Spearman 相关（衡量修正是否保持排序）
    rho_final_oka, _ = spearmanr(E_final, E_oka)

    print(f"[GPR] 保留集 MAE={mae_holdout:.4f}  RMSE={rmse_holdout:.4f}  "
          f"Spearman ρ={rho_holdout:.4f}", flush=True)
    print(f"[GPR] E_final vs E_oka Spearman ρ={rho_final_oka:.4f}", flush=True)

    # -----------------------------------------------------------------------
    # 9. 保存 NPZ（全体顶点 + GPR 新字段）
    # -----------------------------------------------------------------------
    print("[GPR] 保存 vertices_full_gpr.npz ...", flush=True)
    np.savez_compressed(
        str(OUT_NPZ),
        # 继承 step10 字段
        xyz          = xyz,
        normals      = data["normals"],
        Vp           = data["Vp"],
        alpha        = data["alpha"],
        E_oka        = E_oka,
        E_finnie     = E_finnie,
        E_tabakoff   = E_tabakoff,
        E_oka_norm   = E_oka_norm,
        erosion_type = data["erosion_type"],
        risk_level   = risk_level,
        # GPR 新字段
        ensemble_rank    = ensemble_rank.astype(np.float32),
        log_corr_target  = log_target.astype(np.float32),
        log_corr_pred    = log_corr_pred.astype(np.float32),
        correction_factor= correction_factor.astype(np.float32),
        E_final          = E_final.astype(np.float32),
        E_final_norm     = E_final_norm.astype(np.float32),
        sigma_E          = sigma_E.astype(np.float32),
        E_upper_95       = E_upper_95.astype(np.float32),
        E_lower_95       = E_lower_95.astype(np.float32),
    )
    print(f"  → {OUT_NPZ.name}  ({OUT_NPZ.stat().st_size / 1024:.0f} KB)", flush=True)

    # -----------------------------------------------------------------------
    # 10. 摘要 JSON
    # -----------------------------------------------------------------------
    summary = {
        "n_vertices_total":    int(N),
        "n_inducing_points":   int(len(train_idx)),
        "kernel_optimized":    str(gpr.kernel_),
        "log_marginal_likelihood": float(gpr.log_marginal_likelihood_value_),
        "training_time_s":     round(t_train, 2),
        "prediction_time_s":   round(t_pred, 2),
        "log_correction": {
            "min":  float(log_corr_pred.min()),
            "max":  float(log_corr_pred.max()),
            "mean": float(log_corr_pred.mean()),
            "std":  float(log_corr_pred.std()),
        },
        "correction_factor": {
            "min":  float(correction_factor.min()),
            "max":  float(correction_factor.max()),
            "mean": float(correction_factor.mean()),
        },
        "E_final": {
            "min_mm3kg":  float(E_final.min()),
            "max_mm3kg":  float(E_final.max()),
            "mean_mm3kg": float(E_final.mean()),
        },
        "uncertainty_sigma_E": {
            "mean": float(sigma_E.mean()),
            "p95":  float(np.percentile(sigma_E, 95)),
        },
        "validation": {
            "holdout_n":       int(len(holdout_idx)),
            "holdout_mae":     mae_holdout,
            "holdout_rmse":    rmse_holdout,
            "holdout_spearman_rho": float(rho_holdout),
            "E_final_vs_Oka_spearman_rho": float(rho_final_oka),
        },
        "label_strategy": "log_ensemble_rank_vs_oka_rank (P1: no measured data)",
        "non_negative_guarantee": "E_final = E_oka * exp(f_GPR) > 0 always",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"  → {OUT_JSON.name}", flush=True)
    print("Step-11 GPR 修正完成。\n", flush=True)

    return summary


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    n_ind = int(sys.argv[1]) if len(sys.argv) > 1 else N_INDUCING
    run_gpr(n_inducing=n_ind)
