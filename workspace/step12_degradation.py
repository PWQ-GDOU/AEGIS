"""
step12_degradation.py
=====================
灰盒 Layer-3：时间退化积分与剩余寿命预测

核心物理
--------
Oka 侵蚀率 E_final 单位为 [mm³/kg]：每 kg 泥沙撞击移除的材料体积。
将其转换为壁厚减薄速率需乘以"单位面积泥沙质量通量"：

    ṁ_flux = C_s · V_p · η       [kg/(m²·s)]
        C_s : 含沙量 [kg/m³]
        V_p : 局部冲击速度 [m/s]
        η   : 撞击效率（近壁泥沙实际参与撞击的比例，保守取 1.0）

    体积损失速率（单位面积）= E_final · ṁ_flux   [mm³/(m²·s)]
    壁厚减薄速率 dh/dt = 体积损失 / 面积
                       = E_final · ṁ_flux · 1e-6   [mm/s]
       （1e-6：mm³/m² → mm，因 1 m² = 1e6 mm²）

线性退化模型（第一版）
--------------------
    h(x, t) = h0 - (dh/dt)(x) · t
    T_RUL(x) = (h0 - h_min) / (dh/dt)(x)

置信区间由 GPR σ 传播：用 E_upper_95 得到保守（最短）寿命。

参数（待 DEC 图纸确认，当前使用工程缺省值）
-------------------------------------------------
    h0    = 30 mm   初始壁厚
    h_min = 15 mm   最小允许壁厚（50% 损耗判废）

输出
----
output/rul_map.npz          — 逐点壁厚速率、剩余寿命及置信区间
output/degradation_curves.npz — 若干关键点的壁厚-时间曲线
output/critical_points.json — 最先失效 Top-N 点
output/degradation_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
WORKSPACE  = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "output"

NPZ_GPR  = OUTPUT_DIR / "vertices_full_gpr.npz"
OUT_RUL  = OUTPUT_DIR / "rul_map.npz"
OUT_CURVE = OUTPUT_DIR / "degradation_curves.npz"
OUT_CRIT = OUTPUT_DIR / "critical_points.json"
OUT_JSON = OUTPUT_DIR / "degradation_summary.json"

# ---------------------------------------------------------------------------
# 退化参数（可外部覆盖，供敏感性分析）
# ---------------------------------------------------------------------------
H0_DEFAULT:    float = 30.0    # 初始壁厚 [mm]      —— 待 DEC 图纸确认
HMIN_DEFAULT:  float = 15.0    # 最小允许壁厚 [mm]  —— 待 DEC 图纸确认
CS_DEFAULT:    float = 1.31    # 含沙量 [kg/m³]
ETA_IMPACT:    float = 1.0     # 撞击效率（保守=1.0）

# 年运行小时（丰水期含沙，按典型水电站汛期 ~4 个月高含沙运行）
HOURS_PER_YEAR: float = 4.0 * 30 * 24   # ≈ 2880 h/年（高含沙时段）

# 时间积分设置
T_STEP_HOURS: float = 1000.0            # 每步 1000 h
N_STEPS:      int   = 30                # 最长积分 30000 h ≈ 10 年高含沙运行


# ---------------------------------------------------------------------------
# 核心计算
# ---------------------------------------------------------------------------

RHO_WALL: float = 7800.0   # 不锈钢密度 [kg/m³]（Mansouri SS316）


def thickness_loss_rate(
    E_final:  np.ndarray,
    Vp:       np.ndarray,
    C_s:      float = CS_DEFAULT,
    eta:      float = ETA_IMPACT,
    rho_wall: float = RHO_WALL,
) -> np.ndarray:
    """
    壁厚减薄速率 [mm/h]。

    Mansouri ER 单位 [kg/kg]（质量比）：
      dh/dt [m/s] = ER[kg/kg] · C_s[kg/m³] · Vp[m/s] · η / ρ_wall[kg/m³]
      → × 1e3 转 [mm/s] → × 3600 转 [mm/h]
    """
    dhdt_per_s = E_final * C_s * Vp * eta / rho_wall * 1e3   # mm/s
    return dhdt_per_s * 3600.0                                 # mm/h


def compute_rul(
    dhdt: np.ndarray,
    h0:   float = H0_DEFAULT,
    hmin: float = HMIN_DEFAULT,
) -> np.ndarray:
    """
    剩余寿命 [h]。dh/dt→0 的点寿命设为 inf。
    """
    usable = h0 - hmin
    rul = np.full_like(dhdt, np.inf, dtype=np.float64)
    mask = dhdt > 1e-12
    rul[mask] = usable / dhdt[mask]
    return rul


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_degradation(
    h0:   float = H0_DEFAULT,
    hmin: float = HMIN_DEFAULT,
    C_s:  float = CS_DEFAULT,
) -> dict:
    print("=" * 60)
    print("AEGIS Step-12 时间退化与剩余寿命")
    print(f"  h0={h0} mm  h_min={hmin} mm  C_s={C_s} kg/m³")
    print("=" * 60)

    data = np.load(str(NPZ_GPR))
    xyz        = data["xyz"]         # (N,3) mm
    Vp         = data["Vp"]          # (N,) m/s
    E_final    = data["E_final"].astype(np.float64)
    E_upper_95 = data["E_upper_95"].astype(np.float64)
    E_lower_95 = data["E_lower_95"].astype(np.float64)
    risk_level = data["risk_level"]
    N = xyz.shape[0]
    print(f"[DEG] 顶点数: {N:,}", flush=True)

    # -----------------------------------------------------------------------
    # 壁厚减薄速率（期望 + 保守 + 乐观）
    # -----------------------------------------------------------------------
    dhdt_mean  = thickness_loss_rate(E_final,    Vp, C_s)   # mm/h
    dhdt_upper = thickness_loss_rate(E_upper_95, Vp, C_s)   # 保守（最快磨损）
    dhdt_lower = thickness_loss_rate(E_lower_95, Vp, C_s)   # 乐观（最慢磨损）

    print(f"[DEG] 壁厚减薄速率 (mean) 范围: "
          f"[{dhdt_mean.min():.3e}, {dhdt_mean.max():.3e}] mm/h", flush=True)

    # -----------------------------------------------------------------------
    # 剩余寿命（期望 + 95% 区间）
    # -----------------------------------------------------------------------
    rul_mean  = compute_rul(dhdt_mean,  h0, hmin)   # h
    rul_lower = compute_rul(dhdt_upper, h0, hmin)   # 保守寿命（下界）
    rul_upper = compute_rul(dhdt_lower, h0, hmin)   # 乐观寿命（上界）

    # 转年（高含沙运行时段）
    rul_mean_yr  = rul_mean  / HOURS_PER_YEAR
    rul_lower_yr = rul_lower / HOURS_PER_YEAR

    # 全局最小寿命（关键失效点）
    finite = np.isfinite(rul_mean)
    min_rul_h  = float(rul_mean[finite].min())
    min_rul_yr = min_rul_h / HOURS_PER_YEAR
    argmin     = int(np.where(finite)[0][np.argmin(rul_mean[finite])])
    print(f"[DEG] 全局最短寿命: {min_rul_h:,.0f} h ≈ {min_rul_yr:.2f} 年"
          f"（高含沙时段），位置顶点 #{argmin}", flush=True)
    print(f"[DEG]   保守寿命下界: {rul_lower[argmin]:,.0f} h ≈ "
          f"{rul_lower[argmin]/HOURS_PER_YEAR:.2f} 年", flush=True)

    # -----------------------------------------------------------------------
    # 时间积分：各时间步壁厚场（仅存储关键统计，避免巨大文件）
    # -----------------------------------------------------------------------
    times_h = np.arange(0, N_STEPS + 1) * T_STEP_HOURS   # (N_STEPS+1,)
    # h(x,t) = h0 - dhdt*t，逐步 clip 到 [0, h0]
    frac_failed = np.zeros(len(times_h))   # 各时刻失效顶点占比
    mean_thickness = np.zeros(len(times_h))
    min_thickness  = np.zeros(len(times_h))

    for i, t in enumerate(times_h):
        h_t = np.clip(h0 - dhdt_mean * t, 0.0, h0)
        frac_failed[i]    = 100.0 * np.mean(h_t <= hmin)
        mean_thickness[i] = float(h_t.mean())
        min_thickness[i]  = float(h_t.min())

    print(f"[DEG] {times_h[-1]:,.0f} h 后失效顶点占比: {frac_failed[-1]:.1f}%", flush=True)

    # -----------------------------------------------------------------------
    # 关键点退化曲线（Top-10 最短寿命点）
    # -----------------------------------------------------------------------
    order = np.argsort(rul_mean)          # inf 排在最后
    top_n = 10
    crit_idx = order[:top_n]

    # 每个关键点的壁厚-时间曲线
    curves = np.zeros((top_n, len(times_h)))
    for j, idx in enumerate(crit_idx):
        curves[j] = np.clip(h0 - dhdt_mean[idx] * times_h, 0.0, h0)

    # -----------------------------------------------------------------------
    # 保存 rul_map.npz
    # -----------------------------------------------------------------------
    np.savez_compressed(
        str(OUT_RUL),
        xyz          = xyz,
        risk_level   = risk_level,
        dhdt_mean    = dhdt_mean.astype(np.float32),
        dhdt_upper   = dhdt_upper.astype(np.float32),
        dhdt_lower   = dhdt_lower.astype(np.float32),
        rul_mean_h   = rul_mean.astype(np.float32),
        rul_lower_h  = rul_lower.astype(np.float32),
        rul_upper_h  = rul_upper.astype(np.float32),
        rul_mean_yr  = rul_mean_yr.astype(np.float32),
        rul_lower_yr = rul_lower_yr.astype(np.float32),
    )
    print(f"[DEG] → {OUT_RUL.name}  ({OUT_RUL.stat().st_size/1024:.0f} KB)", flush=True)

    # 保存退化曲线
    np.savez_compressed(
        str(OUT_CURVE),
        times_h        = times_h,
        curves         = curves,             # (top_n, T) 关键点壁厚
        crit_idx       = crit_idx,
        frac_failed    = frac_failed,
        mean_thickness = mean_thickness,
        min_thickness  = min_thickness,
    )
    print(f"[DEG] → {OUT_CURVE.name}", flush=True)

    # -----------------------------------------------------------------------
    # 关键点 JSON
    # -----------------------------------------------------------------------
    critical_points = []
    for idx in crit_idx:
        critical_points.append({
            "vertex_id":   int(idx),
            "xyz_mm":      [round(float(c), 2) for c in xyz[idx]],
            "Vp_ms":       round(float(Vp[idx]), 1),
            "E_final":     round(float(E_final[idx]), 4),
            "dhdt_mm_per_h": float(f"{dhdt_mean[idx]:.4e}"),
            "rul_mean_h":  round(float(rul_mean[idx]), 0),
            "rul_mean_yr": round(float(rul_mean_yr[idx]), 2),
            "rul_lower_h": round(float(rul_lower[idx]), 0),
            "risk_level":  int(risk_level[idx]),
        })
    with open(OUT_CRIT, "w", encoding="utf-8") as fh:
        json.dump(critical_points, fh, ensure_ascii=False, indent=2)
    print(f"[DEG] → {OUT_CRIT.name}（Top-{top_n} 失效点）", flush=True)

    # -----------------------------------------------------------------------
    # 摘要
    # -----------------------------------------------------------------------
    summary = {
        "parameters": {
            "h0_mm":   h0,
            "hmin_mm": hmin,
            "C_s_kg_m3": C_s,
            "eta_impact": ETA_IMPACT,
            "hours_per_year_sediment": HOURS_PER_YEAR,
        },
        "global_min_life": {
            "vertex_id":    argmin,
            "rul_mean_h":   round(min_rul_h, 0),
            "rul_mean_yr":  round(min_rul_yr, 2),
            "rul_lower_h":  round(float(rul_lower[argmin]), 0),
            "rul_lower_yr": round(float(rul_lower[argmin] / HOURS_PER_YEAR), 2),
        },
        "thickness_rate_mm_per_h": {
            "min":  float(f"{dhdt_mean.min():.4e}"),
            "max":  float(f"{dhdt_mean.max():.4e}"),
            "mean": float(f"{dhdt_mean.mean():.4e}"),
        },
        "failed_fraction_at_end": {
            "time_h":  float(times_h[-1]),
            "pct":     round(float(frac_failed[-1]), 2),
        },
        "model": "linear degradation h(t)=h0 - (dh/dt)*t (v1, constant-rate)",
        "note": "h0/h_min 为工程缺省值，待 DEC 图纸确认；速率恒定假设未考虑磨损后几何/冲击角变化",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"[DEG] → {OUT_JSON.name}")
    print("Step-12 退化与寿命完成。\n")

    return summary


if __name__ == "__main__":
    run_degradation()
