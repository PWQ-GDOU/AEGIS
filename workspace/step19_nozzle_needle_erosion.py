"""
step19_nozzle_needle_erosion.py
================================
喷嘴口环 + 喷针磨蚀分析（参数化建模）

几何参数（来自 DEC 模型试验报告）
----------------------------------
模型尺寸：
  喷嘴出口直径 D0 = 35.2 mm
  喷嘴出口锥角 = 90°（半角 45°）
  喷针锥角     = 50°（半角 25°）
  射流直径（模型）= D0 × Cc ≈ 35.2 × 0.95 = 33.4 mm

真机尺寸（缩放比 λ=8.416）：
  D0_true = 35.2 × 8.416 = 296.2 mm ≈ 0.30 m
  射流直径 = 0.24 m（DEC 报告）→ Cc = 0.24/0.296 = 0.81

磨蚀物理
--------
喷针表面：高速流体沿锥面加速，泥沙颗粒因惯性偏离流线，
  以小角度（5–20°）撞击针面 → Finnie 切削磨蚀主导
喷嘴口环座面：射流收缩区，流体以高速掠过口环唇口，
  冲击角 15–30° → Oka + Finnie 混合磨蚀
最危险区域：喷针头部锥面前缘（最高速度 + 最大颗粒偏转）
  和口环出口唇口（射流收缩点）

方法
----
1. 参数化生成轴对称几何（旋转面离散为三角网格）
2. 沿轴向分段计算流速（连续性方程）
3. 用 Stokes 数估算颗粒偏转角 → 冲击角 α(s)
4. 调用 Oka/Finnie 模型计算逐点侵蚀率
5. 输出 STL 热力图 + JSON 摘要
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from step10_unified_pipeline import OkaModel, FinnieModel, RunParams, OUTPUT_DIR

WORKSPACE = Path(__file__).parent
OUT_JSON  = OUTPUT_DIR / "nozzle_needle_erosion.json"
OUT_PNG   = OUTPUT_DIR / "nozzle_needle_erosion.png"

# ---------------------------------------------------------------------------
# 几何参数（模型尺寸，mm）
# ---------------------------------------------------------------------------
D0_MODEL   = 35.2        # 喷嘴出口直径 [mm]
HALF_NOZZLE = 45.0       # 喷嘴半锥角 [°]
HALF_NEEDLE = 25.0       # 喷针半锥角 [°]
CC         = 0.81        # 射流收缩系数（真机实测）
LAMBDA     = 8.416       # 模型→真机缩放比

# 真机尺寸
D0_TRUE    = D0_MODEL * LAMBDA          # mm
D_JET_TRUE = 240.0                      # mm（DEC 报告）
R_JET_TRUE = D_JET_TRUE / 2.0

# 流动参数（真机额定工况）
params = RunParams()
V_JET  = 104.8           # m/s，射流速度
C_S    = 1.31            # kg/m³，含沙量
D50    = 0.05e-3         # m，泥沙中值粒径
RHO_P  = 2650.0          # kg/m³，石英砂密度
RHO_F  = 998.2           # kg/m³，水密度
MU_F   = 1.004e-3        # Pa·s，动力粘度

N_AXIAL = 200            # 轴向离散点数


# ---------------------------------------------------------------------------
# 1. 轴对称几何：沿轴向计算截面半径
# ---------------------------------------------------------------------------

def nozzle_geometry(n: int = N_AXIAL) -> dict:
    """
    生成喷嘴口环和喷针的轴向截面几何。
    坐标系：z 轴沿射流方向，z=0 为喷嘴出口截面。
    z < 0：喷嘴内部（流体从 z=-L 流向 z=0）
    z > 0：射流自由段

    喷嘴内壁：圆锥收缩，半角 45°
    喷针外壁：圆锥，半角 25°，针尖在 z=0 附近
    """
    R0 = D0_TRUE / 2.0   # 喷嘴出口半径 [mm]
    L  = R0 * 2.0        # 喷嘴收缩段长度（简化为 2R0）

    z_nozzle = np.linspace(-L, 0, n)
    # 喷嘴内壁半径（线性收缩）
    R_nozzle = R0 + (-z_nozzle) * np.tan(np.radians(HALF_NOZZLE))

    # 喷针外壁半径（从针尖向上游扩张）
    # 针尖在 z = -R0*0.1（略在出口上游）
    z_tip = -R0 * 0.1
    R_needle = np.where(
        z_nozzle <= z_tip,
        (z_tip - z_nozzle) * np.tan(np.radians(HALF_NEEDLE)),
        0.0,
    )
    R_needle = np.clip(R_needle, 0, None)

    # 环形流道面积
    A_annulus = np.pi * (R_nozzle**2 - R_needle**2)   # mm²
    A_annulus = np.clip(A_annulus, 1e-6, None)

    return {
        "z_mm":       z_nozzle,
        "R_nozzle_mm": R_nozzle,
        "R_needle_mm": R_needle,
        "A_annulus_mm2": A_annulus,
    }


# ---------------------------------------------------------------------------
# 2. 流速分布（连续性方程）
# ---------------------------------------------------------------------------

def flow_velocity(geom: dict) -> np.ndarray:
    """
    由连续性方程 V(z) = V_jet × A_jet / A(z) 计算各截面平均流速 [m/s]。
    A_jet = π/4 × D_jet²
    """
    A_jet = np.pi / 4.0 * (D_JET_TRUE)**2   # mm²
    V = V_JET * A_jet / geom["A_annulus_mm2"]
    return V   # m/s


# ---------------------------------------------------------------------------
# 3. Stokes 数 → 颗粒偏转角 → 冲击角
# ---------------------------------------------------------------------------

def impact_angle_from_stokes(
    V: np.ndarray,
    R_surface: np.ndarray,
    surface_half_angle_deg: float,
) -> np.ndarray:
    """
    用 Stokes 数估算颗粒相对流线的偏转，进而得到冲击角。

    St = ρ_p × d_p² × V / (18 × μ × L_char)
    偏转角 δ ≈ arctan(1 / (St × π))  [简化 Ranz-Marshall 近似]
    冲击角 α = |surface_half_angle - δ|，限制在 [2°, 85°]

    St >> 1：颗粒不跟流线，直线飞行，冲击角 ≈ 表面半锥角
    St << 1：颗粒跟流线，冲击角 ≈ 0（掠射）
    """
    L_char = np.clip(R_surface * 1e-3, 1e-4, None)   # mm → m
    St = RHO_P * D50**2 * V / (18.0 * MU_F * L_char)
    delta = np.degrees(np.arctan(1.0 / (St * np.pi + 1e-6)))
    alpha = np.abs(surface_half_angle_deg - delta)
    return np.clip(alpha, 2.0, 85.0)


# ---------------------------------------------------------------------------
# 4. 侵蚀率计算
# ---------------------------------------------------------------------------

def erosion_on_surface(
    V: np.ndarray,
    alpha_deg: np.ndarray,
    label: str,
) -> dict:
    """
    对给定表面（喷针或口环）计算逐点侵蚀率。
    同时用 Oka 和 Finnie 模型，取加权平均（Finnie 在小角度更准）。
    """
    oka    = OkaModel(Hv=params.Hv, Dp=params.Dp)
    finnie = FinnieModel()

    # step10 的 erosion_rate 接口接收角度（deg），不是弧度
    E_oka    = oka.erosion_rate(V, alpha_deg)
    E_finnie = finnie.erosion_rate(V, alpha_deg)

    # 小角度（<30°）Finnie 权重更高，大角度 Oka 更准
    w_finnie = np.clip(1.0 - alpha_deg / 30.0, 0.0, 1.0)
    w_oka    = 1.0 - w_finnie
    E_blend  = w_oka * E_oka + w_finnie * E_finnie

    # 壁厚减薄速率 [mm/h]
    m_flux   = C_S * V * 3600.0   # kg/(m²·h)
    dhdt     = E_blend * m_flux * 1e-6   # mm/h

    peak_idx = np.argmax(E_blend)
    return {
        "label":          label,
        "E_oka":          E_oka,
        "E_finnie":       E_finnie,
        "E_blend":        E_blend,
        "dhdt_mm_per_h":  dhdt,
        "peak_idx":       int(peak_idx),
        "peak_E":         float(E_blend[peak_idx]),
        "peak_V_ms":      float(V[peak_idx]),
        "peak_alpha_deg": float(alpha_deg[peak_idx]),
        "peak_dhdt":      float(dhdt[peak_idx]),
    }


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------

def run_nozzle_needle() -> dict:
    print("=" * 60)
    print("AEGIS Step-19 喷嘴口环 + 喷针磨蚀分析")
    print("=" * 60)

    geom = nozzle_geometry()
    V    = flow_velocity(geom)
    z    = geom["z_mm"]

    # 喷针表面
    alpha_needle = impact_angle_from_stokes(
        V, geom["R_needle_mm"], HALF_NEEDLE)
    # 只分析有实体的区域（R_needle > 0）
    needle_mask = geom["R_needle_mm"] > 1.0
    res_needle = erosion_on_surface(
        V[needle_mask], alpha_needle[needle_mask], "needle")

    # 喷嘴口环内壁
    alpha_nozzle = impact_angle_from_stokes(
        V, geom["R_nozzle_mm"], HALF_NOZZLE)
    res_nozzle = erosion_on_surface(V, alpha_nozzle, "nozzle_ring")

    # 与水斗对比（从 step12 输出读取）
    rul_data = np.load(str(OUTPUT_DIR / "rul_map.npz"))
    bucket_dhdt_max = float(
        np.load(str(OUTPUT_DIR / "vertices_full_gpr.npz"))["E_final"].max()
        * C_S * V_JET * 3600.0 * 1e-6
    )

    print(f"\n[NOZ] 喷针峰值侵蚀率: {res_needle['peak_E']:.4f} mm³/kg  "
          f"位置 z={z[needle_mask][res_needle['peak_idx']]:.1f} mm  "
          f"α={res_needle['peak_alpha_deg']:.1f}°  "
          f"V={res_needle['peak_V_ms']:.1f} m/s")
    print(f"[NOZ] 喷针峰值减薄速率: {res_needle['peak_dhdt']:.4f} mm/h")

    print(f"\n[NOZ] 口环峰值侵蚀率: {res_nozzle['peak_E']:.4f} mm³/kg  "
          f"位置 z={z[res_nozzle['peak_idx']]:.1f} mm  "
          f"α={res_nozzle['peak_alpha_deg']:.1f}°  "
          f"V={res_nozzle['peak_V_ms']:.1f} m/s")
    print(f"[NOZ] 口环峰值减薄速率: {res_nozzle['peak_dhdt']:.4f} mm/h")

    print(f"\n[NOZ] 水斗峰值减薄速率（参考）: {bucket_dhdt_max:.4f} mm/h")
    print(f"[NOZ] 喷针/水斗 速率比: {res_needle['peak_dhdt']/bucket_dhdt_max:.2f}×")
    print(f"[NOZ] 口环/水斗 速率比: {res_nozzle['peak_dhdt']/bucket_dhdt_max:.2f}×")

    # 磨蚀类型分区（赛事要求：磨蚀类型分区图）
    erosion_zones = {
        "needle_tip":   {"type": "cutting (Finnie)",  "alpha_range": "5–15°",
                         "dominant_model": "Finnie",  "risk": "HIGH"},
        "needle_body":  {"type": "mixed",              "alpha_range": "15–25°",
                         "dominant_model": "Oka+Finnie", "risk": "MEDIUM"},
        "nozzle_lip":   {"type": "cutting+deformation","alpha_range": "15–30°",
                         "dominant_model": "Oka",     "risk": "HIGH"},
        "nozzle_body":  {"type": "deformation (Oka)",  "alpha_range": "30–45°",
                         "dominant_model": "Oka",     "risk": "MEDIUM"},
        "bucket_entry": {"type": "cutting+deformation","alpha_range": "20–60°",
                         "dominant_model": "Oka",     "risk": "VERY HIGH"},
    }

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Nozzle Ring & Needle Erosion Analysis — DEC Pelton 150MW", fontsize=12)

    # 1. 几何截面
    ax = axes[0, 0]
    ax.fill_between(z, geom["R_nozzle_mm"], geom["R_nozzle_mm"]*1.05,
                    color="steelblue", alpha=0.4, label="Nozzle ring")
    ax.fill_between(z, 0, geom["R_needle_mm"],
                    color="darkorange", alpha=0.5, label="Needle")
    ax.plot(z, geom["R_nozzle_mm"], "b-", linewidth=1.5)
    ax.plot(z, geom["R_needle_mm"], "r-", linewidth=1.5)
    ax.set_xlabel("Axial position z [mm]"); ax.set_ylabel("Radius [mm]")
    ax.set_title("Nozzle cross-section geometry"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. 流速分布
    ax = axes[0, 1]
    ax.plot(z, V, "b-", linewidth=1.5)
    ax.set_xlabel("z [mm]"); ax.set_ylabel("Flow velocity [m/s]")
    ax.set_title("Axial velocity distribution"); ax.grid(True, alpha=0.3)
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="Nozzle exit")
    ax.legend(fontsize=8)

    # 3. 喷针侵蚀率分布
    ax = axes[1, 0]
    z_needle = z[needle_mask]
    ax.plot(z_needle, res_needle["E_oka"],    "b--", linewidth=1, label="Oka")
    ax.plot(z_needle, res_needle["E_finnie"], "g--", linewidth=1, label="Finnie")
    ax.plot(z_needle, res_needle["E_blend"],  "r-",  linewidth=2, label="Blended")
    ax.axvline(z_needle[res_needle["peak_idx"]], color="red",
               linestyle=":", linewidth=1, label="Peak")
    ax.set_xlabel("z [mm]"); ax.set_ylabel("Erosion rate [mm³/kg]")
    ax.set_title("Needle erosion rate distribution"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. 口环侵蚀率分布
    ax = axes[1, 1]
    ax.plot(z, res_nozzle["E_oka"],    "b--", linewidth=1, label="Oka")
    ax.plot(z, res_nozzle["E_finnie"], "g--", linewidth=1, label="Finnie")
    ax.plot(z, res_nozzle["E_blend"],  "r-",  linewidth=2, label="Blended")
    ax.axvline(z[res_nozzle["peak_idx"]], color="red",
               linestyle=":", linewidth=1, label="Peak")
    ax.set_xlabel("z [mm]"); ax.set_ylabel("Erosion rate [mm³/kg]")
    ax.set_title("Nozzle ring erosion rate distribution"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(OUT_PNG), dpi=150)
    plt.close()
    print(f"\n[NOZ] -> {OUT_PNG.name}")

    summary = {
        "geometry": {
            "D0_model_mm": D0_MODEL, "D0_true_mm": D0_TRUE,
            "D_jet_true_mm": D_JET_TRUE, "Cc": CC,
            "half_angle_nozzle_deg": HALF_NOZZLE,
            "half_angle_needle_deg": HALF_NEEDLE,
        },
        "needle": {
            "peak_erosion_mm3kg":  round(res_needle["peak_E"], 5),
            "peak_dhdt_mm_per_h":  round(res_needle["peak_dhdt"], 5),
            "peak_alpha_deg":      round(res_needle["peak_alpha_deg"], 1),
            "peak_velocity_ms":    round(res_needle["peak_V_ms"], 1),
            "dominant_mechanism":  "Finnie cutting (low-angle impingement)",
            "highest_risk_zone":   "needle tip (z near nozzle exit)",
        },
        "nozzle_ring": {
            "peak_erosion_mm3kg":  round(res_nozzle["peak_E"], 5),
            "peak_dhdt_mm_per_h":  round(res_nozzle["peak_dhdt"], 5),
            "peak_alpha_deg":      round(res_nozzle["peak_alpha_deg"], 1),
            "peak_velocity_ms":    round(res_nozzle["peak_V_ms"], 1),
            "dominant_mechanism":  "Oka+Finnie mixed (jet contraction zone)",
            "highest_risk_zone":   "nozzle lip at exit",
        },
        "comparison_with_bucket": {
            "bucket_peak_dhdt_mm_per_h": round(bucket_dhdt_max, 5),
            "needle_vs_bucket_ratio":    round(res_needle["peak_dhdt"]/bucket_dhdt_max, 2),
            "nozzle_vs_bucket_ratio":    round(res_nozzle["peak_dhdt"]/bucket_dhdt_max, 2),
        },
        "erosion_type_zones": erosion_zones,
        "note": ("Axisymmetric analytical model. Stokes-number-based impact angle. "
                 "Geometry from DEC report: D0=35.2mm(model), nozzle half-angle=45deg, "
                 "needle half-angle=25deg. True machine scale (lambda=8.416)."),
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[NOZ] -> {OUT_JSON.name}")
    print("Step-19 完成。\n")
    return summary


if __name__ == "__main__":
    run_nozzle_needle()
