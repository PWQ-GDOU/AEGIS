"""
step18_fatigue_erosion_rul.py
==============================
疲劳-磨蚀联合寿命预测

原理
----
水斗同时承受：
  1. 磨蚀损伤：D_erosion = t / T_RUL_erosion
  2. 疲劳损伤：D_fatigue = Σ (n_i / N_f_i)  [Miner 线性累积]
     N_f_i = (σ_fatigue_limit / σ_a_i)^m × N_ref

联合失效准则（Miner-erosion 耦合）：
  D_erosion + D_fatigue = 1
  → T_RUL_combined = T_RUL_erosion × (1 - D_fatigue_per_hour × T_RUL_erosion)
    当 D_fatigue_per_hour × T_RUL_erosion ≥ 1 时，疲劳先于磨蚀失效

工况权重
--------
从现场动应力数据（13 个工况）提取各工况应力幅值，
结合典型运行时间比例（汛期高含沙时段）计算每小时疲劳损伤率。

材料参数（来自 DEC 动应力试验报告）
------------------------------------
  弹性模量 E = 206 GPa
  泊松比 ν = 0.3
  材料：ZG0Cr13Ni4Mo（马氏体不锈钢）
  S-N 曲线参考：σ_fatigue_limit ≈ 300 MPa（10^7 循环），m = 5

输出
----
output/combined_rul_map.npz
output/fatigue_summary.json
output/combined_rul_comparison.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from step12_degradation import HOURS_PER_YEAR, H0_DEFAULT, HMIN_DEFAULT
from step10_unified_pipeline import OUTPUT_DIR

WORKSPACE = Path(__file__).parent

# 动应力数据目录
STRESS_DIR = Path(r"DecPelton2024/DecPelton2024\DEC PELTON 2024-150MW级冲击式国产化转轮研发共享数据包\3-现场动应力试验实测数据\3.1-现场动应力实测数据")

OUT_NPZ  = OUTPUT_DIR / "combined_rul_map.npz"
OUT_JSON = OUTPUT_DIR / "fatigue_summary.json"
OUT_PNG  = OUTPUT_DIR / "combined_rul_comparison.png"

# ---------------------------------------------------------------------------
# 材料 S-N 参数（ZG0Cr13Ni4Mo，来自 DEC 报告）
# ---------------------------------------------------------------------------
SIGMA_FATIGUE_LIMIT: float = 150.0   # MPa，保守有效疲劳极限（Goodman 修正，安全系数 2）
N_REF:               float = 1e7     # 参考循环数
M_SN:                float = 5.0     # S-N 曲线斜率指数（马氏体不锈钢典型值）

# 典型汛期工况时间比例（高含沙时段，基于金窝电站运行规律估计）
# 汛期主要在 80–140 MW 高负荷运行，低负荷较少
LOAD_SCHEDULE: dict[str, float] = {
    "额定空转": 0.02,
    "30MW":    0.03,
    "40MW":    0.03,
    "50MW":    0.04,
    "60MW":    0.05,
    "70MW":    0.06,
    "80MW":    0.10,
    "90MW":    0.10,
    "100MW":   0.12,
    "110MW":   0.12,
    "120MW":   0.12,
    "130MW":   0.11,
    "140MW":   0.10,
}


# ---------------------------------------------------------------------------
# 1. 读取动应力数据，提取各工况应力幅值
# ---------------------------------------------------------------------------

def load_stress_amplitude(stress_dir: Path) -> dict[str, dict]:
    """
    读取各工况 .txt 文件，提取：
      - 应力均值 σ_mean [MPa]
      - 应力幅值 σ_a = (max-min)/2 [MPa]（峰峰值的一半）
      - 主频 f_dominant [Hz]（FFT 峰值频率）
    """
    results = {}
    for fname in sorted(stress_dir.glob("*.txt")):
        key = fname.stem   # e.g. "100MW"
        try:
            data = np.loadtxt(str(fname), skiprows=1)
            t    = data[:, 0]
            sig  = data[:, 1]

            sigma_mean = float(sig.mean())
            sigma_a    = float((sig.max() - sig.min()) / 2.0)

            # FFT 主频
            dt   = float(np.median(np.diff(t)))
            fs   = 1.0 / dt if dt > 0 else 1000.0
            N    = len(sig)
            freq = np.fft.rfftfreq(N, d=dt)
            amp  = np.abs(np.fft.rfft(sig - sigma_mean))
            # 排除直流（0 Hz）
            amp[0] = 0
            f_dom  = float(freq[np.argmax(amp)])

            results[key] = {
                "sigma_mean_MPa": round(sigma_mean, 3),
                "sigma_a_MPa":    round(sigma_a, 3),
                "f_dominant_Hz":  round(f_dom, 2),
                "n_samples":      N,
                "duration_s":     round(float(t[-1] - t[0]), 3),
            }
            print(f"  {key:12s}: σ_mean={sigma_mean:.1f}  σ_a={sigma_a:.1f} MPa  "
                  f"f_dom={f_dom:.1f} Hz", flush=True)
        except Exception as e:
            print(f"  {key}: 读取失败 {e}", flush=True)

    return results


# ---------------------------------------------------------------------------
# 2. 计算每小时疲劳损伤率
# ---------------------------------------------------------------------------

def fatigue_damage_per_hour(
    stress_data: dict[str, dict],
    schedule:    dict[str, float],
) -> float:
    """
    D_fatigue_per_hour = Σ_i [ time_fraction_i × f_i × 3600 / N_f_i ]

    N_f_i = N_REF × (σ_fatigue_limit / σ_a_i)^m
    当 σ_a_i < σ_fatigue_limit 时，N_f_i = inf（不产生疲劳损伤）
    """
    D_per_h = 0.0
    detail  = {}

    for key, frac in schedule.items():
        if key not in stress_data:
            continue
        sd    = stress_data[key]
        sigma_a = sd["sigma_a_MPa"]
        f_dom   = sd["f_dominant_Hz"]

        if sigma_a <= 0:
            Nf = np.inf
            d  = 0.0
        else:
            Nf = N_REF * (SIGMA_FATIGUE_LIMIT / sigma_a) ** M_SN
            cycles_per_h = f_dom * 3600.0
            d = frac * cycles_per_h / Nf

        D_per_h += d
        detail[key] = {
            "sigma_a_MPa":    sigma_a,
            "f_dom_Hz":       f_dom,
            "time_fraction":  frac,
            "Nf":             float(Nf) if np.isfinite(Nf) else None,
            "D_per_h":        float(d),
        }

    return D_per_h, detail


# ---------------------------------------------------------------------------
# 3. 联合寿命计算
# ---------------------------------------------------------------------------

def combined_rul(
    rul_erosion: np.ndarray,   # (N,) h，来自 step12
    D_fat_per_h: float,
) -> np.ndarray:
    """
    联合失效：D_erosion(t) + D_fatigue(t) = 1
    D_erosion(t) = t / T_RUL_e
    D_fatigue(t) = D_fat_per_h × t

    解方程：t/T_RUL_e + D_fat_per_h × t = 1
    → t_combined = T_RUL_e / (1 + D_fat_per_h × T_RUL_e)

    当 D_fat_per_h = 0 时退化为纯磨蚀寿命。
    """
    finite = np.isfinite(rul_erosion)
    rul_c  = rul_erosion.copy()
    rul_c[finite] = rul_erosion[finite] / (1.0 + D_fat_per_h * rul_erosion[finite])
    return rul_c


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------

def run_combined_rul() -> None:
    print("=" * 60)
    print("AEGIS Step-18 疲劳-磨蚀联合寿命")
    print("=" * 60)

    # 读取动应力数据
    print("\n[FAT] 读取现场动应力数据 ...", flush=True)
    stress_data = load_stress_amplitude(STRESS_DIR)

    # 每小时疲劳损伤率
    D_per_h, detail = fatigue_damage_per_hour(stress_data, LOAD_SCHEDULE)
    T_fatigue_only = 1.0 / D_per_h if D_per_h > 0 else np.inf
    print(f"\n[FAT] 每小时疲劳损伤率: {D_per_h:.4e}")
    print(f"[FAT] 纯疲劳寿命（均匀应力假设）: {T_fatigue_only:,.0f} h "
          f"≈ {T_fatigue_only/HOURS_PER_YEAR:.1f} 年（高含沙时段）", flush=True)

    # 加载磨蚀寿命（step12 线性版）
    rul_data    = np.load(str(OUTPUT_DIR / "rul_map.npz"))
    rul_erosion = rul_data["rul_mean_h"].astype(np.float64)
    rul_lower   = rul_data["rul_lower_h"].astype(np.float64)
    xyz         = rul_data["xyz"]
    risk_level  = rul_data["risk_level"]

    # 联合寿命
    rul_combined       = combined_rul(rul_erosion, D_per_h)
    rul_combined_lower = combined_rul(rul_lower,   D_per_h)

    finite = np.isfinite(rul_combined)
    min_c  = float(rul_combined[finite].min())
    min_e  = float(rul_erosion[np.isfinite(rul_erosion)].min())
    reduction_pct = 100.0 * (min_e - min_c) / (min_e + 1e-30)

    print(f"\n[FAT] 全局最短寿命:")
    print(f"  纯磨蚀:   {min_e:.0f} h")
    print(f"  联合寿命: {min_c:.0f} h  (缩短 {reduction_pct:.1f}%)", flush=True)

    # 各风险等级平均寿命对比
    print("\n[FAT] 各风险等级平均寿命对比 (h):")
    for lv in range(4):
        mask = (risk_level == lv) & finite
        if mask.any():
            me = float(np.mean(rul_erosion[mask & np.isfinite(rul_erosion)]))
            mc = float(np.mean(rul_combined[mask]))
            print(f"  L{lv}: 磨蚀={me:8.0f}h  联合={mc:8.0f}h  "
                  f"缩短={100*(me-mc)/(me+1e-9):.1f}%", flush=True)

    # 保存
    np.savez_compressed(
        str(OUT_NPZ),
        xyz=xyz, risk_level=risk_level,
        rul_erosion_h      = rul_erosion.astype(np.float32),
        rul_combined_h     = rul_combined.astype(np.float32),
        rul_combined_lower = rul_combined_lower.astype(np.float32),
    )
    print(f"\n[FAT] → {OUT_NPZ.name}", flush=True)

    # 对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fatigue-Erosion Combined RUL vs Pure Erosion RUL", fontsize=12)

    # 散点：纯磨蚀 vs 联合（抽样 5000 点）
    rng  = np.random.default_rng(42)
    idx  = rng.choice(np.where(finite)[0], min(5000, finite.sum()), replace=False)
    axes[0].scatter(rul_erosion[idx], rul_combined[idx],
                    c=risk_level[idx], cmap="RdYlGn_r",
                    s=3, alpha=0.5)
    lim = max(rul_erosion[idx].max(), rul_combined[idx].max())
    axes[0].plot([0, lim], [0, lim], "k--", linewidth=1, label="equal")
    axes[0].set_xlabel("Pure erosion T_RUL [h]")
    axes[0].set_ylabel("Combined T_RUL [h]")
    axes[0].set_title("Combined < Erosion (fatigue shortens life)")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # 各工况应力幅值柱状图
    keys   = [k for k in LOAD_SCHEDULE if k in stress_data]
    sigmas = [stress_data[k]["sigma_a_MPa"] for k in keys]
    fracs  = [LOAD_SCHEDULE[k] for k in keys]
    x = np.arange(len(keys))
    ax2 = axes[1]
    bars = ax2.bar(x, sigmas, color=[
        plt.cm.RdYlGn_r(f * 3) for f in fracs], edgecolor="white", linewidth=0.5)
    ax2.axhline(SIGMA_FATIGUE_LIMIT, color="red", linestyle="--",
                linewidth=1, label=f"fatigue limit {SIGMA_FATIGUE_LIMIT} MPa")
    ax2.set_xticks(x); ax2.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Stress amplitude [MPa]")
    ax2.set_title("Stress amplitude by operating condition")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(str(OUT_PNG), dpi=150)
    plt.close()
    print(f"[FAT] → {OUT_PNG.name}", flush=True)

    # 摘要 JSON
    summary = {
        "material": {
            "sigma_fatigue_limit_MPa": SIGMA_FATIGUE_LIMIT,
            "N_ref": N_REF,
            "m_SN":  M_SN,
            "note":  "ZG0Cr13Ni4Mo, E=206GPa, nu=0.3 (from DEC stress test report)",
        },
        "fatigue_damage_per_hour": float(D_per_h),
        "pure_fatigue_life_h":     float(T_fatigue_only),
        "global_min_rul": {
            "pure_erosion_h":  round(min_e, 0),
            "combined_h":      round(min_c, 0),
            "reduction_pct":   round(reduction_pct, 1),
        },
        "load_schedule": LOAD_SCHEDULE,
        "stress_by_condition": {
            k: {"sigma_a_MPa": v["sigma_a_MPa"], "f_dom_Hz": v["f_dominant_Hz"]}
            for k, v in stress_data.items()
        },
        "fatigue_detail": detail,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FAT] → {OUT_JSON.name}")
    print("Step-18 疲劳-磨蚀联合寿命完成。\n")


if __name__ == "__main__":
    run_combined_rul()
