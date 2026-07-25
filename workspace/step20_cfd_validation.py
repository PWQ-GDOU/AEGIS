"""
step20_cfd_validation.py
========================
交叉验证框架：simpleFoam CFD 壁面剪切应力 ↔ GPR 磨蚀率预测

验证策略（坐标系不同，采用统计分布对比）
-----------------------------------------
1. 归一化空间坐标（各自归一化到 [0,1]³）
2. 比较分布形状：偏度、峰度、高强度区占比
3. 比较峰值相对位置（归一化坐标系下）
4. Kolmogorov-Smirnov 检验分布相似性
5. 输出验证报告 JSON + 对比图 PNG
"""
from __future__ import annotations
import json, numpy as np
from pathlib import Path
from scipy.stats import skew, kurtosis, ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent / "output"
REPORT_JSON = OUTPUT_DIR / "cfd_validation_report.json"
REPORT_PNG  = OUTPUT_DIR / "cfd_validation.png"


def load_cfd():
    p = OUTPUT_DIR / "cfd_wss.npz"
    if not p.exists():
        return None, None
    d = np.load(str(p))
    return d["xyz"].astype(float), d["wss_mag"].astype(float)


def load_gpr():
    d = np.load(str(OUTPUT_DIR / "vertices_full_gpr.npz"))
    return d["xyz"].astype(float), d["E_final"].astype(float)


def normalize_xyz(xyz):
    mn, mx = xyz.min(axis=0), xyz.max(axis=0)
    rng = np.where(mx - mn > 0, mx - mn, 1.0)
    return (xyz - mn) / rng


def distribution_stats(vals):
    v = vals / (vals.max() + 1e-12)
    return {
        "mean":     float(v.mean()),
        "std":      float(v.std()),
        "skewness": float(skew(v)),
        "kurtosis": float(kurtosis(v)),
        "p90":      float(np.percentile(v, 90)),
        "high_pct": float((v > 0.5).mean() * 100),   # 高强度区占比 %
    }


def peak_location(xyz_norm, vals):
    """归一化坐标系下的峰值位置"""
    idx = np.argmax(vals)
    return xyz_norm[idx].tolist()


def run_validation():
    print("=" * 60)
    print("AEGIS Step-20  CFD ↔ GPR 交叉验证")
    print("=" * 60)

    xyz_cfd, wss = load_cfd()
    if xyz_cfd is None:
        print("[VAL] cfd_wss.npz 不存在，请先运行 CFD 仿真")
        return {}

    xyz_gpr, E_gpr = load_gpr()

    # 归一化
    xyz_cfd_n = normalize_xyz(xyz_cfd)
    xyz_gpr_n = normalize_xyz(xyz_gpr)

    # 分布统计
    stats_cfd = distribution_stats(wss)
    stats_gpr = distribution_stats(E_gpr)

    # KS 检验（归一化强度分布）
    wss_n = wss / (wss.max() + 1e-12)
    E_n   = E_gpr / (E_gpr.max() + 1e-12)
    ks_stat, ks_pval = ks_2samp(wss_n, E_n)

    # 峰值位置（归一化坐标）
    peak_cfd = peak_location(xyz_cfd_n, wss)
    peak_gpr = peak_location(xyz_gpr_n, E_gpr)
    peak_dist = float(np.linalg.norm(np.array(peak_cfd) - np.array(peak_gpr)))

    # 高强度区占比差异
    high_pct_diff = abs(stats_cfd["high_pct"] - stats_gpr["high_pct"])

    # 验证通过标准：KS p>0.05（分布相似）或高强度区占比差 <15%
    validation_pass = (ks_pval > 0.05) or (high_pct_diff < 15.0)

    result = {
        "status":           "complete",
        "method":           "statistical_distribution_comparison",
        "note":             ("坐标系不同（CFD方块网格 vs GPR真实几何），"
                             "采用归一化统计分布对比代替空间点匹配"),
        "cfd_stats":        stats_cfd,
        "gpr_stats":        stats_gpr,
        "ks_statistic":     round(float(ks_stat), 4),
        "ks_pvalue":        round(float(ks_pval), 4),
        "peak_dist_norm":   round(peak_dist, 4),
        "high_pct_diff":    round(high_pct_diff, 2),
        "validation_pass":  validation_pass,
        "cfd_n_points":     int(len(xyz_cfd)),
        "gpr_n_points":     int(len(xyz_gpr)),
    }

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("CFD (wallShearStress) vs GPR (E_final) — Distribution Comparison",
                 fontsize=10)

    # 1. 归一化强度分布直方图
    ax = axes[0]
    ax.hist(wss_n, bins=50, alpha=0.6, label="CFD |wss|", density=True, color="steelblue")
    ax.hist(E_n,   bins=50, alpha=0.6, label="GPR E",     density=True, color="tomato")
    ax.set_xlabel("Normalized intensity"); ax.set_ylabel("Density")
    ax.set_title(f"Intensity distribution\nKS p={ks_pval:.3f}")
    ax.legend(fontsize=8)

    # 2. CFD 壁面剪切应力空间分布
    ax = axes[1]
    sc = ax.scatter(xyz_cfd_n[:,2], xyz_cfd_n[:,0], c=wss_n,
                    cmap="hot_r", s=3, alpha=0.7, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="|wss| norm")
    ax.set_xlabel("z (norm)"); ax.set_ylabel("x (norm)")
    ax.set_title("CFD wall shear stress\n(normalized coords)")

    # 3. GPR 磨蚀率空间分布
    ax = axes[2]
    sc2 = ax.scatter(xyz_gpr_n[:,2], xyz_gpr_n[:,0], c=E_n,
                     cmap="hot_r", s=1, alpha=0.5, vmin=0, vmax=1)
    plt.colorbar(sc2, ax=ax, label="E norm")
    ax.set_xlabel("z (norm)"); ax.set_ylabel("x (norm)")
    ax.set_title("GPR erosion rate\n(normalized coords)")

    plt.tight_layout()
    plt.savefig(str(REPORT_PNG), dpi=150)
    plt.close()

    REPORT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[VAL] CFD 近壁点: {len(xyz_cfd)}, GPR 顶点: {len(xyz_gpr)}")
    print(f"[VAL] KS 统计量: {ks_stat:.4f}, p={ks_pval:.4f}")
    print(f"[VAL] 高强度区占比 — CFD: {stats_cfd['high_pct']:.1f}%, GPR: {stats_gpr['high_pct']:.1f}%")
    print(f"[VAL] 验证结论: {'PASS ✓' if validation_pass else 'FAIL'}")
    print(f"[VAL] -> {REPORT_JSON.name}")
    print(f"[VAL] -> {REPORT_PNG.name}")
    return result


if __name__ == "__main__":
    run_validation()
