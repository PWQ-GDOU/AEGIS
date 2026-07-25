"""
step14_report.py
================
灰盒 Layer-5：可视化与汇总报告

输出
----
output/erosion_final_heatmap.stl   — E_final 热力图 STL（ParaView 可打开）
output/rul_heatmap.stl             — T_RUL 热力图 STL
output/uncertainty_heatmap.stl     — GPR sigma 热力图 STL
output/degradation_curves.png      — Top-10 关键点壁厚退化曲线
output/report_summary.json         — 完整验收指标与关键结论
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

WORKSPACE  = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "output"

# ---------------------------------------------------------------------------
# STL 彩色输出（将标量场编码为顶点颜色，ParaView 可读）
# ---------------------------------------------------------------------------

def write_colored_stl(
    xyz: np.ndarray,
    faces: np.ndarray | None,
    scalar: np.ndarray,
    out_path: Path,
    cmap_name: str = "hot",
) -> None:
    """
    将逐顶点标量场写入二进制 STL。
    颜色通过 matplotlib colormap 映射后编码进 attribute byte count 字段
    （R5G6B5 格式，VisCAM/SolidView 约定）。
    """
    scalar_norm = (scalar - scalar.min()) / (scalar.max() - scalar.min() + 1e-30)
    cmap = cm.get_cmap(cmap_name)
    colors_rgba = cmap(scalar_norm)   # (N, 4)

    # 若无面片索引，用顶点顺序构造三角面（每3个顶点一个面）
    if faces is None:
        n_tri = len(xyz) // 3
        xyz_tri = xyz[:n_tri * 3].reshape(n_tri, 3, 3)
        face_colors = colors_rgba[:n_tri * 3:3]
    else:
        n_tri = len(faces)
        xyz_tri = xyz[faces]          # (M, 3, 3)
        face_colors = colors_rgba[faces[:, 0]]   # 用第一顶点颜色代表面

    header = b"AEGIS erosion heatmap" + b" " * (80 - 21)
    buf = bytearray(header)
    buf += struct.pack("<I", n_tri)

    for i in range(n_tri):
        v0, v1, v2 = xyz_tri[i]
        normal = np.cross(v1 - v0, v2 - v0)
        nlen = np.linalg.norm(normal)
        if nlen > 1e-12:
            normal /= nlen
        buf += struct.pack("<fff", *normal.astype(np.float32))
        buf += struct.pack("<fff", *v0.astype(np.float32))
        buf += struct.pack("<fff", *v1.astype(np.float32))
        buf += struct.pack("<fff", *v2.astype(np.float32))
        # R5G6B5 颜色编码
        r = int(face_colors[i, 0] * 31)
        g = int(face_colors[i, 1] * 63)
        b = int(face_colors[i, 2] * 31)
        rgb565 = (r << 11) | (g << 5) | b
        buf += struct.pack("<H", rgb565 | 0x8000)   # bit15=1 表示有颜色

    out_path.write_bytes(bytes(buf))
    print(f"  → {out_path.name}  ({out_path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# 退化曲线图
# ---------------------------------------------------------------------------

def plot_degradation_curves(
    h0: float = 30.0,
    hmin: float = 15.0,
) -> None:
    curve_data = np.load(str(OUTPUT_DIR / "degradation_curves.npz"))
    times_h  = curve_data["times_h"]
    curves   = curve_data["curves"]       # (top_n, T)
    crit_idx = curve_data["crit_idx"]
    frac_failed = curve_data["frac_failed"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("AEGIS Degradation Curves — Top-10 Critical Points", fontsize=12)

    cmap = cm.get_cmap("plasma", len(crit_idx))
    for j, idx in enumerate(crit_idx):
        ax1.plot(times_h, curves[j], color=cmap(j),
                 label=f"v#{idx}", linewidth=1.4)
    ax1.axhline(hmin, color="red", linestyle="--", linewidth=1, label=f"h_min={hmin}mm")
    ax1.set_xlabel("Time [h]")
    ax1.set_ylabel("Wall thickness [mm]")
    ax1.set_title("Top-10 Critical Points")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.plot(times_h, frac_failed, "b-o", markersize=3, linewidth=1.5)
    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("Failed vertices [%]")
    ax2.set_title("Global Failure Fraction vs Time")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "degradation_curves.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  → {out.name}")


# ---------------------------------------------------------------------------
# 汇总报告 JSON
# ---------------------------------------------------------------------------

def build_report_summary() -> dict:
    # 加载各模块输出
    gpr_j  = json.loads((OUTPUT_DIR / "gpr_summary.json").read_text("utf-8"))
    deg_j  = json.loads((OUTPUT_DIR / "degradation_summary.json").read_text("utf-8"))
    pip_j  = json.loads((OUTPUT_DIR / "pipeline_summary.json").read_text("utf-8"))
    sens_j = json.loads((OUTPUT_DIR / "sensitivity_curves.json").read_text("utf-8"))
    crit   = json.loads((OUTPUT_DIR / "critical_points.json").read_text("utf-8"))

    # 敏感性：各参数 T_RUL 变化幅度
    sens_range = {
        p: {
            "min_h":  round(min(v["rul_h"]), 1),
            "max_h":  round(max(v["rul_h"]), 1),
            "ratio":  round(max(v["rul_h"]) / (min(v["rul_h"]) + 1e-9), 2),
        }
        for p, v in sens_j.items()
    }

    summary = {
        "project": "AEGIS — Pelton Bucket Erosion Grey-box Degradation Model",
        "model_architecture": "Oka(physics) + GPR(residual correction) + linear degradation",
        "geometry": {
            "model_runner_verts": pip_j["model_runner"]["n_verts"],
            "full_runner_verts":  pip_j["full_runner"]["n_verts"],
            "cross_scale_spearman_rho": pip_j["cross_scale_spearman_rho"],
        },
        "physics_layer": {
            "models": ["Oka", "Finnie", "Tabakoff"],
            "bug_fixes": pip_j["bug_fixes_applied"],
            "full_runner_risk_high_pct": pip_j["full_runner"]["risk_high_pct"],
            "impact_angle_median_deg":   pip_j["full_runner"]["impact_angle_med"],
        },
        "gpr_layer": {
            "n_training_points":   gpr_j["n_inducing_points"],
            "holdout_spearman_rho": gpr_j["validation"]["holdout_spearman_rho"],
            "holdout_mae":          gpr_j["validation"]["holdout_mae"],
            "E_final_vs_Oka_rho":   gpr_j["validation"]["E_final_vs_Oka_spearman_rho"],
            "mean_correction_factor": gpr_j["correction_factor"]["mean"],
            "label_strategy":       gpr_j["label_strategy"],
        },
        "degradation_layer": {
            "h0_mm":   deg_j["parameters"]["h0_mm"],
            "hmin_mm": deg_j["parameters"]["hmin_mm"],
            "global_min_rul_h":    deg_j["global_min_life"]["rul_mean_h"],
            "global_min_rul_lower_h": deg_j["global_min_life"]["rul_lower_h"],
            "critical_vertex_id":  deg_j["global_min_life"]["vertex_id"],
            "note": deg_j["note"],
        },
        "top10_critical_points": crit,
        "sensitivity_analysis": sens_range,
        "validation_metrics": {
            "G1_spatial_erosion": {
                "cross_scale_spearman": pip_j["cross_scale_spearman_rho"],
                "gpr_holdout_spearman": gpr_j["validation"]["holdout_spearman_rho"],
                "status": "PASS" if gpr_j["validation"]["holdout_spearman_rho"] > 0.6 else "REVIEW",
            },
            "G4_scale_consistency": {
                "spearman_rho": pip_j["cross_scale_spearman_rho"],
                "target": ">0.85",
                "status": "PASS" if pip_j["cross_scale_spearman_rho"] > 0.85 else "FAIL",
            },
        },
        "pending_confirmations": [
            "h0 and h_min: confirm from DEC drawings (current defaults: 30/15 mm)",
            "GPR label upgrade: obtain measured wear scan data for P0 labels",
            "High-fidelity CFD (VOF+DPM+SST-CC) to replace simplified velocity model",
        ],
    }
    return summary


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("AEGIS Step-14 可视化与汇总报告")
    print("=" * 60)

    # 加载 GPR 数据（含 E_final, sigma, faces）
    gpr_data = np.load(str(OUTPUT_DIR / "vertices_full_gpr.npz"))
    xyz        = gpr_data["xyz"]
    E_final    = gpr_data["E_final"].astype(np.float64)
    sigma_E    = gpr_data["sigma_E"].astype(np.float64)

    # 加载 RUL 数据
    rul_data  = np.load(str(OUTPUT_DIR / "rul_map.npz"))
    rul_mean  = rul_data["rul_mean_h"].astype(np.float64)
    rul_plot  = np.where(np.isfinite(rul_mean), rul_mean, rul_mean[np.isfinite(rul_mean)].max())

    # 加载面片索引（来自 step10 model 数据，full 数据同结构）
    full_data = np.load(str(OUTPUT_DIR / "vertices_full.npz"))
    # vertices_full.npz 不含 faces（step10 只存数组字段）
    # 用 None 让 write_colored_stl 按顶点顺序构造面
    faces = None

    print("[RPT] 生成侵蚀率热力图 STL ...", flush=True)
    write_colored_stl(xyz, faces, E_final,
                      OUTPUT_DIR / "erosion_final_heatmap.stl", "hot")

    print("[RPT] 生成剩余寿命热力图 STL ...", flush=True)
    write_colored_stl(xyz, faces, rul_plot,
                      OUTPUT_DIR / "rul_heatmap.stl", "RdYlGn")

    print("[RPT] 生成不确定度热力图 STL ...", flush=True)
    write_colored_stl(xyz, faces, sigma_E,
                      OUTPUT_DIR / "uncertainty_heatmap.stl", "cool")

    print("[RPT] 生成退化曲线图 ...", flush=True)
    plot_degradation_curves()

    print("[RPT] 生成汇总报告 JSON ...", flush=True)
    summary = build_report_summary()
    out_json = OUTPUT_DIR / "report_summary.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"  → {out_json.name}")

    # 打印关键指标
    print("\n" + "=" * 60)
    print("AEGIS 灰盒退化模型 — 关键指标汇总")
    print("=" * 60)
    v = summary["validation_metrics"]
    print(f"  跨尺度 Spearman ρ : {v['G4_scale_consistency']['spearman_rho']:.4f}  "
          f"[{v['G4_scale_consistency']['status']}]")
    print(f"  GPR 保留集 Spearman ρ : {v['G1_spatial_erosion']['gpr_holdout_spearman']:.4f}  "
          f"[{v['G1_spatial_erosion']['status']}]")
    d = summary["degradation_layer"]
    print(f"  全局最短 T_RUL : {d['global_min_rul_h']:.0f} h  "
          f"(保守下界 {d['global_min_rul_lower_h']:.0f} h)")
    s = summary["sensitivity_analysis"]
    print(f"  含沙量敏感性 T_RUL 比值 : {s['C_s']['ratio']:.1f}×  "
          f"({s['C_s']['min_h']}–{s['C_s']['max_h']} h)")
    print(f"  水头敏感性   T_RUL 比值 : {s['H']['ratio']:.1f}×")
    print("=" * 60)
    print("Step-14 完成。所有输出已写入 output/\n")


if __name__ == "__main__":
    main()
