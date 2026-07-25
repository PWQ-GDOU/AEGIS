"""
step30_lagrangian_particle_tracking.py
=======================================
Python 拉格朗日粒子追踪 — 解决局限2

用已收敛的 kOmegaSST 湍流流场追踪泥沙颗粒，
获得每个壁面碰撞点的真实 Vp 和 α，实现同量纲对比：
  CFD_Mansouri(Vp_cfd, α_cfd) vs GPR_Mansouri(Vp_model, α_model)

输出:
  output/particle_hits.npz     — 碰撞点坐标+Vp+alpha+ER
  output/step30_summary.json
  output/particle_tracking_validation.png
"""

from __future__ import annotations
import json, re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

WORKSPACE  = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "output"
OUT_NPZ    = OUTPUT_DIR / "particle_hits.npz"
OUT_JSON   = OUTPUT_DIR / "step30_summary.json"
OUT_PNG    = OUTPUT_DIR / "particle_tracking_validation.png"

# Mansouri (2015) Table 3
K_M, N_M, A1, B1, B2, FS = 2.57e-9, 2.4, 228.5, 7.91, 12.81, 1.0

def mansouri_er(Vp, alpha_deg):
    a = np.deg2rad(np.clip(alpha_deg, 0.5, 89.5))
    s = np.clip(np.sin(a), 1e-6, 1.0)
    fa = A1 * s**B1 * np.clip(1.5-s, 1e-6, None)**B2
    return K_M * FS * np.clip(Vp, 0.1, None)**N_M * fa


def parse_foam_scalar_field(path: Path) -> np.ndarray:
    """解析 OpenFOAM ASCII volScalarField / internalField"""
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"internalField\s+nonuniform List<scalar>\s+(\d+)\s*\n\s*\(\n(.*?)\n\s*\)", 
                  text, re.DOTALL)
    if not m:
        raise ValueError(f"Cannot parse scalar {path.name}")
    data = np.fromstring(m.group(2), sep="\n")
    return data


def parse_foam_vector_field(path: Path) -> np.ndarray:
    """解析 OpenFOAM ASCII volVectorField / internalField"""
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"internalField\s+nonuniform List<vector>\s+(\d+)\s*\n\s*\(\n(.*?)\n\s*\)",
                  text, re.DOTALL)
    if not m:
        raise ValueError(f"Cannot parse vector {path.name}")
    raw = re.findall(r"\(([^)]+)\)", m.group(2))
    data = np.array([[float(x) for x in t.split()] for t in raw], dtype=np.float64)
    return data


def run() -> dict:
    print("=" * 60)
    print("AEGIS Step-30  Python 拉格朗日粒子追踪")
    print("=" * 60)

    # 1. 解析 CFD 场
    print("[PT] 解析 OpenFOAM 场文件 ...", flush=True)
    Cx = parse_foam_scalar_field(OUTPUT_DIR / "cfd_Cx.foam")
    Cy = parse_foam_scalar_field(OUTPUT_DIR / "cfd_Cy.foam")
    Cz = parse_foam_scalar_field(OUTPUT_DIR / "cfd_Cz.foam")
    U  = parse_foam_vector_field(OUTPUT_DIR / "cfd_U.foam")
    cell_centers = np.column_stack([Cx, Cy, Cz])    # 局部坐标系，mm

    print(f"[PT] 单元数: {len(cell_centers):,}")
    print(f"[PT] 坐标范围 x: [{Cx.min():.1f}, {Cx.max():.1f}]  "
          f"y: [{Cy.min():.1f}, {Cy.max():.1f}]  "
          f"z: [{Cz.min():.1f}, {Cz.max():.1f}]")
    print(f"[PT] |U| 范围: [{np.linalg.norm(U,axis=1).min():.2f}, "
          f"{np.linalg.norm(U,axis=1).max():.2f}]")

    # 2. 加载水斗壁面点（目标壁面）
    wss  = np.loadtxt(str(OUTPUT_DIR / "bucket_wss_turb.raw"), comments="#")
    wall_xyz_local = wss[:, :3]      # mm，局部坐标系
    wall_normals_raw = wss[:, 3:6]   # 近似法向（wallShearStress方向与法向有关）

    print(f"[PT] 壁面点数: {len(wall_xyz_local):,}")

    # 3. 构建速度场插值树
    print("[PT] 构建 KDTree 用于速度场插值 ...", flush=True)
    tree_u = cKDTree(cell_centers)

    # 4. 粒子追踪参数
    V_JET    = 104.8          # m/s，射流入口速度（沿局部+X方向）
    D_JET    = 240.0          # mm，射流直径
    R_JET    = D_JET / 2.0    # mm
    N_PART   = 2000           # 粒子数（增加以提高统计量）
    DP_M     = 50e-6          # m，中值粒径（50μm）
    RHO_P    = 2650.0         # kg/m³，颗粒密度
    RHO_F    = 1000.0         # kg/m³，流体密度
    NU_F     = 1e-6           # m²/s，运动粘度
    MU_F     = RHO_F * NU_F   # Pa·s，动力粘度
    # Stokes 弛豫时间: τ_p = ρ_p * Dp² / (18 * μ_f)
    TAU_P_S  = RHO_P * DP_M**2 / (18.0 * MU_F)   # s（约 3.7e-4 s）
    DT_S     = 1e-5           # s，时间步长（约 1/37 松弛时间，足够精确）
    MAX_STEP = 50000          # 最大步数（覆盖 0.5s 追踪时间）
    HIT_DIST_MM = 3.0         # mm，碰壁阈值

    print(f"  Stokes 弛豫时间 τ_p = {TAU_P_S*1000:.4f} ms  时间步 dt = {DT_S*1000:.3f} ms")
    print(f"  射流速度 {V_JET} m/s 下颗粒松弛距离 ≈ {V_JET*TAU_P_S*1000:.1f} mm")

    # 5. 颗粒初始位置：圆形入口 (x=0, r<R_JET)
    rng = np.random.default_rng(42)
    r_part  = R_JET * np.sqrt(rng.uniform(0, 1, N_PART))
    th_part = rng.uniform(0, 2*np.pi, N_PART)
    pos0 = np.zeros((N_PART, 3))
    pos0[:, 0] = 0.0
    pos0[:, 1] = r_part * np.cos(th_part)
    pos0[:, 2] = r_part * np.sin(th_part)
    vel0 = np.zeros((N_PART, 3))
    vel0[:, 0] = V_JET   # m/s（与流场速度单位一致）

    # 6. 对壁面点构建 KDTree 用于碰撞检测
    tree_wall = cKDTree(wall_xyz_local)

    # 7. 追踪循环（正确的 Stokes 惯性方程）
    # 颗粒方程: dv_p/dt = (U_f - v_p) / τ_p
    # 坐标用 mm，速度用 m/s，时间用 s
    # 位置更新: x += v_p * dt * 1000（m/s → mm/s * s = mm）
    print(f"[PT] 追踪 {N_PART} 个颗粒（dt={DT_S*1000:.3f}ms，最大 {MAX_STEP} 步）...", flush=True)
    hit_positions  = []
    hit_Vp         = []
    hit_alpha_deg  = []

    for p_idx in range(N_PART):
        pos_mm = pos0[p_idx].copy()          # mm
        vel_ms = vel0[p_idx].copy()          # m/s（初始=射流速度）

        for step in range(MAX_STEP):
            # 插值流场速度（最近邻单元中心，坐标 mm）
            _, ci = tree_u.query(pos_mm)
            Uf = U[ci]                        # m/s

            # Stokes 阻力（带 Schiller-Naumann 修正）
            slip = Uf - vel_ms
            slip_mag = np.linalg.norm(slip)
            Re_p = RHO_F * slip_mag * DP_M / MU_F + 1e-12
            Cd_corr = 1.0 + 0.15 * Re_p**0.687   # Schiller-Naumann
            accel = Cd_corr * slip / TAU_P_S      # m/s²

            # 积分
            vel_ms  = vel_ms + accel * DT_S
            pos_mm  = pos_mm + vel_ms * DT_S * 1000.0   # m/s * s * 1000 = mm

            # 碰撞检测（速度>0 才算有效冲击）
            vel_mag = np.linalg.norm(vel_ms)
            if vel_mag < 0.5:           # 速度过低，已停止
                break
            d_wall, wi = tree_wall.query(pos_mm)
            if d_wall < HIT_DIST_MM:
                # 碰撞速度 = 粒子当前速度大小
                Vp_ms = vel_mag
                wall_pt = wall_xyz_local[wi]
                # 入射方向
                incident_dir = vel_ms / (vel_mag + 1e-12)
                # 局部壁面法向（最近邻5点SVD拟合）
                _, nbrs = tree_wall.query(wall_pt, k=min(8, len(wall_xyz_local)))
                diffs = wall_xyz_local[nbrs] - wall_xyz_local[nbrs].mean(axis=0)
                _, _, Vt = np.linalg.svd(diffs)
                local_normal = Vt[-1]
                local_normal /= np.linalg.norm(local_normal) + 1e-12
                # 冲击角（粒子速度方向与壁面法向的夹角 → arcsin → 冲击角）
                cos_n = abs(np.dot(incident_dir, local_normal))
                alpha_deg = float(np.degrees(np.arcsin(np.clip(cos_n, 0, 1))))
                hit_positions.append(wall_xyz_local[wi].copy())
                hit_Vp.append(float(Vp_ms))
                hit_alpha_deg.append(alpha_deg)
                break

            # 出域检测
            if pos_mm[0] > 1450 or pos_mm[0] < -50 or abs(pos_mm[1]) > 190 or abs(pos_mm[2]) > 430:
                break

        if (p_idx+1) % 100 == 0:
            print(f"  {p_idx+1}/{N_PART} 粒子，碰撞 {len(hit_positions)}", flush=True)

    if not hit_positions:
        print("[PT] 警告：无粒子碰壁！请检查追踪参数。")
        return {}

    hit_pos = np.array(hit_positions)
    hit_Vp  = np.array(hit_Vp)
    hit_alpha = np.array(hit_alpha_deg)
    ER_cfd  = mansouri_er(hit_Vp, hit_alpha)

    print(f"\n[PT] 碰撞点数: {len(hit_pos)}")
    print(f"[PT] Vp 范围: [{hit_Vp.min():.1f}, {hit_Vp.max():.1f}] m/s")
    print(f"[PT] α 范围: [{hit_alpha.min():.1f}, {hit_alpha.max():.1f}] °")
    print(f"[PT] ER_CFD 范围: [{ER_cfd.min():.3e}, {ER_cfd.max():.3e}]")

    # 8. 变换到全局坐标，与 GPR/Mansouri 比较
    rot        = np.load(str(OUTPUT_DIR / "jet_rotation.npz"))
    R          = rot["R"]
    nozzle_pos = rot["nozzle_pos"]
    hit_global = hit_pos @ R + nozzle_pos

    gpr        = np.load(str(OUTPUT_DIR / "vertices_full_gpr.npz"))
    xyz_gpr    = gpr["xyz"].astype(np.float64)
    E_model    = gpr["E_final"].astype(np.float64)
    tree_gpr   = cKDTree(xyz_gpr)
    dist, idx  = tree_gpr.query(hit_global, k=1)
    valid      = dist < np.percentile(dist, 95)
    print(f"[PT] GPR 匹配距离中位: {np.median(dist):.2f}mm  有效: {valid.sum()}")

    ER_model_matched = E_model[idx[valid]]
    ER_cfd_valid     = ER_cfd[valid]
    rho, pval = spearmanr(ER_cfd_valid, ER_model_matched)
    print(f"\n[PT] 同量纲 Spearman ρ(ER_CFD, ER_Mansouri) = {rho:.4f}  p={pval:.2e}")

    t_c = np.percentile(ER_cfd_valid, 90)
    t_m = np.percentile(ER_model_matched, 90)
    iou = float(np.sum((ER_cfd_valid>=t_c)&(ER_model_matched>=t_m)) /
                (np.sum((ER_cfd_valid>=t_c)|(ER_model_matched>=t_m))+1e-10))
    print(f"[PT] Top-10% IoU: {iou:.4f}")

    # 9. 保存
    np.savez_compressed(str(OUT_NPZ),
                        hit_pos_local=hit_pos.astype(np.float32),
                        hit_pos_global=hit_global.astype(np.float32),
                        Vp=hit_Vp.astype(np.float32),
                        alpha_deg=hit_alpha.astype(np.float32),
                        ER_cfd=ER_cfd.astype(np.float32))

    # 10. 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Lagrangian Particle Tracking: CFD-Mansouri vs Model-Mansouri", fontsize=11)
    si = np.random.choice(valid.sum(), min(200, valid.sum()), replace=False)
    axes[0].scatter(ER_cfd_valid[si], ER_model_matched[si], s=20, alpha=0.5, color="steelblue")
    axes[0].set_xlabel("Mansouri ER from CFD Vp/α [kg/kg]")
    axes[0].set_ylabel("Mansouri ER from model Vp/α [kg/kg]")
    axes[0].set_title(f"Spearman ρ = {rho:.3f}")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(hit_Vp, hit_alpha, s=10, alpha=0.4,
                    c=np.log10(ER_cfd+1e-20), cmap="hot")
    axes[1].set_xlabel("CFD particle Vp [m/s]"); axes[1].set_ylabel("CFD impact α [°]")
    axes[1].set_title("Particle impact map"); axes[1].grid(True, alpha=0.3)

    sc = axes[2].scatter(hit_pos[:,0], np.linalg.norm(hit_pos[:,1:3],axis=1),
                         c=np.log10(ER_cfd+1e-20), cmap="hot", s=8)
    plt.colorbar(sc, ax=axes[2], label="log10(ER_CFD)")
    axes[2].set_xlabel("local x [mm]"); axes[2].set_ylabel("r_yz [mm]")
    axes[2].set_title("Hit distribution on bucket (local frame)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(OUT_PNG), dpi=150)
    plt.close()
    print(f"[PT] → {OUT_PNG.name}")

    summary = {
        "n_particles": int(N_PART),
        "n_hits": int(len(hit_pos)),
        "particle_diameter_m": DP_M,
        "Vp_stats": {"min": float(hit_Vp.min()), "max": float(hit_Vp.max()),
                     "median": float(np.median(hit_Vp))},
        "alpha_stats": {"min": float(hit_alpha.min()), "max": float(hit_alpha.max()),
                        "median": float(np.median(hit_alpha))},
        "validation": {
            "spearman_rho_same_dim": round(float(rho), 4),
            "pval": float(pval),
            "top10_iou": round(iou, 4),
            "n_valid": int(valid.sum()),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[PT] → {OUT_JSON.name}")
    print("\nStep-30 完成。\n")
    return summary


if __name__ == "__main__":
    run()
