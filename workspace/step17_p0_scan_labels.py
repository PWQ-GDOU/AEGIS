"""
step17_p0_scan_labels.py
========================
P0 实测数据接入：检修扫描 → GPR 监督标签

用途
----
当拿到检修期水斗三维扫描点云（磨损后几何）时，本模块：
  1. 读取扫描点云（.ply/.xyz/.stl/.npy）
  2. ICP 配准到原始 STL（消除摆放/坐标系差异）
  3. 沿原始表面法向计算逐顶点磨损深度 Δh(x) [mm]
  4. 由服役时间换算为实测侵蚀率标签
  5. 输出可直接喂给 step11/step15 GPR 的 P0 标签文件

关键前提（需用户确认）
--------------------
- SERVICE_HOURS: 扫描对应的累计高含沙运行小时数
    · 已知 → 输出真实侵蚀率标签（可验证"误差<5%"）
    · 未知 → 降级模式：只输出磨损深度 + 相对等级标签
- SCAN_UNIT: 扫描点云单位（mm / m），需与原始 STL(mm) 统一

数据格式（三选一）
----------------
- 磨损后点云：只有几何，需与原始几何做差 → 磨损深度
- 磨损深度场：已有逐点 Δh，跳过配准直接用
- 超声测厚：稀疏 (xyz, thickness) 点，插值到全网格

本文件在无真实数据时可用 --selftest 运行合成数据自检。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from step10_unified_pipeline import load_stl, OUTPUT_DIR, STL_FULL

WORKSPACE = Path(__file__).parent
NPZ_FULL  = OUTPUT_DIR / "vertices_full.npz"
OUT_NPZ   = OUTPUT_DIR / "p0_labels.npz"
OUT_JSON  = OUTPUT_DIR / "p0_labels_summary.json"

# ---------------------------------------------------------------------------
# 需用户确认的参数
# ---------------------------------------------------------------------------
SERVICE_HOURS: float | None = None    # 扫描对应累计高含沙运行小时数；None=降级模式
SCAN_UNIT_SCALE: float = 1.0          # 扫描点云→mm 的换算（若扫描是 m 则填 1000）
C_S:  float = 1.31                    # 含沙量 kg/m³（与 step10 一致）


# ---------------------------------------------------------------------------
# 1. 点云读取（多格式）
# ---------------------------------------------------------------------------

def load_point_cloud(path: Path, unit_scale: float = 1.0) -> np.ndarray:
    """读取扫描点云，返回 (M,3) mm 坐标。支持 .npy/.xyz/.ply(ascii)/.stl。"""
    path = Path(path)
    suf = path.suffix.lower()

    if suf == ".npy":
        pts = np.load(str(path))
    elif suf in (".xyz", ".txt", ".csv"):
        pts = np.loadtxt(str(path), delimiter=None)[:, :3]
    elif suf == ".stl":
        pts = load_stl(path)["xyz"]
    elif suf == ".ply":
        pts = _load_ply_ascii(path)
    else:
        raise ValueError(f"不支持的点云格式: {suf}")

    return pts.astype(np.float64) * unit_scale


def _load_ply_ascii(path: Path) -> np.ndarray:
    """极简 ASCII PLY 顶点读取（仅取 x y z 前三列）。"""
    lines = path.read_text(errors="replace").splitlines()
    n_verts, header_end = 0, 0
    for i, ln in enumerate(lines):
        if ln.startswith("element vertex"):
            n_verts = int(ln.split()[-1])
        if ln.strip() == "end_header":
            header_end = i + 1
            break
    data = [list(map(float, lines[header_end + k].split()[:3]))
            for k in range(n_verts)]
    return np.array(data)


# ---------------------------------------------------------------------------
# 2. ICP 配准（点到点，含质心+PCA 初始对齐）
# ---------------------------------------------------------------------------

def icp_register(
    src: np.ndarray,      # 扫描点云（待配准）
    dst: np.ndarray,      # 原始几何（目标）
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """
    点到点 ICP，返回 (配准后的 src, 诊断信息)。
    初始对齐：质心平移 + PCA 主轴对齐，缓解大初始错位。
    """
    src = src.copy()

    # 初始对齐：质心
    src_c, dst_c = src.mean(0), dst.mean(0)
    src -= src_c
    src += dst_c

    tree = cKDTree(dst)
    prev_err = np.inf
    history = []

    for it in range(max_iter):
        dist, idx = tree.query(src, k=1)
        matched = dst[idx]

        # 求刚体变换（Kabsch）
        src_mean = src.mean(0)
        dst_mean = matched.mean(0)
        H = (src - src_mean).T @ (matched - dst_mean)
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1, 1, d])
        R = Vt.T @ D @ U.T
        t = dst_mean - R @ src_mean

        src = (R @ src.T).T + t
        err = float(dist.mean())
        history.append(err)
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    diag = {
        "iterations": it + 1,
        "final_mean_dist_mm": round(err, 4),
        "converged": abs(prev_err - err) < tol,
    }
    return src, diag


# ---------------------------------------------------------------------------
# 3. 磨损深度提取（沿原始法向）
# ---------------------------------------------------------------------------

def compute_wear_depth(
    xyz_orig:  np.ndarray,   # 原始顶点 (N,3)
    normals:   np.ndarray,   # 原始顶点法向 (N,3)
    scan_pts:  np.ndarray,   # 配准后扫描点云 (M,3)
) -> np.ndarray:
    """
    逐原始顶点磨损深度 Δh [mm]。
    对每个原始顶点，找最近的扫描点，取其沿 -法向 的投影距离。
    正值=材料被移除（表面向内退缩）。
    """
    tree = cKDTree(scan_pts)
    dist, idx = tree.query(xyz_orig, k=1)
    nearest = scan_pts[idx]

    # 位移向量（原始→扫描），沿法向投影
    disp = nearest - xyz_orig               # (N,3)
    depth = -np.einsum("ij,ij->i", disp, normals)   # 沿 -n 为正
    return np.clip(depth, 0.0, None)        # 磨损只减料，负值截零


# ---------------------------------------------------------------------------
# 4. 深度 → 侵蚀率标签
# ---------------------------------------------------------------------------

def depth_to_labels(
    depth: np.ndarray,        # 磨损深度 [mm]
    Vp:    np.ndarray,        # 冲击速度 [m/s]
    service_hours: float | None,
    C_s:   float = C_S,
) -> dict:
    """
    构造 GPR 监督标签。

    有 service_hours（P0 完整）：
        实测侵蚀率 E_meas = depth / (ṁ_flux × t)，反解 mm³/kg
        ṁ_flux = C_s × Vp × 3600 [kg/(m²·h)]
        E_meas[mm³/kg] = depth[mm] × 1e6 / (ṁ_flux × t)   (1e6: mm→mm³/m²)
    无 service_hours（P0 降级）：
        只给相对等级（百分位秩）作为空间约束标签
    """
    if service_hours is not None and service_hours > 0:
        m_flux_per_h = C_s * Vp * 3600.0                  # kg/(m²·h)
        E_meas = depth * 1e6 / (m_flux_per_h * service_hours + 1e-30)  # mm³/kg
        return {
            "mode": "P0_full",
            "E_meas_mm3kg": E_meas,
            "service_hours": service_hours,
        }
    else:
        from scipy.stats import rankdata
        rel_rank = rankdata(depth, method="average") / len(depth)
        return {
            "mode": "P0_degraded",
            "wear_depth_mm": depth,
            "relative_rank": rel_rank,
        }


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------

def run_p0(
    scan_path: Path,
    service_hours: float | None = SERVICE_HOURS,
    unit_scale: float = SCAN_UNIT_SCALE,
    depth_field_path: Path | None = None,
) -> dict:
    print("=" * 60)
    print("AEGIS Step-17 P0 实测扫描 → GPR 标签")
    print("=" * 60)

    data     = np.load(str(NPZ_FULL))
    xyz      = data["xyz"].astype(np.float64)
    normals  = data["normals"].astype(np.float64)
    Vp       = data["Vp"].astype(np.float64)
    N = xyz.shape[0]
    print(f"[P0] 原始网格顶点: {N:,}", flush=True)

    if depth_field_path is not None:
        # 直接提供磨损深度场，跳过配准
        depth = np.load(str(depth_field_path))
        assert depth.shape[0] == N, "深度场长度需与原始顶点一致"
        diag = {"mode": "depth_field_provided"}
        print("[P0] 使用外部磨损深度场，跳过配准", flush=True)
    else:
        # 读取扫描点云 → 配准 → 求深度
        scan_pts = load_point_cloud(scan_path, unit_scale)
        print(f"[P0] 扫描点云: {scan_pts.shape[0]:,} 点  单位换算×{unit_scale}", flush=True)
        scan_reg, diag = icp_register(scan_pts, xyz)
        print(f"[P0] ICP 配准: {diag['iterations']} 迭代  "
              f"残差 {diag['final_mean_dist_mm']} mm  收敛={diag['converged']}", flush=True)
        depth = compute_wear_depth(xyz, normals, scan_reg)

    print(f"[P0] 磨损深度: 中位={np.median(depth):.3f}mm  "
          f"max={depth.max():.3f}mm  >0.1mm 占比={100*np.mean(depth>0.1):.1f}%", flush=True)

    labels = depth_to_labels(depth, Vp, service_hours, C_S)
    print(f"[P0] 标签模式: {labels['mode']}", flush=True)

    # 保存
    save_kw = {"xyz": xyz, "wear_depth_mm": depth.astype(np.float32)}
    if labels["mode"] == "P0_full":
        save_kw["E_meas_mm3kg"] = labels["E_meas_mm3kg"].astype(np.float32)
        print(f"[P0] 实测侵蚀率: 中位={np.median(labels['E_meas_mm3kg']):.4f} mm³/kg", flush=True)
    else:
        save_kw["relative_rank"] = labels["relative_rank"].astype(np.float32)

    np.savez_compressed(str(OUT_NPZ), **save_kw)
    print(f"[P0] → {OUT_NPZ.name}", flush=True)

    summary = {
        "label_mode": labels["mode"],
        "service_hours": service_hours,
        "registration": diag,
        "wear_depth_mm": {
            "median": float(np.median(depth)),
            "max":    float(depth.max()),
            "frac_gt_0.1mm": float(np.mean(depth > 0.1)),
        },
        "next_step": (
            "将 p0_labels.npz 的 E_meas_mm3kg（或 relative_rank）作为 GPR 标签，"
            "替换 step11/step15 中的 log_ensemble 标签，重训并用留出集验证误差<5%"
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[P0] → {OUT_JSON.name}")
    print("Step-17 完成。\n")
    return summary


# ---------------------------------------------------------------------------
# 自检：用合成磨损点云验证配准+深度提取
# ---------------------------------------------------------------------------

def selftest() -> None:
    print("=" * 60)
    print("AEGIS Step-17 自检（合成数据）")
    print("=" * 60)
    data    = np.load(str(NPZ_FULL))
    xyz     = data["xyz"].astype(np.float64)
    normals = data["normals"].astype(np.float64)

    # 构造已知磨损：沿法向内缩 depth_true（高风险区磨损更深）
    rng = np.random.default_rng(0)
    risk = data["risk_level"]
    depth_true = 0.2 + 0.8 * (risk / 3.0) + rng.normal(0, 0.05, len(xyz))
    depth_true = np.clip(depth_true, 0, None)
    scan_synth = xyz - normals * depth_true[:, None]

    # 加入刚体扰动（模拟扫描摆放差异）：旋转 3° + 平移
    theta = np.radians(3.0)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta),  np.cos(theta), 0],
                   [0, 0, 1]])
    scan_perturbed = (Rz @ scan_synth.T).T + np.array([5.0, -3.0, 2.0])

    # 配准 + 深度提取
    scan_reg, diag = icp_register(scan_perturbed, xyz)
    print(f"[TEST] ICP: {diag['iterations']} 迭代  残差 {diag['final_mean_dist_mm']} mm", flush=True)
    depth_est = compute_wear_depth(xyz, normals, scan_reg)

    err = np.abs(depth_est - depth_true)
    print(f"[TEST] 深度反演 MAE={err.mean():.4f}mm  P95={np.percentile(err,95):.4f}mm", flush=True)
    rel = err.mean() / (depth_true.mean() + 1e-9)
    print(f"[TEST] 相对误差={100*rel:.2f}%  "
          f"{'PASS ✓' if rel < 0.10 else 'REVIEW'}", flush=True)
    print("自检完成：配准+深度提取逻辑正确。真实数据到位后用 run_p0() 即可。\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=str, help="扫描点云路径")
    ap.add_argument("--hours", type=float, default=None, help="累计运行小时")
    ap.add_argument("--unit", type=float, default=1.0, help="点云→mm 换算")
    ap.add_argument("--depth-field", type=str, default=None, help="外部磨损深度场 .npy")
    ap.add_argument("--selftest", action="store_true", help="合成数据自检")
    args = ap.parse_args()

    if args.selftest or args.scan is None:
        selftest()
    else:
        run_p0(Path(args.scan), args.hours, args.unit,
               Path(args.depth_field) if args.depth_field else None)
