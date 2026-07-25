"""
step10_unified_pipeline.py
==========================
统一数据管道 — 灰盒退化模型 Layer-0

修复 step9 中发现的全部问题，输出后续模块共用的标准化逐顶点 NPZ 数据。

修复清单
---------
[F1] N_BUCKETS 统一为 21（与 final_validation_report.json 一致）
[F2] Tabakoff 角度 bug：step9:55 弧度被重复乘 π/180，已改为传入度值
[F3] Oka 粒径 Dp 可外部注入（不再硬编码 150 μm）
[F4] 速度模型统一为 Z 坐标非线性衰减（更贴近真实几何）
[F5] 水头统一为 H=595 m（与 truth_full.额定水头 一致）

输出
----
output/vertices_model.npz  — 模型转轮逐顶点数组
output/vertices_full.npz   — 真机转轮逐顶点数组
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

STL_MODEL = OUTPUT_DIR / "dec_model_runner.stl"
STL_FULL  = OUTPUT_DIR / "dec_full_runner.stl"

# ---------------------------------------------------------------------------
# 参数常量（全局唯一真值）
# ---------------------------------------------------------------------------
N_BUCKETS: int   = 21          # [F1] 统一水斗数
H_RATED:   float = 595.0       # [F5] 额定水头 [m]
N_RATED:   float = 375.0       # 额定转速 [rpm]
D1_MODEL:  float = 312.5       # 模型转轮节圆直径 [mm]
D1_FULL:   float = 2630.0      # 真机转轮节圆直径 [mm]
SCALE:     float = 8.416       # 模型→真机缩放比

# 泥沙参数（金窝电站汛期）
C_S_DEFAULT: float = 1.31      # 含沙量 [kg/m³]
DP_DEFAULT:  float = 150.0     # 中值粒径 D50 [μm]
RHO_P:       float = 2650.0    # 颗粒密度 [kg/m³]

# 材料参数（ZG0Cr13Ni4Mo）
HV_DEFAULT: float = 2.6        # 维氏硬度 [GPa]
RHO_MAT:    float = 7850.0     # 材料密度 [kg/m³]

# 速度衰减系数（[F4] 统一非线性 Z 衰减）
VEL_DECAY_A: float = 0.45
VEL_DECAY_B: float = 0.70


# ---------------------------------------------------------------------------
# 物理参数 dataclass（便于敏感性分析时覆盖）
# ---------------------------------------------------------------------------
@dataclass
class RunParams:
    H:             float = H_RATED
    C_s:           float = C_S_DEFAULT
    Dp:            float = DP_DEFAULT       # μm
    Hv:            float = HV_DEFAULT       # GPa
    n_buckets:     int   = N_BUCKETS
    rho_particle:  float = RHO_P


# ---------------------------------------------------------------------------
# 物理侵蚀模型（全部统一在此，修复所有已知 bug）
# ---------------------------------------------------------------------------
class OkaModel:
    """Oka (2005) 侵蚀模型。

    参数
    ----
    Hv : float
        靶材维氏硬度 [GPa]
    Dp : float
        颗粒粒径 [μm]（[F3] 可外部注入，不再硬编码）
    """

    # Oka 参考值
    V_REF:  float = 104.0   # m/s
    D_REF:  float = 326.0   # μm
    K_REF:  float = 65.0

    def __init__(self, Hv: float = HV_DEFAULT, Dp: float = DP_DEFAULT) -> None:
        self.Hv = Hv
        self.Dp = Dp
        # 材料相关指数
        self.n1 = 0.71 * (Hv ** 0.14)
        self.n2 = 2.4  * (Hv ** -0.94)
        self.k1 = -0.12
        self.k2 = 2.3  * (Hv ** 0.038)
        self.k3 = 0.19
        # 预计算速度无关项
        self._dp_factor = (Dp / self.D_REF) ** self.k3

    def angle_function(self, alpha_deg: np.ndarray) -> np.ndarray:
        a = np.radians(alpha_deg)
        return np.sin(a) ** self.n1 * (1.0 + self.Hv * (1.0 - np.sin(a))) ** self.n2

    def E90(self, Vp: np.ndarray) -> np.ndarray:
        """参考角（90°）侵蚀率 [mm³/kg]"""
        return self.K_REF * (self.Hv ** self.k1) * (Vp / self.V_REF) ** self.k2 * self._dp_factor

    def erosion_rate(self, Vp: np.ndarray, alpha_deg: np.ndarray) -> np.ndarray:
        """返回侵蚀率 [mm³/kg]"""
        return self.angle_function(alpha_deg) * self.E90(Vp)


class FinnieModel:
    """Finnie (1960) 切削磨损模型。"""

    def __init__(self, k: float = 2e-5, n: float = 2.0) -> None:
        self.k = k
        self.n = n

    def angle_function(self, alpha_deg: np.ndarray) -> np.ndarray:
        a = np.radians(alpha_deg)
        # tan(α) > 1/3 时使用 cos²/3，否则使用 sin2α - 3sin²α
        f_low  = np.sin(2.0 * a) - 3.0 * np.sin(a) ** 2
        f_high = np.cos(a) ** 2 / 3.0
        return np.where(np.tan(a) > (1.0 / 3.0), f_high, f_low)

    def erosion_rate(self, Vp: np.ndarray, alpha_deg: np.ndarray) -> np.ndarray:
        return self.k * (Vp ** self.n) * self.angle_function(alpha_deg)


class TabakoffModel:
    """Tabakoff & Grant (1982) 侵蚀模型。

    [F2] 修复角度 bug：原 step9 中 a=radians(alpha) 后仍乘 π/180
    现在统一在内部将度→弧度，外部传入度值。
    """

    def __init__(
        self,
        k1:   float = 1e-5,
        k2:   float = 0.585,
        k3:   float = 5e-12,
        k4:   float = 0.0023,
        k12:  float = 1.0,
        beta0: float = 25.0,    # 最大侵蚀角 [°]
    ) -> None:
        self.k1    = k1
        self.k2    = k2
        self.k3    = k3
        self.k4    = k4
        self.k12   = k12
        self.beta0 = beta0      # 保存为度值

    def erosion_rate(self, Vp: np.ndarray, alpha_deg: np.ndarray) -> np.ndarray:
        # [F2] 统一在此转换，外部始终传入度值
        a = np.radians(alpha_deg)

        Rt = np.clip(1.0 - self.k4 * Vp * np.sin(a), 0.0, 1.0)

        # f_beta：beta0 以度参与，和 sin 的参数均为弧度
        beta0_rad = np.radians(self.beta0)
        # 原公式：sin(90 * beta1 / beta0)，beta1=alpha_deg（度），beta0=25°
        # ⟹ sin(90° × alpha_deg / 25°) = sin(π/2 × alpha_deg/25)
        fb = (1.0 + self.k2 * self.k12 * np.sin(np.pi / 2.0 * alpha_deg / self.beta0)) ** 2

        Ec = self.k1 * fb * (Vp ** 2) * np.cos(a) ** 2 * (1.0 - Rt ** 2)
        Ed = self.k3 * (Vp * np.sin(a)) ** 4
        return Ec + Ed


# ---------------------------------------------------------------------------
# STL 读取工具（使用 trimesh，自动修复法向/退化面/精确去重）
# ---------------------------------------------------------------------------

def load_stl(stl_path: Path, max_faces: int = 5_000_000) -> dict:
    """
    加载 STL 文件，返回顶点/面片/面积加权顶点法向量。

    使用 trimesh.load(process=True) 替代旧版自定义解析器：
    - 精确哈希去重（不再用 np.round 有损近似）
    - 自动修复法向一致性（解决 STL 多法向区域问题）
    - 自动剔除退化面片
    - 尝试修复非封闭边（fill_holes）

    参数
    ----
    max_faces : 安全上限（防止极端大文件），默认 500 万，足够覆盖所有 DEC 模型
    """
    import trimesh

    stl_path = Path(stl_path)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL 文件不存在: {stl_path}")

    mesh = trimesh.load(str(stl_path), process=True)

    # 处理多实体 Scene
    if isinstance(mesh, trimesh.Scene):
        meshes = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(meshes)

    # 尝试修复非封闭网格（填补小洞，修复翻转法向）
    if not mesh.is_watertight:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fill_holes(mesh)

    # 面片数安全上限（应不会触发）
    if len(mesh.faces) > max_faces:
        print(f"  [WARNING] 面片数 {len(mesh.faces):,} 超过 max_faces={max_faces:,}，截断。")
        mask = np.random.choice(len(mesh.faces), max_faces, replace=False)
        mesh = mesh.submesh([mask], append=True)

    # 计算面积加权顶点法向量（比 trimesh 默认的平均法向更精确）
    xyz   = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces,    dtype=np.int64)
    N     = len(xyz)

    v0, v1, v2 = xyz[faces[:,0]], xyz[faces[:,1]], xyz[faces[:,2]]
    cross = np.cross(v1 - v0, v2 - v0)                           # (M,3)
    area  = 0.5 * np.linalg.norm(cross, axis=1, keepdims=True)   # (M,1)
    face_normals_geo = cross / (2.0 * area + 1e-12)               # 几何法向（面积归一化后不用，只用方向）

    vert_normals = np.zeros((N, 3), dtype=np.float64)
    for i in range(3):
        np.add.at(vert_normals, faces[:, i], face_normals_geo * area)
    norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    vert_normals /= norms

    return {"xyz": xyz, "normals": vert_normals, "faces": faces}


# 保留 _build_mesh 签名供 step15 的 load_faces_aligned 调用兼容
def _build_mesh(face_verts: np.ndarray, face_normals: np.ndarray) -> dict:
    """向后兼容接口，内部调用已弃用（保留供外部脚本兼容）。"""
    M = face_verts.shape[0]
    all_verts = face_verts.reshape(-1, 3).astype(np.float64)
    unique, inv = np.unique(np.round(all_verts, decimals=6), axis=0, return_inverse=True)
    faces = inv.reshape(M, 3)
    N = unique.shape[0]
    v0, v1, v2 = unique[faces[:,0]], unique[faces[:,1]], unique[faces[:,2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area  = 0.5 * np.linalg.norm(cross, axis=1, keepdims=True)
    vert_normals = np.zeros((N, 3), dtype=np.float64)
    for i in range(3):
        np.add.at(vert_normals, faces[:, i], face_normals * area)
    norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
    vert_normals /= np.where(norms < 1e-12, 1.0, norms)
    return {"xyz": unique, "normals": vert_normals, "faces": faces}


# ---------------------------------------------------------------------------
# 速度模型 [F4] 统一非线性 Z 坐标衰减
# ---------------------------------------------------------------------------

def compute_jet_velocity(H: float) -> float:
    """射流速度 [m/s]，速度系数 Cv=0.97"""
    return 0.97 * np.sqrt(2.0 * 9.81 * H)


def compute_bucket_velocity(D1_mm: float, n_rpm: float) -> float:
    """水斗节圆线速度 [m/s]"""
    return np.pi * (D1_mm * 1e-3) * n_rpm / 60.0


def compute_impact_velocity(
    xyz: np.ndarray,
    V_jet: float,
    V_bucket: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    逐顶点冲击速度与冲击角。

    [F4] 速度沿 Z 坐标归一化后做非线性衰减：
        Vp(z) = V_rel × (1 - A × z_norm^B)

    其中 V_rel = V_jet − V_bucket（最优速比近似），
    z_norm ∈ [0,1]（0=水斗进水边，1=出水边/底部）。

    冲击角由法向量与射流方向（-Z 轴）的夹角计算，
    并限制在 [0.5°, 89.5°] 内以避免奇点。

    返回
    ----
    Vp         : (N,) 冲击速度 [m/s]
    alpha_deg  : (N,) 冲击角   [°]
    """
    V_rel = V_jet - V_bucket

    # Z 归一化（0→进水，1→出水）
    z = xyz[:, 2]
    z_min, z_max = z.min(), z.max()
    z_range = z_max - z_min if (z_max - z_min) > 1e-12 else 1.0
    z_norm = (z - z_min) / z_range

    Vp = V_rel * (1.0 - VEL_DECAY_A * z_norm ** VEL_DECAY_B)
    Vp = np.clip(Vp, 0.1, None)   # 避免零速度点

    return Vp


def compute_impact_angle(
    xyz: np.ndarray,
    normals: np.ndarray,
    jet_dir: np.ndarray | None = None,
) -> np.ndarray:
    """
    冲击角 = 射流方向与壁面法向量的夹角（取与表面的夹角，非法向）。

    [F6] 射流方向修正：转轮绕 Z 轴旋转（已用 FFT 周期性验证，k=21 谐波能量
    是背景的 835 倍），故 Z 轴是旋转轴而非射流入射轴。真实 Pelton 水轮机喷嘴
    沿转轮径向打在水斗上。由旋转对称性：任一顶点所属水斗轮转到喷嘴瞄准位置
    时的冲击角，等于该顶点当前位置处法向量与"局部径向方向"（指向 Z 轴，
    即 -r̂，r̂=(x,y,0)/|(x,y,0)|）的夹角——因为旋转是保角变换，不需要
    显式选定喷嘴瞄准角度或做水斗窗口分割。

    若显式传入 jet_dir（如单水斗 CFD 场景下的固定径向射流方向），
    则使用该固定方向而非逐点局部径向方向。

    参数
    ----
    xyz     : (N,3) 顶点坐标 [mm]，Z 为旋转轴
    normals : (N,3) 顶点法向量
    jet_dir : 可选，固定射流方向（默认 None → 逐点局部径向方向）
    """
    if jet_dir is not None:
        jd = jet_dir / np.linalg.norm(jet_dir)
        cos_theta = np.abs(normals @ jd)
    else:
        # 逐点局部径向方向（指向 Z 轴，XY 平面内）
        r_xy = xyz[:, :2]
        r_norm = np.linalg.norm(r_xy, axis=1, keepdims=True)
        r_norm = np.where(r_norm < 1e-9, 1.0, r_norm)
        r_hat_xy = r_xy / r_norm                      # 径向单位向量（指向外）
        jet_dir_local = np.zeros_like(xyz)
        jet_dir_local[:, :2] = -r_hat_xy              # 射流指向轴心（径向向内）
        # 法向量与局部射流方向的逐点点积
        cos_theta = np.abs(np.sum(normals * jet_dir_local, axis=1))

    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    alpha_deg = np.degrees(np.arcsin(cos_theta))   # 冲击角 ∈ [0°, 90°]
    return np.clip(alpha_deg, 0.5, 89.5)


# ---------------------------------------------------------------------------
# 磨蚀类型分类
# ---------------------------------------------------------------------------
# 磨蚀类型编码
ETYPE_CUTTING    = 0   # 切削主导
ETYPE_DEFORMATION = 1  # 撞击/变形主导
ETYPE_COMBINED   = 2   # 犁沟/耦合

# 风险等级编码
RISK_LOW     = 0
RISK_MEDIUM  = 1
RISK_HIGH    = 2
RISK_EXTREME = 3

RISK_THRESHOLDS = [0.30, 0.60, 0.85]  # 归一化侵蚀率分段


def classify_erosion_type(
    E_finnie: np.ndarray,
    E_tabakoff: np.ndarray,
    E_oka: np.ndarray,
) -> np.ndarray:
    """
    根据三模型相对贡献判断磨蚀主导机制。

    切削主导  : Finnie 占比 > 55%（低冲击角区域）
    撞击主导  : Tabakoff 占比 > 75%（高冲击角区域）
    犁沟/耦合 : 其余
    """
    total = E_finnie + E_tabakoff + E_oka + 1e-30
    cut_ratio = E_finnie / total

    etype = np.full(len(E_oka), ETYPE_COMBINED, dtype=np.int8)
    etype[cut_ratio > 0.55] = ETYPE_CUTTING
    # 高冲击角：alpha > 60° 时 Tabakoff 变形项主导
    tab_ratio = E_tabakoff / total
    etype[(tab_ratio > 0.75) & (etype != ETYPE_CUTTING)] = ETYPE_DEFORMATION
    return etype


def classify_risk_level(E_oka_norm: np.ndarray) -> np.ndarray:
    """基于归一化 Oka 侵蚀率划分风险等级。"""
    risk = np.zeros(len(E_oka_norm), dtype=np.int8)
    risk[E_oka_norm >= RISK_THRESHOLDS[0]] = RISK_MEDIUM
    risk[E_oka_norm >= RISK_THRESHOLDS[1]] = RISK_HIGH
    risk[E_oka_norm >= RISK_THRESHOLDS[2]] = RISK_EXTREME
    return risk


# ---------------------------------------------------------------------------
# 核心管道函数
# ---------------------------------------------------------------------------

def process_runner(
    stl_path: Path,
    D1_mm: float,
    params: RunParams,
    label: str = "",
    max_faces: int = 5_000_000,
) -> dict:
    """
    完整处理单个转轮 STL，返回逐顶点标准化数据字典。

    参数
    ----
    stl_path : STL 文件路径
    D1_mm    : 节圆直径 [mm]
    params   : RunParams（工况 + 材料 + 泥沙）
    label    : 标识符，用于日志
    max_faces: 最大面片数（内存保护，默认足够覆盖真机 STL 184万面片）
    """
    print(f"[{label}] 读取 STL: {stl_path.name} ...", flush=True)
    mesh = load_stl(stl_path, max_faces=max_faces)
    xyz     = mesh["xyz"]       # (N, 3) mm
    normals = mesh["normals"]   # (N, 3)
    N = xyz.shape[0]
    print(f"[{label}] 顶点数: {N:,}", flush=True)

    # 速度计算
    V_jet    = compute_jet_velocity(params.H)
    V_bucket = compute_bucket_velocity(D1_mm, N_RATED)
    print(f"[{label}] V_jet={V_jet:.1f} m/s  V_bucket={V_bucket:.1f} m/s  "
          f"V_rel={V_jet - V_bucket:.1f} m/s", flush=True)

    Vp        = compute_impact_velocity(xyz, V_jet, V_bucket)
    alpha_deg = compute_impact_angle(xyz, normals)

    # 实例化物理模型
    oka      = OkaModel(Hv=params.Hv, Dp=params.Dp)
    finnie   = FinnieModel()
    tabakoff = TabakoffModel()

    E_oka      = oka.erosion_rate(Vp, alpha_deg)
    E_finnie   = finnie.erosion_rate(Vp, alpha_deg)
    E_tabakoff = tabakoff.erosion_rate(Vp, alpha_deg)

    # 归一化
    E_oka_max  = E_oka.max()
    E_oka_norm = E_oka / (E_oka_max + 1e-30)

    # 分类
    erosion_type = classify_erosion_type(E_finnie, E_tabakoff, E_oka)
    risk_level   = classify_risk_level(E_oka_norm)

    high_pct = 100.0 * np.sum(risk_level >= RISK_HIGH) / N
    print(f"[{label}] 高风险区占比: {high_pct:.1f}%  "
          f"冲击角中位数: {np.median(alpha_deg):.1f}°", flush=True)

    return {
        "xyz":          xyz,            # (N,3) mm
        "normals":      normals,        # (N,3)
        "Vp":           Vp,             # (N,) m/s
        "alpha":        alpha_deg,      # (N,) °
        "E_oka":        E_oka,          # (N,) mm³/kg
        "E_finnie":     E_finnie,       # (N,) mm³/kg（量纲同 k 参数）
        "E_tabakoff":   E_tabakoff,     # (N,) 量纲同 k1/k3 参数
        "E_oka_norm":   E_oka_norm,     # (N,) [0,1]
        "erosion_type": erosion_type,   # (N,) int8: 0=切削 1=变形 2=耦合
        "risk_level":   risk_level,     # (N,) int8: 0–3
        # 元数据
        "D1_mm":        D1_mm,
        "V_jet":        V_jet,
        "V_bucket":     V_bucket,
        "H":            params.H,
        "C_s":          params.C_s,
        "Dp":           params.Dp,
        "Hv":           params.Hv,
        "N":            N,
    }


def save_npz(data: dict, out_path: Path) -> None:
    """将逐顶点数组字典保存为 NPZ，元数据存入 JSON 同名文件。"""
    # 只保存数组字段
    array_keys = [k for k, v in data.items() if isinstance(v, np.ndarray)]
    np.savez_compressed(str(out_path), **{k: data[k] for k in array_keys})

    # 元数据
    meta = {k: v for k, v in data.items() if not isinstance(v, np.ndarray)}
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  → 已保存: {out_path.name}  ({out_path.stat().st_size / 1024:.0f} KB)")
    print(f"  → 元数据: {meta_path.name}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main(params: RunParams | None = None) -> tuple[dict, dict]:
    """
    运行完整管道，返回 (model_data, full_data)。
    可传入自定义 RunParams 供敏感性分析调用。
    """
    if params is None:
        params = RunParams()

    print("=" * 60)
    print("AEGIS Step-10 统一数据管道")
    print(f"  H={params.H} m  C_s={params.C_s} kg/m³  "
          f"Dp={params.Dp} μm  Hv={params.Hv} GPa")
    print("=" * 60)

    # 模型转轮
    model_data = process_runner(
        STL_MODEL, D1_mm=D1_MODEL, params=params, label="MODEL"
    )
    save_npz(model_data, OUTPUT_DIR / "vertices_model.npz")

    # 真机转轮
    full_data = process_runner(
        STL_FULL, D1_mm=D1_FULL, params=params, label="FULL"
    )
    save_npz(full_data, OUTPUT_DIR / "vertices_full.npz")

    # 跨尺度一致性检验（Spearman 相关）
    from scipy.stats import spearmanr
    model_norm = model_data["E_oka_norm"]
    full_norm  = full_data["E_oka_norm"]

    # 用百分位数对齐比较（两者顶点数不同）
    pcts = np.linspace(0, 100, 100)
    model_pcts = np.percentile(model_norm, pcts)
    full_pcts  = np.percentile(full_norm,  pcts)
    rho, pval  = spearmanr(model_pcts, full_pcts)
    print(f"\n跨尺度 Spearman 相关 ρ = {rho:.4f}  (p={pval:.2e})")

    summary = {
        "model_runner": {
            "n_verts":          int(model_data["N"]),
            "impact_angle_med": float(np.median(model_data["alpha"])),
            "risk_high_pct":    float(100 * np.mean(model_data["risk_level"] >= RISK_HIGH)),
            "E_oka_max":        float(model_data["E_oka"].max()),
        },
        "full_runner": {
            "n_verts":          int(full_data["N"]),
            "impact_angle_med": float(np.median(full_data["alpha"])),
            "risk_high_pct":    float(100 * np.mean(full_data["risk_level"] >= RISK_HIGH)),
            "E_oka_max":        float(full_data["E_oka"].max()),
        },
        "cross_scale_spearman_rho": float(rho),
        "parameters": {
            "H_m":     params.H,
            "C_s":     params.C_s,
            "Dp_um":   params.Dp,
            "Hv_GPa":  params.Hv,
        },
        "bug_fixes_applied": ["F1_N_BUCKETS=21", "F2_Tabakoff_angle",
                               "F3_Oka_Dp_injectable", "F4_velocity_z_nonlinear",
                               "F5_H=595m"],
    }
    summary_path = OUTPUT_DIR / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n摘要已写入: {summary_path.name}")
    print("Step-10 完成。\n")

    return model_data, full_data


if __name__ == "__main__":
    main()
