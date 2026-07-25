"""
step15_geodesic_gpr.py
======================
升级版 GPR：测地距离核（替换欧氏 RBF）

方法
----
1. 从原始 STL 重载面片索引（与 step10 顶点去重顺序一致）
2. 用网格边构建稀疏图（边权 = 欧氏长度），Dijkstra 近似测地距离
3. 自定义 GeodesicGPR：手写 Cholesky 推断，核 = σ_f² × exp(-d²/(2l²))
4. 分层采样 N_IND 个训练点，计算 (N_IND, N_total) 距离矩阵
5. 输出 vertices_full_geo.npz，字段与 step11 输出兼容

与 step11 的对比
---------------
- 欧氏 RBF 把水斗两侧在 3D 空间中距离近但曲面上距离远的点视为相关
- 测地核沿曲面度量，更符合磨蚀空间相关性的物理假设
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import spearmanr

from step10_unified_pipeline import load_stl, OUTPUT_DIR

WORKSPACE = Path(__file__).parent
STL_FULL  = OUTPUT_DIR / "dec_full_runner.stl"
NPZ_FULL  = OUTPUT_DIR / "vertices_full.npz"
OUT_NPZ   = OUTPUT_DIR / "vertices_full_geo.npz"
OUT_JSON  = OUTPUT_DIR / "geo_gpr_summary.json"

# ---------------------------------------------------------------------------
# 超参数
# ---------------------------------------------------------------------------
N_IND:       int   = 2000    # 训练点数（测地距离计算量大，比 step11 略少）
RANDOM_SEED: int   = 42
W_OKA, W_FINNIE, W_TABAKOFF = 0.60, 0.25, 0.15
LOG_CORR_MIN, LOG_CORR_MAX  = -1.5, 1.5


# ---------------------------------------------------------------------------
# 1. 重载面片索引（与 step10 顶点顺序对齐）
# ---------------------------------------------------------------------------

def load_faces_aligned(stl_path: Path, xyz_ref: np.ndarray) -> np.ndarray:
    """
    重新加载 STL，用 KDTree 把重载网格顶点映射到 NPZ 顶点索引，
    返回以 NPZ 顶点顺序为准的面片索引。

    对 max_faces 截断和 np.unique 排序差异免疫：只依赖空间坐标匹配，
    不依赖复现完全相同的去重顺序。
    """
    print("[GEO] 重载 STL 面片索引 ...", flush=True)
    # 用 trimesh 加载（与 step10 一致，process=True 自动修复法向/去重）
    mesh = load_stl(stl_path, max_faces=5_000_000)
    xyz_new = mesh["xyz"]         # 重载顶点
    faces_new = mesh["faces"]     # 引用 xyz_new 索引

    # KDTree：把 xyz_new 每个点映射到最近的 xyz_ref 索引
    tree = cKDTree(xyz_ref)
    dist, ref_idx = tree.query(xyz_new, k=1)   # ref_idx: xyz_new → xyz_ref
    max_d = float(dist.max())
    print(f"[GEO] 顶点映射最大距离: {max_d:.4f} mm  "
          f"(中位 {np.median(dist):.2e})", flush=True)

    # 重映射面片到 NPZ 索引空间
    faces_ref = ref_idx[faces_new]   # (M, 3)

    # 剔除退化面（重映射后三顶点有重合的）
    valid = (faces_ref[:,0] != faces_ref[:,1]) & \
            (faces_ref[:,1] != faces_ref[:,2]) & \
            (faces_ref[:,0] != faces_ref[:,2])
    faces_ref = faces_ref[valid]
    print(f"[GEO] 面片数: {len(faces_ref):,}（剔除退化面 {np.sum(~valid):,}）", flush=True)
    return faces_ref


# ---------------------------------------------------------------------------
# 2. 构建网格图 + Dijkstra 测地距离
# ---------------------------------------------------------------------------

def build_mesh_graph(xyz: np.ndarray, faces: np.ndarray) -> csr_matrix:
    """从面片边构建稀疏无向图，边权为欧氏长度。"""
    N = xyz.shape[0]
    rows, cols, data = [], [], []

    for i, j in [(0,1),(1,2),(0,2)]:
        a, b = faces[:, i], faces[:, j]
        d = np.linalg.norm(xyz[a] - xyz[b], axis=1)
        rows.extend(a); cols.extend(b); data.extend(d)
        rows.extend(b); cols.extend(a); data.extend(d)

    graph = csr_matrix((data, (rows, cols)), shape=(N, N))
    return graph


def compute_geodesic_distances(
    graph: csr_matrix,
    source_idx: np.ndarray,
) -> np.ndarray:
    """
    从 source_idx 中每个点出发做 Dijkstra，返回 (|source|, N) 距离矩阵。
    未连通点距离设为图中最大有限距离的 2 倍（避免 inf 破坏核矩阵）。
    """
    print(f"[GEO] Dijkstra from {len(source_idx)} sources ...", flush=True)
    t0 = time.time()
    dist = dijkstra(graph, indices=source_idx, directed=False)   # (S, N)
    t1 = time.time()
    print(f"[GEO] Dijkstra 完成，耗时 {t1-t0:.1f} s", flush=True)

    # 处理 inf（孤立顶点）
    finite_max = dist[np.isfinite(dist)].max() if np.isfinite(dist).any() else 1.0
    dist = np.where(np.isfinite(dist), dist, 2.0 * finite_max)
    return dist.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. 手写 GeodesicGPR（核矩阵基于预计算距离）
# ---------------------------------------------------------------------------

class GeodesicGPR:
    """
    零均值 GP 回归，核 k(i,j) = σ_f² × exp(-d_geo(i,j)² / (2l²)) + σ_n²δ_ij

    参数（对数空间优化）
    -------------------
    theta = [log(σ_f), log(l), log(σ_n)]
    """

    def __init__(self) -> None:
        self.theta_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None   # K⁻¹ y
        self.L_:     np.ndarray | None = None
        self.D_train_train_: np.ndarray | None = None   # (S,S) 训练集内距离

    # ------------------------------------------------------------------
    def _kernel(self, D: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """D: 距离矩阵，theta: [log σ_f, log l, log σ_n]"""
        sf2 = np.exp(2 * theta[0])
        l   = np.exp(theta[1])
        return sf2 * np.exp(-0.5 * (D / l) ** 2)

    def _lml(self, theta: np.ndarray, D_tt: np.ndarray, y: np.ndarray) -> float:
        """负对数边际似然（用于超参数优化）。自适应 jitter 保证正定。"""
        sn2 = np.exp(2 * theta[2])
        K   = self._kernel(D_tt, theta)
        n   = len(y)
        jitter = sn2 + 1e-6
        for _ in range(6):
            try:
                c, low = cho_factor(K + jitter * np.eye(n), lower=True)
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        else:
            return 1e10
        alpha = cho_solve((c, low), y)
        lml   = -0.5 * y @ alpha - np.sum(np.log(np.diag(c))) - 0.5 * n * np.log(2 * np.pi)
        return -float(lml)

    def fit(
        self,
        D_train_train: np.ndarray,   # (S, S)
        y_train: np.ndarray,         # (S,)
        n_restarts: int = 3,
    ) -> "GeodesicGPR":
        self.D_train_train_ = D_train_train.astype(np.float64)
        y = y_train.astype(np.float64)
        S = len(y)

        # 初始超参数：σ_f=1, l=中位距离, σ_n=0.1
        l0 = np.median(D_train_train[D_train_train > 0]) if (D_train_train > 0).any() else 1.0
        theta0 = np.array([0.0, np.log(l0), np.log(0.1)])

        best_lml, best_theta = np.inf, theta0
        rng = np.random.default_rng(RANDOM_SEED)

        for i in range(n_restarts + 1):
            t = theta0 if i == 0 else rng.normal(theta0, 0.5)
            res = minimize(
                self._lml, t, args=(self.D_train_train_, y),
                method="L-BFGS-B",
                bounds=[(-3, 3), (np.log(l0*0.05), np.log(l0*20)), (-5, 1)],
                options={"maxiter": 200},
            )
            if res.fun < best_lml:
                best_lml, best_theta = res.fun, res.x

        self.theta_ = best_theta
        sn2 = np.exp(2 * best_theta[2])
        K   = self._kernel(self.D_train_train_, best_theta)
        # 自适应 jitter：从 1e-6 逐步增大直到 Cholesky 成功
        jitter = sn2 + 1e-6
        for _ in range(8):
            try:
                self.L_ = cho_factor(K + jitter * np.eye(S), lower=True)
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        self.alpha_ = cho_solve(self.L_, y)

        print(f"[GEO] GPR 超参数: σ_f={np.exp(best_theta[0]):.3f}  "
              f"l={np.exp(best_theta[1]):.2f} mm  "
              f"σ_n={np.exp(best_theta[2]):.4f}  "
              f"LML={-best_lml:.2f}", flush=True)
        return self

    def predict(
        self,
        D_test_train: np.ndarray,   # (B, S) 测试点到训练点的距离
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 (均值, 标准差)，批量处理。"""
        Ks = self._kernel(D_test_train.astype(np.float64), self.theta_)   # (B, S)
        mu = Ks @ self.alpha_                                               # (B,)

        # 方差：var = k(x,x) - Ks K⁻¹ Ks^T（对角线）
        v  = cho_solve(self.L_, Ks.T)   # (S, B)
        sf2 = np.exp(2 * self.theta_[0])
        var = sf2 - np.einsum("ij,ji->i", Ks, v)
        var = np.clip(var, 0.0, None)
        return mu, np.sqrt(var)


# ---------------------------------------------------------------------------
# 4. 标签构建（与 step11 相同策略）
# ---------------------------------------------------------------------------

def percentile_rank(arr: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(arr, method="average") / len(arr)


def build_log_target(E_oka, E_finnie, E_tabakoff) -> np.ndarray:
    eps = 1e-6
    ensemble = W_OKA*percentile_rank(E_oka) + W_FINNIE*percentile_rank(E_finnie) + W_TABAKOFF*percentile_rank(E_tabakoff)
    oka_rank  = percentile_rank(E_oka)
    log_t = np.log(np.clip(ensemble, eps, None) / np.clip(oka_rank, eps, None))
    return np.clip(log_t, LOG_CORR_MIN, LOG_CORR_MAX)


def stratified_sample(risk_level, n_total, rng):
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    idx_by_lv = [np.where(risk_level == lv)[0] for lv in range(4)]
    counts = weights * np.array([len(i) for i in idx_by_lv], dtype=float)
    counts = counts / counts.sum() * n_total
    counts = np.clip(counts.astype(int), 1, None)
    counts[-1] += n_total - counts.sum()
    sel = []
    for idx, cnt in zip(idx_by_lv, counts):
        if len(idx): sel.append(rng.choice(idx, min(cnt, len(idx)), replace=False))
    return np.concatenate(sel)


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------

def run_geodesic_gpr(n_ind: int = N_IND) -> None:
    print("=" * 60)
    print("AEGIS Step-15 测地距离核 GPR")
    print("=" * 60)
    rng = np.random.default_rng(RANDOM_SEED)

    # 加载数据
    data       = np.load(str(NPZ_FULL))
    xyz        = data["xyz"]
    E_oka      = data["E_oka"].astype(np.float64)
    E_finnie   = data["E_finnie"].astype(np.float64)
    E_tabakoff = data["E_tabakoff"].astype(np.float64)
    risk_level = data["risk_level"]
    N = xyz.shape[0]
    print(f"[GEO] 顶点数: {N:,}", flush=True)

    # 面片索引
    faces = load_faces_aligned(STL_FULL, xyz)

    # 构建网格图
    print("[GEO] 构建网格图 ...", flush=True)
    t0 = time.time()
    graph = build_mesh_graph(xyz, faces)
    print(f"[GEO] 图构建完成 {time.time()-t0:.1f} s  边数: {graph.nnz//2:,}", flush=True)

    # 标签
    log_target = build_log_target(E_oka, E_finnie, E_tabakoff)

    # 分层采样训练点
    n_ind = min(n_ind, N)
    train_idx = stratified_sample(risk_level, n_ind, rng)
    print(f"[GEO] 训练点: {len(train_idx)}  风险分布: "
          + "  ".join(f"L{lv}:{np.sum(risk_level[train_idx]==lv)}" for lv in range(4)),
          flush=True)

    # 测地距离：训练点 → 全体顶点
    dist_train_all = compute_geodesic_distances(graph, train_idx)   # (S, N) float32

    # 归一化：除以训练集内距离的中位数，使长度尺度 l 在 O(1) 量级
    d_median = float(np.median(dist_train_all[dist_train_all > 0]))
    dist_train_all /= (d_median + 1e-12)
    print(f"[GEO] 测地距离中位数（归一化前）: {d_median:.1f} mm", flush=True)

    D_tt = dist_train_all[:, train_idx].astype(np.float64)          # (S, S)

    # 训练 GPR
    print("[GEO] 训练 GeodesicGPR ...", flush=True)
    t0 = time.time()
    gpr = GeodesicGPR()
    gpr.fit(D_tt, log_target[train_idx], n_restarts=3)
    print(f"[GEO] 训练耗时 {time.time()-t0:.1f} s", flush=True)

    # 全量预测（分批）
    BATCH = 5_000
    log_corr = np.zeros(N, dtype=np.float32)
    sigma    = np.zeros(N, dtype=np.float32)
    print(f"[GEO] 全量预测 {N:,} 顶点 ...", flush=True)
    t0 = time.time()
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        D_bt = dist_train_all[:, start:end].T.astype(np.float64)   # (B, S)
        mu, std = gpr.predict(D_bt)
        log_corr[start:end] = mu.astype(np.float32)
        sigma[start:end]    = std.astype(np.float32)
        if (start // BATCH) % 20 == 0:
            print(f"  {100*end/N:.0f}%", flush=True)
    print(f"[GEO] 预测耗时 {time.time()-t0:.1f} s", flush=True)

    log_corr = np.clip(log_corr, LOG_CORR_MIN, LOG_CORR_MAX)
    correction = np.exp(log_corr)
    E_final    = (E_oka * correction).astype(np.float32)
    sigma_E    = (E_final * sigma).astype(np.float32)
    E_upper_95 = (E_oka * np.exp(log_corr + 2*sigma)).astype(np.float32)
    E_lower_95 = np.clip(E_oka * np.exp(log_corr - 2*sigma), 0, None).astype(np.float32)
    E_final_norm = (E_final / (E_final.max() + 1e-30)).astype(np.float32)

    # 保留集验证
    train_set   = set(train_idx.tolist())
    holdout_idx = np.array([i for i in range(N) if i not in train_set])
    rho_ho, _   = spearmanr(log_corr[holdout_idx], log_target[holdout_idx])
    rho_oka, _  = spearmanr(E_final, E_oka)
    print(f"[GEO] 保留集 Spearman ρ={rho_ho:.4f}  E_final vs Oka ρ={rho_oka:.4f}", flush=True)

    # 与 step11 欧氏版对比（若存在）
    geo_vs_rbf = {}
    rbf_npz = OUTPUT_DIR / "vertices_full_gpr.npz"
    if rbf_npz.exists():
        rbf = np.load(str(rbf_npz))
        rho_cross, _ = spearmanr(E_final, rbf["E_final"])
        geo_vs_rbf["spearman_vs_rbf"] = float(rho_cross)
        # 平滑性：相邻顶点侵蚀率差异（用面片边）
        e0, e1 = E_final[faces[:,0]], E_final[faces[:,1]]
        smooth_geo = float(np.mean(np.abs(e0 - e1)))
        e0r, e1r = rbf["E_final"][faces[:,0]], rbf["E_final"][faces[:,1]]
        smooth_rbf = float(np.mean(np.abs(e0r - e1r)))
        geo_vs_rbf["edge_smoothness_geo"] = smooth_geo
        geo_vs_rbf["edge_smoothness_rbf"] = smooth_rbf
        print(f"[GEO] 与欧氏 RBF 版 Spearman ρ={rho_cross:.4f}  "
              f"边平滑性: geo={smooth_geo:.4f}  rbf={smooth_rbf:.4f}", flush=True)

    # 保存
    np.savez_compressed(
        str(OUT_NPZ),
        xyz=xyz, normals=data["normals"], Vp=data["Vp"], alpha=data["alpha"],
        E_oka=E_oka.astype(np.float32), E_finnie=E_finnie.astype(np.float32),
        E_tabakoff=E_tabakoff.astype(np.float32),
        E_oka_norm=data["E_oka_norm"], erosion_type=data["erosion_type"],
        risk_level=risk_level,
        log_corr_pred=log_corr, correction_factor=correction.astype(np.float32),
        E_final=E_final, E_final_norm=E_final_norm,
        sigma_E=sigma_E, E_upper_95=E_upper_95, E_lower_95=E_lower_95,
    )
    print(f"[GEO] → {OUT_NPZ.name}  ({OUT_NPZ.stat().st_size//1024} KB)", flush=True)

    summary = {
        "kernel": "geodesic_RBF (mesh-Dijkstra)",
        "n_training": int(len(train_idx)),
        "hyperparams": {
            "sigma_f": float(np.exp(gpr.theta_[0])),
            "length_scale_mm": float(np.exp(gpr.theta_[1])),
            "sigma_n": float(np.exp(gpr.theta_[2])),
        },
        "validation": {
            "holdout_spearman": float(rho_ho),
            "E_final_vs_Oka_spearman": float(rho_oka),
        },
        "geo_vs_rbf": geo_vs_rbf,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[GEO] → {OUT_JSON.name}")
    print("Step-15 测地距离核 GPR 完成。\n")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_IND
    run_geodesic_gpr(n)
