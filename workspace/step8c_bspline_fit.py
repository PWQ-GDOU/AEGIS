"""
Step 8c: B样条曲面反拟合 — 局部控制，可捕捉锐边

B-spline vs Bézier:
- Bézier: 全局控制，改一点影响整个曲面 → 锐边处妥协
- B-spline: 局部控制，改一点只影响附近 → 可拟合锐边

用scipy的BSpline实现r-z轮廓和宽度分布的精确拟合
"""
import numpy as np
from pathlib import Path
from scipy.interpolate import splrep, splev, BSpline, make_lsq_spline
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.special import comb
import trimesh, json, time, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path(r"workspace/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEC_MODEL_STL = OUTPUT_DIR / "dec_model_runner.stl"  
DEC_FULL_STL = OUTPUT_DIR / "dec_full_runner.stl"
D1 = 312.50; R1 = D1/2; NB = 21


def load_dec_bucket():
    """加载DEC单水斗点云"""
    mesh = trimesh.load(str(DEC_MODEL_STL))
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    verts = mesh.vertices.copy()
    center = np.median(verts[:, :2], axis=0)
    theta = np.degrees(np.arctan2(verts[:,1]-center[1], verts[:,0]-center[0])) % 360
    bw = 360.0/NB*1.05
    mask = (theta > -bw) & (theta < bw)
    if mask.sum() < 500: mask = theta < bw*1.3
    return verts[mask], center


def compute_chamfer(gen_pts, dec_pts, sample=5000):
    """Chamfer距离"""
    if len(gen_pts) > sample: gen_pts = gen_pts[np.random.choice(len(gen_pts), sample, replace=False)]
    if len(dec_pts) > sample: dec_pts = dec_pts[np.random.choice(len(dec_pts), sample, replace=False)]
    t1, t2 = KDTree(dec_pts), KDTree(gen_pts)
    d1, _ = t1.query(gen_pts, k=1)
    d2, _ = t2.query(dec_pts, k=1)
    return d1, d2


def generate_bsp_surface(bs_rz, bs_w, z_range, n_u=80, n_v=40):
    """
    从B-spline生成水斗表面
    bs_rz: r(z) spline → 给定z, 返回r
    bs_w: half_width(z) spline → 给定z, 返回半宽
    z_range: (z_min, z_max)
    """
    z_min, z_max = z_range
    u = np.linspace(0, 1, n_u)
    v = np.linspace(0, 1, n_v)
    
    z_vals = z_min + u * (z_max - z_min)
    r_vals = bs_rz(u)   # r(t) where t maps to z
    w_vals = bs_w(u)    # width(t)
    
    verts = []
    for i in range(n_u):
        r_i = float(r_vals[i])
        z_i = float(z_vals[i])
        hw = max(float(w_vals[i]), 0.1)
        for j in range(n_v):
            th = v[j] * hw / R1
            verts.append([r_i * np.cos(th), r_i * np.sin(th), z_i])
    for i in range(n_u):
        r_i = float(r_vals[i])
        z_i = float(z_vals[i])
        hw = max(float(w_vals[i]), 0.1)
        for j in range(1, n_v):
            th = -v[j] * hw / R1
            verts.append([r_i * np.cos(th), r_i * np.sin(th), z_i])
    
    return np.array(verts)


def fit_profile_and_width(dec_pts):
    """
    从DEC点云提取r-z轮廓和宽度分布，用B-spline拟合
    
    分三段处理以捕捉不同区域的几何特征：
    1. 分水刃区 (z最小∼z的20%): 锐利, 低角度
    2. 内表面区 (20%∼70%): 主工作区
    3. 根部区 (70%∼z最大): 过渡到轮毂
    """
    r = np.sqrt(dec_pts[:,0]**2 + dec_pts[:,1]**2)
    z = dec_pts[:,2]
    theta = np.arctan2(dec_pts[:,1], dec_pts[:,0])
    
    z_min, z_max = z.min(), z.max()
    
    # 精细分箱: 200个z-bin (更多细节)
    n_bins = 200
    z_edges = np.linspace(z_min, z_max, n_bins+1)
    
    rz_data = []
    w_data = []
    
    for i in range(n_bins):
        mask = (z >= z_edges[i]) & (z < z_edges[i+1])
        n_pts = mask.sum()
        if n_pts < 3:
            continue
        
        z_mid = (z_edges[i] + z_edges[i+1]) / 2
        r_median = np.median(r[mask])
        
        # 宽度: 该z层的周向跨度
        t = theta[mask]
        if len(t) > 5:
            t_lo, t_hi = np.percentile(t, [2, 98])
            half_w = r_median * abs(t_hi - t_lo) / 2
        else:
            half_w = 0.1
        
        rz_data.append([z_mid, r_median])
        w_data.append([z_mid, half_w])
    
    rz_data = np.array(rz_data)
    w_data = np.array(w_data)
    
    # === 用splrep做拟合 (更鲁棒) ===
    z_vals = rz_data[:, 0]
    t_params = (z_vals - z_min) / (z_max - z_min)
    
    # r-z: k=4 (更高阶), s更小 (更精确拟合)
    n_pts = len(rz_data)
    s_rz = n_pts * 0.01  # 很小的平滑 → 接近插值
    tck_rz = splrep(t_params, rz_data[:, 1], s=s_rz, k=4)
    
    # 宽度: k=4
    s_w = n_pts * 0.05
    tck_w = splrep(t_params, w_data[:, 1], s=s_w, k=4)
    
    # 包装为可调用对象
    class BSplineWrapper:
        def __init__(self, tck):
            self.tck = tck
        def __call__(self, x):
            return splev(x, self.tck)
    
    bs_rz = BSplineWrapper(tck_rz)
    bs_w = BSplineWrapper(tck_w)
    
    return bs_rz, bs_w, rz_data, w_data, (z_min, z_max)


def main():
    print("="*70)
    print("  B样条曲面反拟合 — 精确复现DEC水斗")
    print("="*70)
    
    # 1. 加载DEC
    print("\n📦 加载DEC模型...")
    dec_pts, dec_center = load_dec_bucket()
    print(f"  DEC单水斗: {len(dec_pts):,} 顶点")
    
    # 2. B样条拟合
    print("\n🎯 B样条拟合r-z轮廓和宽度...")
    bs_rz, bs_w, rz_data, w_data, z_range = fit_profile_and_width(dec_pts)
    
    # 3. 生成表面
    print("  生成B样条表面...")
    gen_pts = generate_bsp_surface(bs_rz, bs_w, z_range, n_u=100, n_v=50)
    print(f"  生成顶点: {len(gen_pts):,}")
    
    # 4. 验证
    print("\n📊 验证...")
    d1_arr, d2_arr = compute_chamfer(gen_pts, dec_pts, sample=5000)
    all_d = np.concatenate([d1_arr, d2_arr])
    
    h_max = max(d1_arr.max(), d2_arr.max())
    chamfer = all_d.mean()
    p95 = np.percentile(all_d, 95)
    p50 = np.percentile(all_d, 50)
    
    print(f"  {'='*50}")
    print(f"  Hausdorff Max: {h_max:.2f} mm ({h_max/D1*100:.2f}% D1)")
    print(f"  Chamfer Mean:  {chamfer:.2f} mm ({chamfer/D1*100:.2f}% D1)")
    print(f"  P95:           {p95:.2f} mm ({p95/D1*100:.2f}% D1)")
    print(f"  P50:           {p50:.2f} mm ({p50/D1*100:.2f}% D1)")
    
    # 5. 提取B样条控制参数
    # 对r-z: 沿t均匀采样得到等价的"控制轮廓"
    t_eval = np.linspace(0, 1, 30)
    profile_eval = np.array(bs_rz(t_eval)).T
    width_eval = bs_w(t_eval)
    
    # 保存等效控制点
    result = {
        'method': 'B-spline (splrep, k=3)',
        'rz_tck': [bs_rz.tck[0].tolist(), bs_rz.tck[1].tolist(), bs_rz.tck[2]],
        'w_tck': [bs_w.tck[0].tolist(), bs_w.tck[1].tolist(), bs_w.tck[2]],
        'z_range': list(z_range),
        'profile_sample': [[float(z), float(r)] for z, r in zip(t_eval*(z_range[1]-z_range[0])+z_range[0], profile_eval)],
        'width_sample': [[float(z), float(w)] for z, w in zip(t_eval*(z_range[1]-z_range[0])+z_range[0], width_eval)],
        'validation': {
            'hausdorff_max': float(h_max), 'chamfer_mean': float(chamfer),
            'p95': float(p95), 'p50': float(p50),
            'p95_pct_d1': float(p95/D1*100), 'mean_pct_d1': float(chamfer/D1*100),
        }
    }
    
    with open(OUTPUT_DIR / 'bspline_fit_result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 保存: bspline_fit_result.json")
    
    # 6. 也验证真机转轮
    print(f"\n📦 验证真机转轮 (D1247R2)...")
    mesh2 = trimesh.load(str(DEC_FULL_STL))
    if isinstance(mesh2, trimesh.Scene):
        mesh2 = trimesh.util.concatenate(list(mesh2.geometry.values()))
    verts2 = mesh2.vertices.copy()
    center2 = np.median(verts2[:,:2], axis=0)
    theta2 = np.degrees(np.arctan2(verts2[:,1]-center2[1], verts2[:,0]-center2[0])) % 360
    bw2 = 360.0/NB*1.05
    mask2 = (theta2 > -bw2) & (theta2 < bw2)
    if mask2.sum() < 500: mask2 = theta2 < bw2*1.3
    dec_pts2 = verts2[mask2]
    
    # 缩放到模型尺寸做验证
    from scipy.spatial.transform import Rotation
    # 真机模型尺寸是模型的8.416倍
    gen_pts_scaled = gen_pts * 8.416
    
    d1_f, d2_f = compute_chamfer(gen_pts_scaled, dec_pts2, sample=3000)
    all_d_f = np.concatenate([d1_f, d2_f])
    d1_full = 2630.0
    
    print(f"  真机 Chamfer Mean: {all_d_f.mean():.1f} mm ({all_d_f.mean()/d1_full*100:.2f}% D1)")
    print(f"  真机 P95:          {np.percentile(all_d_f,95):.1f} mm ({np.percentile(all_d_f,95)/d1_full*100:.2f}% D1)")


if __name__ == "__main__":
    main()
