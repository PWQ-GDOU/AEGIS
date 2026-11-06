"""
Step 8: 从DEC STL反拟合Bézier控制点 — 精确复现DEC几何

方法: Katsura, Xiao, Guo, Liang et al. (2025)
- 从DEC表面提取r-z轮廓和宽度分布
- 用梯度下降优化Bézier控制点
- 目标: Hausdorff < 1% D1
"""
import numpy as np
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from scipy.special import comb
import trimesh, json, time

OUTPUT_DIR = Path(r"workspace/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEC_MODEL_STL = OUTPUT_DIR / "dec_model_runner.stl"
D1_MODEL = 312.50
N_BUCKETS = 21


# ============================================================
# Bézier 工具
# ============================================================
def bernstein(i, n, t):
    return comb(n, i) * (t**i) * ((1-t)**(n-1-i)) if n-1-i >= 0 else 0


def bezier_value(ctrl, t):
    """计算Bézier曲线值"""
    ctrl = np.asarray(ctrl)
    n = len(ctrl) - 1
    t = np.atleast_1d(t)
    result = np.zeros((len(t), ctrl.shape[1]))
    for i in range(n + 1):
        b = np.array([bernstein(i, n, float(ti)) for ti in t])
        result += np.outer(b, ctrl[i])
    return result


# ============================================================
# 从DEC STL提取目标数据
# ============================================================
def load_and_extract_bucket():
    """加载DEC STL并提取单水斗的r-z轮廓和宽度分布"""
    print("加载DEC模型...")
    mesh = trimesh.load(str(DEC_MODEL_STL))
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    
    verts = mesh.vertices.copy()
    center = np.median(verts[:, :2], axis=0)
    r = np.sqrt((verts[:,0]-center[0])**2 + (verts[:,1]-center[1])**2)
    z = verts[:,2]
    theta = np.degrees(np.arctan2(verts[:,1]-center[1], verts[:,0]-center[0])) % 360
    
    # 提取单水斗
    bucket_w = 360.0 / N_BUCKETS
    bucket_mask = (theta > -bucket_w) & (theta < bucket_w)  # 水斗0附近
    
    if bucket_mask.sum() < 500:
        bucket_mask = theta < bucket_w * 1.2
    
    bucket_verts = verts[bucket_mask]
    bucket_r = r[bucket_mask]
    bucket_z = z[bucket_mask]
    bucket_theta = theta[bucket_mask]
    
    print(f"  DEC单水斗: {len(bucket_verts):,} 顶点")
    
    # === 提取 r-z 目标轮廓 ===
    z_min, z_max = bucket_z.min(), bucket_z.max()
    z_bins = np.linspace(z_min, z_max, 40)
    
    target_rz = []
    for i in range(len(z_bins)-1):
        mask = (bucket_z >= z_bins[i]) & (bucket_z < z_bins[i+1])
        if mask.sum() > 10:
            target_rz.append({
                'z': (z_bins[i] + z_bins[i+1]) / 2,
                'r_mean': float(bucket_r[mask].mean()),
                'r_median': float(np.median(bucket_r[mask])),
                'n': int(mask.sum()),
            })
    
    # === 提取 宽度目标 ===
    # 在每个z位置, 宽度 = r × theta_span
    target_width = []
    for i in range(len(z_bins)-1):
        mask = (bucket_z >= z_bins[i]) & (bucket_z < z_bins[i+1])
        if mask.sum() > 10:
            t = bucket_theta[mask]
            r_m = bucket_r[mask].mean()
            # 用P5-P95来避免离群值
            t_low, t_high = np.percentile(t, [5, 95])
            half_width = r_m * np.radians(t_high - t_low) / 2
            target_width.append({
                'z': (z_bins[i] + z_bins[i+1]) / 2,
                'half_width': float(half_width),
            })
    
    print(f"  r-z轮廓: {len(target_rz)} 点")
    print(f"  宽度分布: {len(target_width)} 点")
    
    return {
        'target_rz': target_rz,
        'target_width': target_width,
        'z_range': (z_min, z_max),
        'r_range': (bucket_r.min(), bucket_r.max()),
    }


def extract_full_bucket_surface():
    """提取DEC单水斗的完整表面点云（用于最终验证）"""
    mesh = trimesh.load(str(DEC_MODEL_STL))
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    
    verts = mesh.vertices.copy()
    center = np.median(verts[:, :2], axis=0)
    theta = np.degrees(np.arctan2(verts[:,1]-center[1], verts[:,0]-center[0])) % 360
    
    bucket_w = 360.0 / N_BUCKETS * 1.1
    mask = (theta > -bucket_w) & (theta < bucket_w)
    if mask.sum() < 500:
        mask = theta < bucket_w * 1.3
    
    return verts[mask], center


# ============================================================
# 损失函数: 我们的Bézier模型 vs DEC数据
# ============================================================
def loss_function(params, target_data):
    """
    params: 扁平化的Bézier控制点
    - ctrl_rz (5×2): r-z轮廓控制点 (5点)
    - ctrl_width (5×2): 宽度分布控制点 (5点)
    """
    # 解析参数: 5 × 2 × 2 = 20
    ctrl_rz = params[:10].reshape(5, 2)
    ctrl_width = params[10:20].reshape(5, 2)
    
    R = D1_MODEL / 2
    # P0.r = R, P0.z 也固定
    ctrl_rz[0, 0] = R
    
    loss = 0.0
    
    # === 1. r-z轮廓损失 ===
    target_rz = target_data['target_rz']
    z_target = np.array([p['z'] for p in target_rz])
    r_target = np.array([p['r_median'] for p in target_rz])
    weights = np.array([p['n'] for p in target_rz], dtype=float)
    weights /= weights.sum()
    
    # 参数化: 用z坐标映射到 t∈[0,1]
    z_all = ctrl_rz[:, 1]
    z_min, z_max = z_all.min(), z_all.max()
    
    for z_i, r_i, w in zip(z_target, r_target, weights):
        if z_min < z_i < z_max:
            t_i = (z_i - z_min) / (z_max - z_min + 1e-10)
            t_i = np.clip(t_i, 0, 1)
            r_pred = bezier_value(ctrl_rz, t_i)[0, 0]
            loss += w * (r_pred - r_i)**2
    
    # === 2. 宽度损失 ===
    target_w = target_data['target_width']
    z_w = np.array([p['z'] for p in target_w])
    w_target = np.array([p['half_width'] for p in target_w])
    n_w = np.array([p['half_width'] for p in target_w])
    w_weights = np.ones_like(w_target) / len(w_target)
    
    z_w_min, z_w_max = ctrl_width[:, 0].min(), ctrl_width[:, 0].max()
    for z_i, w_i, wt in zip(z_w, w_target, w_weights):
        if z_w_min < z_i < z_w_max:
            t_i = (z_i - z_w_min) / (z_w_max - z_w_min + 1e-10)
            t_i = np.clip(t_i, 0, 1)
            w_pred = bezier_value(ctrl_width, t_i)[0, 1]
            loss += wt * (w_pred - w_i)**2
    
    # === 3. 正则化: 平滑控制点 ===
    smooth_weight = 0.01
    for i in range(1, len(ctrl_rz)-1):
        loss += smooth_weight * np.sum((ctrl_rz[i] - (ctrl_rz[i-1] + ctrl_rz[i+1])/2)**2)
    for i in range(1, len(ctrl_width)-1):
        loss += smooth_weight * np.sum((ctrl_width[i] - (ctrl_width[i-1] + ctrl_width[i+1])/2)**2)
    
    return loss


def optimize_control_points(target_data):
    """用梯度下降优化控制点"""
    print(f"\n🎯 优化Bézier控制点...")
    
    R = D1_MODEL / 2  # 156.25
    
    # 从DEC数据获取z范围
    z_vals = np.array([p['z'] for p in target_data['target_rz']])
    z_min, z_max = z_vals.min(), z_vals.max()
    r_vals = np.array([p['r_median'] for p in target_data['target_rz']])
    r_min, r_max = r_vals.min(), r_vals.max()
    
    print(f"  DEC z范围: [{z_min:.1f}, {z_max:.1f}]")
    print(f"  DEC r范围: [{r_min:.1f}, {r_max:.1f}]")
    
    # 从DEC提取的宽度数据
    w_vals = np.array([p['half_width'] for p in target_data['target_width']])
    w_max = w_vals.max()
    print(f"  DEC 半宽范围: [{w_vals.min():.1f}, {w_max:.1f}]")
    
    # 5控制点 r-z Bézier (更好拟合复杂轮廓)
    # P0固定在(R, z_min)
    ctrl_rz_init = np.array([
        [R, z_min],                # P0: 分水刃
        [R - 10, z_min + 8],       # P1: 内凹
        [R + 10, z_min + 30],      # P2: 外扩
        [R + 5, z_min + 55],       # P3: 保持
        [R - 3, z_max],            # P4: 根部
    ])
    
    ctrl_w_init = np.array([
        [z_min, 5.0],
        [z_min + 10, 30.0],
        [z_min + 30, 45.0],
        [z_min + 55, 20.0],
        [z_max, 3.0],
    ])
    
    x0 = np.concatenate([ctrl_rz_init.flatten(), ctrl_w_init.flatten()])
    
    # 更宽的搜索范围
    margin = 30
    bounds_rz = [
        (R-30, R+50), (z_min, z_min+40),     # P1: 大范围搜索
        (R-40, R+60), (z_min+10, z_min+60),  # P2
        (R-40, R+20), (z_min+30, z_max-5),   # P3
        (R-30, R+10), (z_max-15, z_max),     # P4
    ]
    bounds_w = [
        (z_min, z_min), (0, 20),           # P0.z固定, w搜索
        (z_min+2, z_min+25), (5, 60),      # P1
        (z_min+15, z_min+50), (15, 60),    # P2
        (z_min+35, z_max-5), (5, 50),      # P3
        (z_max-10, z_max), (0, 20),        # P4
    ]
    
    # P0.r固定, P0.z固定
    all_bounds = [(R, R), (z_min, z_min)] + bounds_rz + bounds_w
    
    print("  阶段1: 全局搜索 (differential_evolution)...")
    result_de = differential_evolution(
        loss_function, all_bounds, args=(target_data,),
        maxiter=300, popsize=40, seed=42, polish=True, disp=False
    )
    print(f"  全局最优损失: {result_de.fun:.2f}")
    
    print("  阶段2: 局部精细优化 (L-BFGS-B)...")
    result_local = minimize(
        loss_function, result_de.x, args=(target_data,),
        method='L-BFGS-B', bounds=all_bounds,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )
    print(f"  局部最优损失: {result_local.fun:.2f}")
    
    best_params = result_local.x
    ctrl_rz_opt = best_params[:10].reshape(5, 2)
    ctrl_w_opt = best_params[10:20].reshape(5, 2)
    
    # 修正P0
    ctrl_rz_opt[0] = [R, z_min]
    
    return ctrl_rz_opt, ctrl_w_opt, result_local.fun


# ============================================================
# 用优化后的控制点生成面并验证
# ============================================================
def generate_optimized_surface(ctrl_rz, ctrl_width, n_u=80, n_v=50):
    """生成优化后的水斗表面"""
    u = np.linspace(0, 1, n_u)
    v_half = np.linspace(0, 1, n_v)
    
    profile_rz = bezier_value(ctrl_rz, u)
    
    # 宽度插值
    z_ctrl = ctrl_width[:, 0]
    w_ctrl = ctrl_width[:, 1]
    width_fn = interp1d(z_ctrl, w_ctrl, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    R_ref = D1_MODEL / 2
    vertices = []
    
    # 正半
    for i, ui in enumerate(u):
        r_i = float(profile_rz[i, 0])
        z_i = float(profile_rz[i, 1])
        hw = float(width_fn(z_i))
        hw = max(hw, 0.1)
        
        for j, vj in enumerate(v_half):
            theta = vj * hw / R_ref
            vertices.append([r_i * np.cos(theta), r_i * np.sin(theta), z_i])
    
    # 负半 (跳过j=0)
    for i, ui in enumerate(u):
        r_i = float(profile_rz[i, 0])
        z_i = float(profile_rz[i, 1])
        hw = float(width_fn(z_i))
        hw = max(hw, 0.1)
        
        for j in range(1, n_v):
            theta = -v_half[j] * hw / R_ref
            vertices.append([r_i * np.cos(theta), r_i * np.sin(theta), z_i])
    
    return np.array(vertices)


def compute_validation_metrics(opt_verts, dec_verts):
    """计算优化后的验证指标"""
    # 采样
    n_sample = min(5000, len(opt_verts), len(dec_verts))
    if len(opt_verts) > n_sample:
        opt_s = opt_verts[np.random.choice(len(opt_verts), n_sample, replace=False)]
    else:
        opt_s = opt_verts
    if len(dec_verts) > n_sample:
        dec_s = dec_verts[np.random.choice(len(dec_verts), n_sample, replace=False)]
    else:
        dec_s = dec_verts
    
    tree_dec = KDTree(dec_s)
    dist_o2d, _ = tree_dec.query(opt_s)
    
    tree_opt = KDTree(opt_s)
    dist_d2o, _ = tree_opt.query(dec_s)
    
    all_dist = np.concatenate([dist_o2d, dist_d2o])
    
    return {
        'hausdorff_max': float(max(dist_o2d.max(), dist_d2o.max())),
        'chamfer_mean': float(all_dist.mean()),
        'p95': float(np.percentile(all_dist, 95)),
        'p50': float(np.percentile(all_dist, 50)),
        'p95_pct_d1': float(np.percentile(all_dist, 95) / D1_MODEL * 100),
        'mean_pct_d1': float(all_dist.mean() / D1_MODEL * 100),
    }


def main():
    print("="*70)
    print("  Bézier控制点反拟合 — 精确复现DEC水斗几何")
    print("="*70)
    
    # 1. 提取DEC目标数据
    target_data = load_and_extract_bucket()
    
    # 2. 优化控制点
    start = time.time()
    ctrl_rz, ctrl_w, final_loss = optimize_control_points(target_data)
    elapsed = time.time() - start
    print(f"\n  优化耗时: {elapsed:.1f}s")
    
    print(f"\n  优化后的 r-z 控制点:")
    for i, (r, z) in enumerate(ctrl_rz):
        print(f"    P{i}: r={r:.3f}, z={z:.3f}")
    
    print(f"\n  优化后的宽度控制点:")
    for i, (z, w) in enumerate(ctrl_w):
        print(f"    P{i}: z={z:.3f}, half_w={w:.3f}")
    
    # 3. 生成优化后的表面
    print(f"\n  生成优化后的表面...")
    opt_verts = generate_optimized_surface(ctrl_rz, ctrl_w)
    print(f"  顶点: {len(opt_verts):,}")
    
    # 4. 加载DEC单水斗做验证
    dec_bucket, dec_center = extract_full_bucket_surface()
    
    # 5. 计算验证指标
    metrics = compute_validation_metrics(opt_verts, dec_bucket)
    
    print(f"\n{'='*50}")
    print(f"📊 优化后验证结果")
    print(f"{'='*50}")
    print(f"  Hausdorff Max: {metrics['hausdorff_max']:.2f} mm ({metrics['hausdorff_max']/D1_MODEL*100:.2f}% D1)")
    print(f"  Chamfer Mean:  {metrics['chamfer_mean']:.2f} mm ({metrics['mean_pct_d1']:.2f}% D1)")
    print(f"  P95:           {metrics['p95']:.2f} mm ({metrics['p95_pct_d1']:.2f}% D1)")
    print(f"  P50:           {metrics['p50']:.2f} mm")
    
    # 6. 保存优化后的控制点
    result = {
        'ctrl_rz': ctrl_rz.tolist(),
        'ctrl_width': ctrl_w.tolist(),
        'D1': D1_MODEL,
        'n_buckets': N_BUCKETS,
        'optimization_loss': float(final_loss),
        'validation': metrics,
    }
    with open(OUTPUT_DIR / 'optimized_control_points.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    # 7. 导出优化后的STL
    print(f"\n  导出优化后的STL...")
    with open(OUTPUT_DIR / 'optimized_bucket.stl', 'w') as f:
        f.write("solid optimized_bucket\n")
        for i in range(0, len(opt_verts)-2, 3):
            v0, v1, v2 = opt_verts[i], opt_verts[i+1], opt_verts[i+2]
            n = np.cross(v1-v0, v2-v0)
            nl = np.linalg.norm(n)
            if nl > 0: n /= nl
            else: n = [0,0,1]
            f.write(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            f.write(f"    outer loop\n")
            f.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
            f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
            f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
            f.write(f"    endloop\n  endfacet\n")
        f.write("endsolid optimized_bucket\n")
    
    print(f"\n✅ 优化控制点保存: {OUTPUT_DIR / 'optimized_control_points.json'}")
    print(f"✅ 优化STL保存: {OUTPUT_DIR / 'optimized_bucket.stl'}")
    
    # 对比改进
    old_p95 = 49.2  # from previous run
    old_mean = 26.2
    print(f"\n{'='*50}")
    print(f"📊 改进对比")
    print(f"{'='*50}")
    print(f"  {'指标':<20} {'优化前':>10} {'优化后':>10} {'改进'}")
    print(f"  {'-'*50}")
    print(f"  {'P95偏差':<20} {old_p95:>8.1f}mm {metrics['p95']:>8.1f}mm {old_p95-metrics['p95']:>+6.1f}mm")
    print(f"  {'平均偏差':<20} {old_mean:>8.1f}mm {metrics['chamfer_mean']:>8.1f}mm {old_mean-metrics['chamfer_mean']:>+6.1f}mm")


if __name__ == "__main__":
    main()
