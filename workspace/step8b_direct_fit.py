"""
Step 8b: 直接3D Chamfer距离优化Bézier控制点

方法: 直接从DEC点云计算Chamfer距离 → 梯度下降优化
      不通过中间表示的r-z/width分离
"""
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from scipy.special import comb
import trimesh, json, time

OUTPUT_DIR = Path(r"workspace/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEC_MODEL_STL = OUTPUT_DIR / "dec_model_runner.stl"
D1 = 312.50
R1 = D1 / 2
NB = 21


def bern(i, n, t):
    return comb(n, i) * (t**i) * ((1-t)**(n-i))


def bezier_curve(ctrl, t_vals):
    """计算Bézier曲线上的点"""
    ctrl = np.asarray(ctrl)
    n = len(ctrl) - 1
    result = np.zeros((len(t_vals), ctrl.shape[1]))
    for i in range(n + 1):
        b = np.array([bern(i, n, t) for t in t_vals])
        result += np.outer(b, ctrl[i])
    return result


def generate_surface(ctrl_rz, ctrl_w, n_u=80, n_v=40):
    """从控制点生成水斗表面顶点"""
    u = np.linspace(0, 1, n_u)
    v = np.linspace(0, 1, n_v)
    
    profile = bezier_curve(ctrl_rz, u)  # (n_u, 2) → (r, z)
    
    # 宽度插值器
    z_ctrl = ctrl_w[:, 0]
    w_ctrl = ctrl_w[:, 1]
    w_fn = interp1d(z_ctrl, w_ctrl, kind='linear', bounds_error=False, fill_value=0.1)
    
    verts = []
    # 正半
    for i in range(n_u):
        r_i = profile[i, 0]
        z_i = profile[i, 1]
        hw = max(float(w_fn(z_i)), 0.05)
        for j in range(n_v):
            th = v[j] * hw / R1
            verts.append([r_i * np.cos(th), r_i * np.sin(th), z_i])
    # 负半
    for i in range(n_u):
        r_i = profile[i, 0]
        z_i = profile[i, 1]
        hw = max(float(w_fn(z_i)), 0.05)
        for j in range(1, n_v):
            th = -v[j] * hw / R1
            verts.append([r_i * np.cos(th), r_i * np.sin(th), z_i])
    
    return np.array(verts)


def chamfer_loss(params, dec_pts_sample, tree_dec, n_u=40, n_v=25):
    """Chamfer距离损失 + 平滑正则"""
    ctrl_rz = params[:10].reshape(5, 2)
    ctrl_w = params[10:20].reshape(5, 2)
    
    # P0约束: r = R1
    ctrl_rz[0, 0] = R1
    
    # 生成表面
    try:
        gen = generate_surface(ctrl_rz, ctrl_w, n_u=n_u, n_v=n_v)
    except:
        return 1e10
    
    # → DEC: 生成点到DEC的距离
    d1, _ = tree_dec.query(gen, k=1)
    
    # DEC → 生成: 采样点
    tree_gen = KDTree(gen)
    d2, _ = tree_gen.query(dec_pts_sample, k=1)
    
    chamfer = d1.mean() + d2.mean()
    
    # 平滑正则
    smooth = 0.0
    for i in range(1, len(ctrl_rz)-1):
        smooth += np.sum((ctrl_rz[i] - (ctrl_rz[i-1]+ctrl_rz[i+1])/2)**2)
    for i in range(1, len(ctrl_w)-1):
        smooth += np.sum((ctrl_w[i] - (ctrl_w[i-1]+ctrl_w[i+1])/2)**2)
    
    return chamfer + 0.001 * smooth


def main():
    print("="*70)
    print("  直接3D Chamfer优化 Bézier控制点")
    print("="*70)
    
    # 1. 加载DEC并提取参考点
    print("\n📦 加载DEC模型...")
    mesh = trimesh.load(str(DEC_MODEL_STL))
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    
    verts = mesh.vertices.copy()
    center = np.median(verts[:, :2], axis=0)
    r = np.sqrt((verts[:,0]-center[0])**2 + (verts[:,1]-center[1])**2)
    theta = np.degrees(np.arctan2(verts[:,1]-center[1], verts[:,0]-center[0])) % 360
    
    # 单水斗
    bw = 360.0 / NB * 1.05
    mask = (theta > -bw) & (theta < bw)
    if mask.sum() < 500:
        mask = theta < bw * 1.3
    
    dec_pts = verts[mask]
    dec_r = r[mask]
    dec_z = verts[mask, 2]
    
    print(f"  DEC单水斗: {len(dec_pts):,} 顶点")
    print(f"  r范围: [{dec_r.min():.1f}, {dec_r.max():.1f}]")
    print(f"  z范围: [{dec_z.min():.1f}, {dec_z.max():.1f}]")
    
    # 降采样DEC点以加速
    n_dec = min(3000, len(dec_pts))
    idx = np.random.choice(len(dec_pts), n_dec, replace=False)
    dec_sample = dec_pts[idx]
    tree_dec = KDTree(dec_sample)
    
    z_min, z_max = dec_z.min(), dec_z.max()
    
    # 2. 初始猜测
    ctrl_rz0 = np.array([
        [R1, z_min],                # P0: 分水刃在节圆, z最低
        [R1-8, z_min+15],
        [R1+15, (z_min+z_max)*0.35],
        [R1+5, (z_min+z_max)*0.7],
        [R1-2, z_max],
    ])
    
    ctrl_w0 = np.array([
        [z_min, 5.0],
        [z_min+12, 30.0],
        [(z_min+z_max)*0.4, 40.0],
        [(z_min+z_max)*0.7, 18.0],
        [z_max, 3.0],
    ])
    
    x0 = np.concatenate([ctrl_rz0.flatten(), ctrl_w0.flatten()])
    
    # 3. 优化 - 多轮
    print(f"\n🎯 优化中 (Chamfer距离最小化)...")
    start = time.time()
    
    # 轮1: 粗优化
    print("  轮1: 粗优化 (Nelder-Mead, 500 iters)...")
    result = minimize(
        chamfer_loss, x0, args=(dec_sample, tree_dec, 25, 15),
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 0.1, 'fatol': 0.5, 'disp': False}
    )
    print(f"    loss={result.fun:.1f}")
    
    # 轮2: 精细(更多面片)
    print("  轮2: 精细 (Nelder-Mead, 500 iters)...")
    result = minimize(
        chamfer_loss, result.x, args=(dec_sample, tree_dec, 40, 25),
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 0.05, 'fatol': 0.1, 'disp': False}
    )
    print(f"    loss={result.fun:.1f}")
    
    # 轮3: 更精细
    print("  轮3: 更精细 (Nelder-Mead, 300 iters)...")
    result = minimize(
        chamfer_loss, result.x, args=(dec_sample, tree_dec, 60, 35),
        method='Nelder-Mead',
        options={'maxiter': 300, 'xatol': 0.02, 'fatol': 0.05, 'disp': False}
    )
    print(f"    loss={result.fun:.1f}")
    
    elapsed = time.time() - start
    print(f"\n  优化耗时: {elapsed:.1f}s")
    
    # 4. 提取结果
    ctrl_rz_opt = result.x[:10].reshape(5, 2)
    ctrl_w_opt = result.x[10:20].reshape(5, 2)
    ctrl_rz_opt[0, 0] = R1
    
    print(f"\n  r-z控制点:")
    for i, (r_i, z_i) in enumerate(ctrl_rz_opt):
        print(f"    P{i}: r={r_i:.2f}, z={z_i:.2f}")
    
    print(f"\n  宽度控制点:")
    for i, (z_i, w_i) in enumerate(ctrl_w_opt):
        print(f"    P{i}: z={z_i:.2f}, half_w={w_i:.2f}")
    
    # 5. 生成精细表面并验证
    print(f"\n  生成精细表面...")
    opt_verts = generate_surface(ctrl_rz_opt, ctrl_w_opt, n_u=100, n_v=50)
    print(f"  顶点: {len(opt_verts):,}")
    
    # 6. 全量验证
    tree_opt = KDTree(opt_verts)
    tree_full = KDTree(dec_pts[:5000] if len(dec_pts) > 5000 else dec_pts)
    
    d1, _ = tree_full.query(opt_verts[:5000], k=1)
    d2, _ = tree_opt.query(dec_pts[:5000], k=1)
    all_d = np.concatenate([d1, d2])
    
    h_max = max(d1.max(), d2.max())
    chamfer = all_d.mean()
    p95 = np.percentile(all_d, 95)
    p50 = np.percentile(all_d, 50)
    
    print(f"\n{'='*50}")
    print(f"📊 最终验证")
    print(f"{'='*50}")
    print(f"  Hausdorff Max: {h_max:.2f} mm ({h_max/D1*100:.2f}% D1)")
    print(f"  Chamfer Mean:  {chamfer:.2f} mm ({chamfer/D1*100:.2f}% D1)")
    print(f"  P95:           {p95:.2f} mm ({p95/D1*100:.2f}% D1)")
    print(f"  P50:           {p50:.2f} mm ({p50/D1*100:.2f}% D1)")
    
    # 7. 保存
    result_data = {
        'ctrl_rz': ctrl_rz_opt.tolist(),
        'ctrl_w': ctrl_w_opt.tolist(),
        'validation': {
            'hausdorff_max': float(h_max),
            'chamfer_mean': float(chamfer),
            'p95': float(p95),
            'p50': float(p50),
            'p95_pct_d1': float(p95/D1*100),
            'mean_pct_d1': float(chamfer/D1*100),
        }
    }
    with open(OUTPUT_DIR / 'optimized_control_points_v2.json', 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n✅ 保存: {OUTPUT_DIR / 'optimized_control_points_v2.json'}")


if __name__ == "__main__":
    main()
