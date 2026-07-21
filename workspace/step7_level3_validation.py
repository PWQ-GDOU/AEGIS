"""
Step 7: 三级验证 - DEC真机模型 vs Bézier重建模型
全面的网格级对比
"""
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree
from scipy import stats
import json, time

OUTPUT_DIR = Path(r"B:\undertaking\学业\非专业相关\比赛\挑战杯\水轮机\workspace\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 文件路径
DEC_MODEL_STL = OUTPUT_DIR / "dec_model_runner.stl"       # DEC 模型转轮
DEC_FULL_STL = OUTPUT_DIR / "dec_full_runner.stl"         # DEC 真机转轮
OUR_MODEL_STL = OUTPUT_DIR / "model_runner_D1276P21.stl"  # 我们的重建

# 已知参数
D1_MODEL = 312.50  # mm, 模型转轮节圆直径
D1_FULL = 2630.0   # mm, 真机转轮节圆直径
N_BUCKETS = 21
SCALE = 8.416


def load_stl(path):
    """加载STL并返回顶点+面片"""
    import trimesh
    print(f"   加载: {path.name} ({path.stat().st_size/1024/1024:.1f} MB)...")
    mesh = trimesh.load(str(path))
    if isinstance(mesh, trimesh.Scene):
        meshes = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(meshes)
    print(f"   顶点: {len(mesh.vertices):,}, 面片: {len(mesh.faces):,}")
    return mesh


def extract_bucket(mesh, bucket_idx=0, n_buckets=21):
    """从完整转轮中提取单个水斗 (按角度分割)"""
    vertices = mesh.vertices.copy()
    center = np.median(vertices[:, :2], axis=0)
    
    # 计算每个顶点的角度
    theta = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    theta_deg = np.degrees(theta) % 360
    
    # 每个水斗占 360/21 ≈ 17.14°
    bucket_width = 360.0 / n_buckets
    bucket_center = bucket_idx * bucket_width
    
    # 取该水斗的角度范围 (稍微扩展以包含完整水斗)
    margin = bucket_width * 1.2
    bucket_mask = (theta_deg > bucket_center - margin/2) & (theta_deg < bucket_center + margin/2)
    
    # 处理环绕
    if bucket_mask.sum() < 100:
        bucket_mask = (theta_deg > 360 - margin/2) | (theta_deg < margin/2 - (360 - bucket_center))
    
    bucket_verts = vertices[bucket_mask]
    
    if len(bucket_verts) < 100:
        print(f"   ⚠️ 水斗{bucket_idx}只有{len(bucket_verts)}个顶点, 返回全部")
        return vertices, center
    
    print(f"   水斗{bucket_idx}: {len(bucket_verts):,} 顶点 (共{len(vertices):,})")
    return bucket_verts, center


def compute_hausdorff(pts_a, pts_b, sample=10000):
    """计算Hausdorff距离"""
    # 采样以加速
    if len(pts_a) > sample:
        idx_a = np.random.choice(len(pts_a), sample, replace=False)
        pts_a = pts_a[idx_a]
    if len(pts_b) > sample:
        idx_b = np.random.choice(len(pts_b), sample, replace=False)
        pts_b = pts_b[idx_b]
    
    # A → B 的距离
    tree_b = KDTree(pts_b)
    dist_a2b, _ = tree_b.query(pts_a, k=1)
    
    # B → A 的距离
    tree_a = KDTree(pts_a)
    dist_b2a, _ = tree_a.query(pts_b, k=1)
    
    return {
        'hausdorff_max': max(dist_a2b.max(), dist_b2a.max()),
        'hausdorff_mean': (dist_a2b.mean() + dist_b2a.mean()) / 2,
        'chamfer': (dist_a2b.mean() + dist_b2a.mean()) / 2,  # same as hausdorff_mean here
        'directional_a2b': {'max': float(dist_a2b.max()), 'mean': float(dist_a2b.mean()), 'std': float(dist_a2b.std())},
        'directional_b2a': {'max': float(dist_b2a.max()), 'mean': float(dist_b2a.mean()), 'std': float(dist_b2a.std())},
        'p95': float(np.percentile(np.concatenate([dist_a2b, dist_b2a]), 95)),
        'p50': float(np.percentile(np.concatenate([dist_a2b, dist_b2a]), 50)),
    }


def compute_cross_section(pts, center, z_range, n_bins=30):
    """提取 r-z 截面轮廓"""
    z = pts[:, 2]
    r = np.sqrt((pts[:, 0] - center[0])**2 + (pts[:, 1] - center[1])**2)
    
    z_min, z_max = z_range
    bins = np.linspace(z_min, z_max, n_bins + 1)
    
    profile = []
    for i in range(n_bins):
        mask = (z >= bins[i]) & (z < bins[i+1])
        if mask.sum() > 5:
            profile.append({
                'z': (bins[i] + bins[i+1]) / 2,
                'r_mean': float(r[mask].mean()),
                'r_min': float(r[mask].min()),
                'r_max': float(r[mask].max()),
                'r_std': float(r[mask].std()),
                'n': int(mask.sum()),
            })
    
    return profile


def compute_curvature_stats(mesh):
    """计算曲率统计"""
    try:
        # 用trimesh计算曲率
        gauss = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, 0.5)
        mean = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, 0.5)
        
        # 过滤离群值
        gauss = gauss[np.abs(gauss) < np.percentile(np.abs(gauss), 99)]
        mean = mean[np.abs(mean) < np.percentile(np.abs(mean), 99)]
        
        return {
            'gaussian': {
                'mean': float(np.mean(gauss)), 'std': float(np.std(gauss)),
                'p50': float(np.percentile(gauss, 50)), 'p95': float(np.percentile(gauss, 95)),
            },
            'mean_curvature': {
                'mean': float(np.mean(mean)), 'std': float(np.std(mean)),
                'p50': float(np.percentile(mean, 50)), 'p95': float(np.percentile(mean, 95)),
            }
        }
    except Exception as e:
        print(f"   曲率计算失败: {e}")
        return None


def compute_global_stats(mesh, label):
    """全局统计"""
    verts = mesh.vertices
    center = np.median(verts[:, :2], axis=0)
    r = np.sqrt((verts[:, 0] - center[0])**2 + (verts[:, 1] - center[1])**2)
    z = verts[:, 2]
    
    return {
        'label': label,
        'n_vertices': len(verts),
        'n_faces': len(mesh.faces),
        'bbox': {
            'x': [float(verts[:,0].min()), float(verts[:,0].max())],
            'y': [float(verts[:,1].min()), float(verts[:,1].max())],
            'z': [float(z.min()), float(z.max())],
        },
        'r_stats': {
            'min': float(r.min()), 'max': float(r.max()),
            'mean': float(r.mean()), 'median': float(np.median(r)),
            'p10': float(np.percentile(r,10)), 'p90': float(np.percentile(r,90)),
        },
        'surface_area': float(mesh.area) if hasattr(mesh, 'area') else None,
        'volume': float(mesh.volume) if hasattr(mesh, 'volume') and mesh.is_watertight else None,
    }


def main():
    print("="*70)
    print("  三级验证: DEC真机模型 vs Bézier参数化重建")
    print("="*70)
    
    # 1. 加载模型
    print(f"\n📦 加载模型...")
    dec_full = load_stl(DEC_FULL_STL)
    dec_model = load_stl(DEC_MODEL_STL)
    our_model = load_stl(OUR_MODEL_STL)
    
    # 2. 全局统计对比
    print(f"\n{'='*50}")
    print(f"📊 全局统计对比")
    print(f"{'='*50}")
    
    stats_dec = compute_global_stats(dec_model, "DEC 模型转轮")
    stats_our = compute_global_stats(our_model, "我们的重建")
    
    for key in ['n_vertices', 'n_faces', 'surface_area']:
        vd = stats_dec.get(key)
        vo = stats_our.get(key)
        if vd and vo:
            err = abs(vd - vo) / max(abs(vd), 1e-10) * 100
            flag = "✅" if err < 20 else "⚠️" if err < 50 else "❌"
            print(f"  {key:<20}: DEC={vd:>10,.0f}, 重建={vo:>10,.0f} (差异{err:.1f}%) {flag}")
    
    # 3. 提取单个水斗并做点云对比
    print(f"\n{'='*50}")
    print(f"📊 单水斗点云对比")
    print(f"{'='*50}")
    
    dec_bucket, dec_center = extract_bucket(dec_model, bucket_idx=0)
    our_bucket = our_model.vertices
    
    print(f"\n  DEC单水斗点: {len(dec_bucket):,}")
    print(f"  重建单水斗点: {len(our_bucket):,}")
    
    # 计算 Hausdorff 距离
    print(f"\n  🔬 计算点云距离...")
    dists = compute_hausdorff(dec_bucket, our_bucket, sample=5000)
    
    print(f"\n  📏 点云距离统计:")
    print(f"    Hausdorff Max:   {dists['hausdorff_max']:.3f} mm")
    print(f"    Chamfer Mean:    {dists['chamfer']:.3f} mm")
    print(f"    P95 距离:        {dists['p95']:.3f} mm")
    print(f"    P50 距离:        {dists['p50']:.3f} mm")
    
    # 与节圆直径的比例
    d1_ratio_max = dists['hausdorff_max'] / D1_MODEL * 100
    d1_ratio_p95 = dists['p95'] / D1_MODEL * 100
    print(f"\n  相对D1(312.5mm)的比例:")
    print(f"    Max/D1: {d1_ratio_max:.2f}%")
    print(f"    P95/D1: {d1_ratio_p95:.2f}%")
    
    # 4. 截面轮廓对比
    print(f"\n{'='*50}")
    print(f"📊 截面轮廓 (r-z) 对比")
    print(f"{'='*50}")
    
    z_range_dec = (dec_bucket[:,2].min(), dec_bucket[:,2].max())
    z_range_our = (our_bucket[:,2].min(), our_bucket[:,2].max())
    
    profile_dec = compute_cross_section(dec_bucket, dec_center, z_range_dec)
    profile_our = compute_cross_section(our_bucket, np.array([0,0]), z_range_our)
    
    # 比较相同z位置的r值
    z_dec = np.array([p['z'] for p in profile_dec])
    r_dec = np.array([p['r_mean'] for p in profile_dec])
    z_our = np.array([p['z'] for p in profile_our])
    r_our = np.array([p['r_mean'] for p in profile_our])
    
    # 插值比较
    from scipy.interpolate import interp1d
    z_common = np.linspace(max(z_dec.min(), z_our.min()), min(z_dec.max(), z_our.max()), 20)
    r_dec_interp = interp1d(z_dec, r_dec, bounds_error=False)(z_common)
    r_our_interp = interp1d(z_our, r_our, bounds_error=False)(z_common)
    
    valid = ~np.isnan(r_dec_interp) & ~np.isnan(r_our_interp)
    if valid.sum() > 3:
        r_diff = np.abs(r_dec_interp[valid] - r_our_interp[valid])
        print(f"  截面r值平均偏差: {r_diff.mean():.2f} mm")
        print(f"  截面r值最大偏差: {r_diff.max():.2f} mm")
        print(f"  相对D1的比例: {r_diff.mean()/D1_MODEL*100:.2f}%")
    
    # 5. 保存结果
    report = {
        'global_stats': {'dec': stats_dec, 'our': stats_our},
        'hausdorff': dists,
        'cross_section_diff_mean_mm': float(r_diff.mean()) if valid.sum() > 3 else None,
        'D1': D1_MODEL,
        'hausdorff_max_pct_D1': d1_ratio_max,
        'hausdorff_p95_pct_D1': d1_ratio_p95,
    }
    
    with open(OUTPUT_DIR / 'validation_level3_report.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ 三级验证报告: {OUTPUT_DIR / 'validation_level3_report.json'}")
    
    # 总结
    print(f"\n{'='*70}")
    print(f"  验证结论")
    print(f"{'='*70}")
    print(f"""
  📏 Hausdorff距离:
     - 最大偏差: {dists['hausdorff_max']:.1f} mm ({d1_ratio_max:.1f}% D1)
     - P95偏差:  {dists['p95']:.1f} mm ({d1_ratio_p95:.1f}% D1)
     - 平均偏差: {dists['chamfer']:.1f} mm
     
  📊 评估标准:
     - <5% D1: ✅ 优秀
     - 5-10% D1: ⚠️ 可接受
     - >10% D1: ❌ 需改进
""")


if __name__ == "__main__":
    main()
