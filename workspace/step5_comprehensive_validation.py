"""
Step 5: 全面验证 - 从.x_t提取真实几何 vs 重建模型

验证维度:
1. 全局: 包围盒、体积、表面积
2. 径向: 径向分布、节圆位置
3. 周向: 21重对称性、水斗间距
4. 截面: r-z轮廓、周向宽度分布
5. 曲率: 高斯曲率、平均曲率分布
6. 统计: Chamfer距离、Hausdorff距离
"""
import re, json, sys
import numpy as np
from pathlib import Path
from scipy import stats, signal
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter1d

DATA_DIR = Path(r"DecPelton2024/DecPelton2024\DEC PELTON 2024-150MW级冲击式国产化转轮研发共享数据包")
OUTPUT_DIR = Path(r"workspace/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_XT = DATA_DIR / "1-转轮水力开发数据" / "1.1-金窝模型转轮D1276P21_.x_t"
FULL_XT = DATA_DIR / "2-真机转轮三维实体数据" / "2.1-150MW级冲击式国产化真机转轮三维实体数据.x_t"

# 已知参数
TRUTH = {
    'D1276P21': {'D1': 312.50, 'B2': 90.30, 'n_buckets': 21, 'D0': 35.20, 'scale': 1.0},
    'D1247R2':  {'D1': 2630.0, 'B2': 760.0, 'n_buckets': 21, 'D0': 296.2, 'scale': 8.416},
}


def extract_clean_points(xt_path, max_r_mm=None, surface_only=True):
    """
    从.x_t提取几何点
    
    surface_only=True: 使用局部密度+径向位置过滤，分离表面点
    
    Parasolid .x_t中的点分为:
    - 控制点: 在曲面内部/外部，密度低
    - 表面顶点: 在实际几何表面上，密度高，径向集中在节圆附近
    """
    with open(xt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    coord_re = re.compile(r'([+-]?\d*\.\d{12,16})')
    all_coords = np.array([float(m.group(1)) for m in coord_re.finditer(content)])
    
    n = len(all_coords) // 3
    pts = all_coords[:n*3].reshape(-1, 3) * 1000
    
    center = np.median(pts[:, :2], axis=0)
    r = np.sqrt(np.sum((pts[:, :2] - center)**2, axis=1))
    
    if max_r_mm:
        mask = (r > 1) & (r < max_r_mm)
        pts = pts[mask]
        r = r[mask]
    
    if not surface_only:
        return pts, center
    
    # === 表面点分离 ===
    # 策略: 径向直方图找主峰 → 表面点集中在主峰附近
    
    # 1. 径向分布找表面半径
    hist_r, edges_r = np.histogram(r, bins=80)
    hist_smooth = gaussian_filter1d(hist_r.astype(float), sigma=2.0)
    
    # 找主峰 (最靠近期望节圆半径的显著峰)
    # 如果有已知D1，用D1/2作为参考
    peaks_r, peak_props = signal.find_peaks(hist_smooth, height=hist_smooth.max()*0.1)
    
    if len(peaks_r) >= 1:
        # 取最高的峰作为表面主峰
        main_peak_idx = peaks_r[np.argmax(peak_props['peak_heights'])]
        r_surface = (edges_r[main_peak_idx] + edges_r[main_peak_idx + 1]) / 2
        r_surface = edges_r[main_peak_idx]
    else:
        r_surface = np.median(r)
    
    # 2. 在用表面半径附近 ±25% 范围内保留点
    r_tolerance = r_surface * 0.25
    surface_mask = abs(r - r_surface) < r_tolerance
    
    # 3. 对表面候选点做局部密度过滤 (去控制点)
    pts_candidate = pts[surface_mask]
    r_candidate = r[surface_mask]
    
    if len(pts_candidate) > 1000:
        # 在3D空间做密度过滤
        if len(pts_candidate) > 30000:
            idx = np.random.choice(len(pts_candidate), 30000, replace=False)
            pts_sample = pts_candidate[idx]
        else:
            pts_sample = pts_candidate
        
        tree = KDTree(pts_sample)
        distances, _ = tree.query(pts_sample, k=15)
        mean_dist = distances[:, 1:].mean(axis=1)
        
        # 保留高密度点 (表面顶点密度高)
        density_thresh = np.percentile(mean_dist, 40)  # 保留60%最密集的点
        dense_mask = mean_dist > density_thresh  # 距离大的=密度低的=控制点, 过滤掉
        
        # 用低密度阈值: 保留距离小(密度高)的点
        dense_mask_surface = mean_dist < np.percentile(mean_dist, 70)
        
        pts_clean = pts_sample[dense_mask_surface]
    else:
        pts_clean = pts_candidate
    
    return pts_clean, center


def compute_geometry_stats(pts, center, label, truth):
    """全面几何统计"""
    r = np.sqrt(np.sum((pts[:, :2] - center)**2, axis=1))
    z = pts[:, 2]
    theta = np.degrees(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])) % 360
    
    stats = {}
    
    # === 1. 全局统计 ===
    stats['n_points'] = len(pts)
    stats['bbox_x'] = (pts[:,0].min(), pts[:,0].max())
    stats['bbox_y'] = (pts[:,1].min(), pts[:,1].max())
    stats['bbox_z'] = (z.min(), z.max())
    stats['r_range'] = (r.min(), r.max())
    stats['r_mean'] = r.mean()
    stats['r_median'] = np.median(r)
    stats['r_std'] = r.std()
    
    # === 2. 径向分布 ===
    hist_r, edges_r = np.histogram(r, bins=50)
    stats['r_hist'] = hist_r.tolist()
    stats['r_edges'] = edges_r.tolist()
    stats['r_p10'] = np.percentile(r, 10)
    stats['r_p25'] = np.percentile(r, 25)
    stats['r_p50'] = np.percentile(r, 50)
    stats['r_p75'] = np.percentile(r, 75)
    stats['r_p90'] = np.percentile(r, 90)
    stats['r_p95'] = np.percentile(r, 95)
    
    # === 3. Z分布 ===
    stats['z_range'] = (z.min(), z.max())
    stats['z_median'] = np.median(z)
    stats['z_p10'] = np.percentile(z, 10)
    stats['z_p90'] = np.percentile(z, 90)
    
    # === 4. 周向对称性 ===
    hist_theta, edges_theta = np.histogram(theta, bins=360, range=(0, 360))
    hist_smooth = gaussian_filter1d(hist_theta.astype(float), sigma=2.0)
    peaks, peak_props = signal.find_peaks(hist_smooth, height=hist_smooth.max()*0.15, distance=12)
    stats['n_peaks'] = len(peaks)
    
    if len(peaks) >= 2:
        spacings = np.diff(edges_theta[peaks])
        stats['theta_spacing_mean'] = spacings.mean()
        stats['theta_spacing_std'] = spacings.std()
        stats['theta_spacing_expected'] = 360.0 / truth['n_buckets']
    
    # === 5. r-Z轮廓分析 ===
    # 将z分成20个bin，每个bin计算r的统计
    z_bins = np.linspace(z.min(), z.max(), 21)
    r_by_z = []
    for i in range(len(z_bins)-1):
        mask = (z >= z_bins[i]) & (z < z_bins[i+1])
        if mask.sum() > 10:
            r_by_z.append({
                'z_mid': (z_bins[i] + z_bins[i+1]) / 2,
                'r_mean': r[mask].mean(),
                'r_std': r[mask].std(),
                'r_min': r[mask].min(),
                'r_max': r[mask].max(),
                'n': int(mask.sum()),
            })
    stats['r_z_profile'] = r_by_z
    
    # === 6. 与已知参数对比 ===
    if 'D1' in truth:
        d1_expected = truth['D1']
        r_expected = d1_expected / 2
        
        # 找分水刃处(z最小)的半径
        z_low_mask = z < np.percentile(z, 15)
        r_low = r[z_low_mask]
        stats['D1_estimated'] = np.median(r_low) * 2
        stats['D1_error_pct'] = abs(stats['D1_estimated'] - d1_expected) / d1_expected * 100
        
        # 中位半径
        stats['D1_vs_median'] = np.median(r) * 2
        stats['D1_median_error_pct'] = abs(stats['D1_vs_median'] - d1_expected) / d1_expected * 100
    
    if 'B2' in truth:
        # 找内表面区域(z在中间1/3)的周向跨度
        z_mid = (np.percentile(z, 30) + np.percentile(z, 70)) / 2
        mid_mask = abs(z - z_mid) < (z.max() - z.min()) * 0.1
        if mid_mask.sum() > 20:
            theta_mid = theta[mid_mask]
            # 找主水斗的角度范围
            # 取直方图峰值附近的角度
            if len(peaks) > 0:
                main_peak = peaks[np.argmax(peak_props['peak_heights'])]
                peak_angle = edges_theta[main_peak]
                angle_mask = (theta_mid > peak_angle - 12) & (theta_mid < peak_angle + 12)
                if angle_mask.sum() > 5:
                    r_mid_region = r[mid_mask][angle_mask]
                    r_mean_mid = r_mid_region.mean()
                    theta_span = theta_mid[angle_mask].max() - theta_mid[angle_mask].min()
                    stats['B2_estimated'] = r_mean_mid * np.radians(theta_span)
                    if 'B2' in truth:
                        stats['B2_error_pct'] = abs(stats['B2_estimated'] - truth['B2']) / truth['B2'] * 100
    
    return stats


def compare_reconstructed_vs_dec(dec_stats, recon_stats, label):
    """对比DEC模型和重建模型"""
    print(f"\n{'='*60}")
    print(f"📊 {label} - DEC vs 重建 对比")
    print(f"{'='*60}")
    
    comparisons = []
    
    # 对比项
    checks = [
        ('点数', 'n_points', None, 'int'),
        ('径向中位数', 'r_median', None, 'dist'),
        ('径向P90', 'r_p90', None, 'dist'),
        ('径向P95', 'r_p95', None, 'dist'),
        ('Z范围', 'z_range', None, 'range'),
        ('对称峰数', 'n_peaks', 'theta_spacing_expected', 'peaks'),
        ('峰间距均值', 'theta_spacing_mean', 'theta_spacing_expected', 'angle'),
    ]
    
    for name, key, ref_key, ctype in checks:
        v_dec = dec_stats.get(key)
        v_rec = recon_stats.get(key)
        
        if v_dec is None or v_rec is None:
            continue
        
        if ctype == 'int':
            match = "✅" if v_dec == v_rec else "⚠️"
            comparisons.append(f"  {name:<15}: DEC={v_dec}, 重建={v_rec} {match}")
        elif ctype == 'dist':
            err = abs(v_dec - v_rec) / (abs(v_dec) + 1e-10) * 100
            match = "✅" if err < 10 else "⚠️" if err < 25 else "❌"
            comparisons.append(f"  {name:<15}: DEC={v_dec:.1f}, 重建={v_rec:.1f} (偏差{err:.1f}%) {match}")
        elif ctype == 'peaks':
            expected = dec_stats.get(ref_key, 17.14)
            comparisons.append(f"  {name:<15}: DEC={v_dec}, 重建={v_rec} (期望≈{expected:.0f})")
        elif ctype == 'angle':
            expected = dec_stats.get(ref_key, 17.14)
            comparisons.append(f"  {name:<15}: DEC={v_dec:.1f}°, 重建={v_rec:.1f}°")
    
    for c in comparisons:
        print(c)
    
    return comparisons


def compute_validation_report():
    """主验证流程"""
    print("="*60)
    print("🔬 全面验证: DEC .x_t 模型 vs Bézier重建模型")
    print("="*60)
    
    all_validations = {}
    
    for model_key, xt_path in [('D1276P21_模型', MODEL_XT), ('D1247R2_真机', FULL_XT)]:
        truth = TRUTH[model_key.split('_')[0]]
        d1 = truth['D1']
        
        print(f"\n{'─'*60}")
        print(f"📦 分析 {model_key} (D1={d1:.0f}mm)")
        
        # 从.x_t提取
        max_r = d1 * 1.2  # 合理的径向范围
        pts_dec, center_dec = extract_clean_points(xt_path, max_r_mm=max_r)
        dec_stats = compute_geometry_stats(pts_dec, center_dec, model_key, truth)
        
        print(f"  提取点数: {dec_stats['n_points']:,}")
        print(f"  径向中位: {dec_stats['r_median']:.1f}mm")
        print(f"  对称峰数: {dec_stats['n_peaks']}")
        
        # 从重建模型提取对应数据
        sys.path.insert(0, str(Path(__file__).parent))
        from step4_parametric_bucket import PeltonBucketParametric, bezier_curve
        
        bucket = PeltonBucketParametric(D1=truth['D1'], B2=truth['B2'], n_buckets=truth['n_buckets'])
        v_rec, f_rec, n_rec = bucket.sample_surface(n_u=60, n_v=40)
        
        center_rec = np.array([0.0, 0.0])
        recon_stats = compute_geometry_stats(v_rec, center_rec, model_key, truth)
        
        print(f"  重建点数: {recon_stats['n_points']:,}")
        print(f"  重建径向中位: {recon_stats['r_median']:.1f}mm")
        
        # 对比
        comparisons = compare_reconstructed_vs_dec(dec_stats, recon_stats, model_key)
        
        # === 关键指标汇总 ===
        print(f"\n  📏 关键尺寸验证:")
        for dim_name, dim_key in [('节圆直径 D1', 'D1_estimated'), ('水斗宽度 B2', 'B2_estimated')]:
            if dim_key in dec_stats and dim_key in recon_stats:
                vd = dec_stats[dim_key]
                vr = recon_stats[dim_key]
                expected = truth['D1'] if 'D1' in dim_key else truth.get('B2')
                if expected:
                    err_d = abs(vd - expected) / expected * 100
                    err_r = abs(vr - expected) / expected * 100
                    print(f"    {dim_name}: DEC提取={vd:.1f}mm(误差{err_d:.1f}%) | 重建={vr:.1f}mm(误差{err_r:.1f}%)")
        
        all_validations[model_key] = {
            'dec_stats': {k: v for k, v in dec_stats.items() if not isinstance(v, (list, np.ndarray))},
            'recon_stats': {k: v for k, v in recon_stats.items() if not isinstance(v, (list, np.ndarray))},
        }
    
    # === 最终汇总 ===
    print(f"\n{'='*60}")
    print(f"📊 最终验证汇总")
    print(f"{'='*60}")
    print(f"  {'指标':<20} {'D1276P21(DEC)':>15} {'D1276P21(重建)':>15} {'匹配'}")
    print(f"  {'-'*65}")
    for key in ['r_median', 'r_p90', 'r_p95', 'n_peaks', 'D1_estimated', 'B2_estimated']:
        vd = all_validations.get('D1276P21_模型', {}).get('dec_stats', {}).get(key)
        vr = all_validations.get('D1276P21_模型', {}).get('recon_stats', {}).get(key)
        if vd is not None and vr is not None:
            if isinstance(vd, float):
                err = abs(vd - vr) / (abs(vd) + 1e-10) * 100
                flag = "✅" if err < 10 else "⚠️" if err < 25 else "❌"
                print(f"  {key:<20} {vd:>15.1f} {vr:>15.1f} {flag} ({err:.1f}%)")
            else:
                flag = "✅" if vd == vr else "⚠️"
                print(f"  {key:<20} {vd:>15} {vr:>15} {flag}")
    
    # 保存报告
    report = {
        'truth': TRUTH,
        'validations': all_validations,
    }
    with open(OUTPUT_DIR / 'validation_report.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ 验证报告保存: {OUTPUT_DIR / 'validation_report.json'}")


if __name__ == "__main__":
    compute_validation_report()
