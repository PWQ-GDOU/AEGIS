"""
Step 3: 从 Parasolid .x_t 文件中提取几何验证基准
对比已知参数，建立参数化模型验证框架
"""
import re
import numpy as np
from pathlib import Path
from scipy import signal

DATA_DIR = Path(r"DecPelton2024/DecPelton2024\DEC PELTON 2024-150MW级冲击式国产化转轮研发共享数据包")
MODEL_XT = DATA_DIR / "1-转轮水力开发数据" / "1.1-金窝模型转轮D1276P21_.x_t"
FULL_XT = DATA_DIR / "2-真机转轮三维实体数据" / "2.1-150MW级冲击式国产化真机转轮三维实体数据.x_t"

# ============================================================
# 金窝电站模型转轮 - 已知参数 (来自DEC模型试验报告 DEC-TWH-SL-006)
# ============================================================
TRUTH_MODEL = {
    "转轮节圆直径 D1": 312.50,    # mm
    "水斗宽度 B2": 90.30,         # mm
    "喷嘴出口直径 D0": 35.20,      # mm
    "水斗数": 21,
    "喷嘴数": 6,
    "放大比尺": 8.416,
    "最优效率": 91.72,            # %
    "额定效率": 91.61,            # %
}

TRUTH_FULL = {
    "额定出力": 143.60,           # MW
    "额定水头": 595.00,           # m
    "额定转速": 375.00,           # rpm
    "转轮直径": 2.63,             # m = 2630 mm
    "射流直径": 0.24,             # m
    "水斗数": 21,
    "喷嘴数": 6,
}


def extract_points_from_xt(xt_path: Path):
    """从 Parasolid 文本中提取高精度坐标点"""
    with open(xt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 匹配高精度浮点数 (几何坐标的特征: 12-16位小数)
    coord_re = re.compile(r'([+-]?\d*\.\d{12,16})')
    all_coords = np.array([float(m.group(1)) for m in coord_re.finditer(content)])
    
    n = len(all_coords) // 3
    pts = all_coords[:n*3].reshape(-1, 3)
    
    return pts


def analyze_model_geometry(pts, label, truth):
    """分析提取的几何点，与已知参数对比"""
    print(f"\n{'='*60}")
    print(f"📐 {label} 几何分析")
    print(f"{'='*60}")
    
    # 坐标单位判断：已知D1=312.50mm
    # 如果坐标单位是 m → 缩放 ×1000
    scale = 1000.0  # 假设 m → mm
    pts_mm = pts * scale
    
    # 找旋转中心
    center = np.median(pts_mm[:, :2], axis=0)
    
    # 径向距离和角度
    r = np.sqrt(np.sum((pts_mm[:, :2] - center)**2, axis=1))
    theta = np.arctan2(pts_mm[:, 1] - center[1], pts_mm[:, 0] - center[0])
    theta_deg = np.degrees(theta) % 360
    
    print(f"\n📊 基本统计:")
    print(f"  总提取点数: {len(pts):,}")
    print(f"  坐标范围 X: [{pts_mm[:,0].min():.1f}, {pts_mm[:,0].max():.1f}] mm")
    print(f"  坐标范围 Y: [{pts_mm[:,1].min():.1f}, {pts_mm[:,1].max():.1f}] mm")
    print(f"  坐标范围 Z: [{pts_mm[:,2].min():.1f}, {pts_mm[:,2].max():.1f}] mm")
    print(f"  旋转中心: ({center[0]:.1f}, {center[1]:.1f}) mm")
    
    # 过滤异常值：只保留合理半径范围内的点
    # 模型转轮D1=312.50mm → 半径范围应该在 0~200mm 左右
    r_max_expected = truth.get("转轮节圆直径 D1", 312.50) / 2 * 1.5  # 1.5倍容差
    valid_mask = (r > 1) & (r < r_max_expected)
    r_valid = r[valid_mask]
    pts_valid = pts_mm[valid_mask]
    theta_valid = theta_deg[valid_mask]
    
    print(f"\n📊 过滤后 (r < {r_max_expected:.0f}mm):")
    print(f"  有效点数: {len(pts_valid):,} / {len(pts):,} ({len(pts_valid)/len(pts)*100:.1f}%)")
    print(f"  径向距离: min={r_valid.min():.1f}  median={np.median(r_valid):.1f}  P90={np.percentile(r_valid,90):.1f}  max={r_valid.max():.1f} mm")
    
    # === 验证项1: 旋转对称性 (水斗数) ===
    print(f"\n🔍 验证项1: 水斗数量 (期望 {truth.get('水斗数', '?')})")
    
    # 角度直方图 - 找21个水斗对应的峰值
    hist, edges = np.histogram(theta_valid, bins=360, range=(0, 360))
    # 平滑
    from scipy.ndimage import gaussian_filter1d
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=1.5)
    peaks, props = signal.find_peaks(hist_smooth, height=hist_smooth.max()*0.15, distance=12)
    
    n_buckets_detected = len(peaks)
    expected = truth.get("水斗数", 21)
    spacing_expected = 360.0 / expected
    
    if n_buckets_detected > 0:
        spacings = np.diff(edges[peaks])
        print(f"  检测到 {n_buckets_detected} 个峰值")
        print(f"  峰间距: mean={spacings.mean():.1f}° (期望 {spacing_expected:.1f}°)")
        print(f"  匹配度: {'✅ 一致' if abs(n_buckets_detected - expected) <= 2 else '⚠️ 偏差' if abs(n_buckets_detected - expected) <= 5 else '❌ 不匹配'}")
    else:
        print(f"  ❌ 无法检测到足够的峰值")
    
    # === 验证项2: 节圆直径 ===
    print(f"\n🔍 验证项2: 节圆直径 D1 (期望 {truth.get('转轮节圆直径 D1', '?')} mm)")
    
    if "转轮节圆直径 D1" in truth:
        d1_expected = truth["转轮节圆直径 D1"]
        r_expected = d1_expected / 2
        
        # 节圆通常在P50-P75的径向位置
        for pctl in [50, 60, 70, 75, 80]:
            r_p = np.percentile(r_valid, pctl)
            diam_est = r_p * 2
            err = abs(diam_est - d1_expected) / d1_expected * 100
            print(f"  P{pctl}径向 → D≈{diam_est:.1f}mm (误差 {err:.1f}%)", 
                  end="")
            if err < 5:
                print(" ✅")
            elif err < 15:
                print(" ⚠️")
            else:
                print("")
    
    # === 验证项3: 水斗宽度 ===
    print(f"\n🔍 验证项3: 水斗宽度 B2 (期望 {truth.get('水斗宽度 B2', '?')} mm)")
    
    if "水斗宽度 B2" in truth:
        b2_expected = truth["水斗宽度 B2"]
        
        # 水斗宽度 ≈ 在同一角度区域内, r_max - r_min
        # 在峰值角度附近取径向跨度
        if n_buckets_detected > 0:
            bucket_widths = []
            for peak_angle in edges[peaks]:
                angle_mask = (theta_valid > peak_angle - 8) & (theta_valid < peak_angle + 8)
                if angle_mask.sum() > 10:
                    r_bucket = r_valid[angle_mask]
                    w = np.percentile(r_bucket, 90) - np.percentile(r_bucket, 10)
                    bucket_widths.append(w)
            
            if bucket_widths:
                w_mean = np.mean(bucket_widths)
                w_std = np.std(bucket_widths)
                err = abs(w_mean - b2_expected) / b2_expected * 100
                print(f"  水斗径向跨度: {w_mean:.1f} ± {w_std:.1f} mm (误差 {err:.1f}%)")
                if err < 10:
                    print(f"  ✅ 与已知B2={b2_expected}mm 基本一致")
                else:
                    print(f"  ⚠️ 偏差较大，注意：水斗宽度定义可能与径向跨度不完全等价")
    
    # === 验证项4: 整体外廓 ===
    print(f"\n🔍 验证项4: 转轮外廓")
    max_r = r_valid.max()
    diam_max = max_r * 2
    print(f"  最大半径: {max_r:.1f} mm → 最大直径: {diam_max:.1f} mm")
    
    return {
        'pts_valid': pts_valid,
        'r_valid': r_valid,
        'theta_valid': theta_valid,
        'center': center,
        'n_buckets': n_buckets_detected,
        'key_dims': {
            'D1_estimated': np.percentile(r_valid, 70) * 2,
            'max_diameter': diam_max,
        }
    }


def main():
    print("="*60)
    print("🔬 DEC模型几何验证")
    print("="*60)
    
    print(f"\n📋 已知参数来源: DEC模型试验报告 DEC-TWH-SL-006")
    for k, v in TRUTH_MODEL.items():
        print(f"  {k}: {v}")
    
    # 分析模型转轮
    print(f"\n⏳ 提取模型转轮几何点...")
    pts_model = extract_points_from_xt(MODEL_XT)
    result_model = analyze_model_geometry(pts_model, "模型转轮 D1276P21", TRUTH_MODEL)
    
    print(f"\n{'='*60}")
    print(f"📋 验证总结")
    print(f"{'='*60}")
    print(f"""
    验证方法:
    1. 从 Parasolid .x_t 文件中提取 {len(pts_model):,} 个几何点（控制点+网格顶点）
    2. 与DEC模型试验报告的已知参数进行对比
    
    结论:
    - 坐标数据已成功提取，包含水斗表面的控制点和网格顶点
    - 点云数据可验证整体尺寸和对称性
    - 完整的网格级验证需要 CAD Assistant 或 Siemens NX 导出 STL 后完成
    - 参数化模型可直接使用 DEC 试验报告的参数 (D1=312.50, B2=90.30, 21斗) 构建
    
    后续完整验证:
    ┌─────────────────────────────────────────────┐
    │ 1. 获取STL网格 (CAD Assistant / 队友NX导出)   │
    │ 2. Hausdorff距离对比 (我们的模型 vs DEC模型)  │
    │ 3. 关键截面轮廓对齐验证                       │
    │ 4. 曲率分布一致性检验                         │
    └─────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()
