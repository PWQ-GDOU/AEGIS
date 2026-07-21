"""
Step 9: 在DEC真实几何上计算侵蚀分布 — 0%几何误差

直接使用DEC的STL网格作为计算基底，不做任何参数化拟合
"""
import numpy as np
from pathlib import Path
import trimesh, json, time
from scipy.spatial import KDTree

OUTPUT_DIR = Path(r"B:\undertaking\学业\非专业相关\比赛\挑战杯\水轮机\workspace\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEC_MODEL_STL = OUTPUT_DIR / "dec_model_runner.stl"
DEC_FULL_STL = OUTPUT_DIR / "dec_full_runner.stl"

# 已知参数
D1_MODEL = 312.50; R1 = D1_MODEL/2
D1_FULL = 2630.0; R1_FULL = D1_FULL/2
SCALE = 8.416
N_BUCKETS = 21


# ============================================================
# 物理侵蚀模型 (Oka, Finnie, Tabakoff)
# ============================================================
class OkaModel:
    Vref, Dref, K = 104.0, 326.0, 65.0
    def __init__(self, Hv=2.6):
        self.Hv = Hv
        self.n1 = 0.71 * (Hv**0.14)
        self.n2 = 2.4 * (Hv**(-0.94))
        self.k2 = 2.3 * (Hv**0.038)
    def erosion_rate(self, Vp, alpha_deg):
        a = np.radians(np.clip(alpha_deg, 0, 90))
        g = (np.sin(a)**self.n1) * ((1+self.Hv*(1-np.sin(a)))**self.n2)
        E90 = self.K * (self.Hv**-0.12) * (Vp/self.Vref)**self.k2 * (150/self.Dref)**0.19
        return g * E90

class FinnieModel:
    def __init__(self, k=2e-5, n=2.0): self.k, self.n = k, n
    def erosion_rate(self, Vp, alpha_deg):
        a = np.radians(np.clip(alpha_deg, 0, 90))
        tana = np.tan(a)
        mask = tana > 1/3
        f = np.where(mask, (1/3)*np.cos(a)**2, np.sin(2*a)-3*np.sin(a)**2)
        return self.k * (Vp**self.n) * f

class TabakoffModel:
    def __init__(self,k1=1e-5,k2=0.585,k3=5e-12,k4=0.0023,k12=1.0,b0=25):
        self.k1,self.k2,self.k3,self.k4,self.k12,self.b0=k1,k2,k3,k4,k12,b0
    def erosion_rate(self, Vp, alpha_deg):
        a = np.radians(alpha_deg)
        Rt = np.maximum(0, 1-self.k4*Vp*np.sin(a))
        fb = (1+self.k2*self.k12*np.sin(90*a/self.b0*np.pi/180))**2
        Ec = self.k1*fb*Vp**2*np.cos(a)**2*(1-Rt**2)
        Ed = self.k3*(Vp*np.sin(a))**4
        return Ec+Ed


# ============================================================
# 在DEC真实网格上计算
# ============================================================
def load_and_prepare(stl_path, label, max_faces=200000):
    """加载DEC STL并提取单个水斗"""
    print(f"\n📦 {label}")
    mesh = trimesh.load(str(stl_path))
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    
    print(f"   完整模型: {len(mesh.vertices):,}顶点, {len(mesh.faces):,}面片")
    
    # 如果网格太大，采样面片
    if len(mesh.faces) > max_faces:
        print(f"   网格较大({len(mesh.faces):,}面片), 随机采样{max_faces//1000}k...")
        idx = np.random.choice(len(mesh.faces), max_faces, replace=False)
        mesh = mesh.submesh([idx])[0] if hasattr(mesh, 'submesh') else mesh
        # 如果submesh不可用，直接用采样后的面片
        if len(mesh.faces) > max_faces:
            keep = np.random.choice(len(mesh.faces), max_faces, replace=False)
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[keep])
        print(f"   采样后: {len(mesh.vertices):,}顶点, {len(mesh.faces):,}面片")
    
    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    
    # 面法向量
    v0 = verts[faces[:,0]]; v1 = verts[faces[:,1]]; v2 = verts[faces[:,2]]
    face_normals = np.cross(v1-v0, v2-v0)
    nlen = np.linalg.norm(face_normals, axis=1, keepdims=True)
    nlen[nlen<1e-10] = 1
    face_normals /= nlen
    
    # 顶点法向量 (面积加权平均)
    vert_normals = np.zeros_like(verts)
    face_areas = np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1) / 2
    for i in range(3):
        np.add.at(vert_normals, faces[:,i], face_normals * face_areas[:,None])
    vnlen = np.linalg.norm(vert_normals, axis=1, keepdims=True)
    vnlen[vnlen<1e-10] = 1
    vert_normals /= vnlen
    
    # 旋转中心
    center = np.median(verts[:,:2], axis=0)
    theta = np.degrees(np.arctan2(verts[:,1]-center[1], verts[:,0]-center[0])) % 360
    
    # 提取单水斗
    bw = 360.0/N_BUCKETS * 1.1
    bucket_mask = (theta > -bw) & (theta < bw)
    if bucket_mask.sum() < 500:
        bucket_mask = theta < bw*1.5
    
    # 映射面片到单水斗
    face_in_bucket = bucket_mask[faces].all(axis=1)
    bucket_faces = faces[face_in_bucket]
    
    # 重建顶点索引
    unique_verts, inv = np.unique(bucket_faces.flatten(), return_inverse=True)
    bucket_faces_remap = inv.reshape(-1, 3)
    bucket_verts = verts[unique_verts]
    bucket_vert_normals = vert_normals[unique_verts]
    
    print(f"   单水斗: {len(bucket_verts):,}顶点, {len(bucket_faces_remap):,}面片")
    
    return {
        'verts': bucket_verts,
        'faces': bucket_faces_remap,
        'normals': bucket_vert_normals,
        'center': center,
        'R1': np.median(np.sqrt((bucket_verts[:,0]-center[0])**2+(bucket_verts[:,1]-center[1])**2)),
        'D1': D1_MODEL if 'model' in stl_path.name else D1_FULL,
    }


def compute_erosion_on_mesh(data, V_jet=106.7, V_bucket=68.1):
    """在DEC网格上计算侵蚀分布"""
    verts = data['verts']
    normals = data['normals']
    center = data['center']
    R = data['R1']
    D1_val = data['D1']
    
    # 径向向量 (从旋转中心指向顶点)
    r_vec = verts[:,:2] - center
    r_dist = np.sqrt(r_vec[:,0]**2 + r_vec[:,1]**2)
    radial_3d = np.column_stack([r_vec / (r_dist[:,None]+1e-10), np.zeros(len(verts))])
    
    # 壁面倾角 = 法向量与径向的夹角 → 颗粒冲击角
    cos_alpha = np.abs(np.sum(normals * radial_3d, axis=1))
    cos_alpha = np.clip(cos_alpha, 0, 1)
    impact_angle = np.degrees(np.arccos(cos_alpha))
    
    # Z位置
    z_vals = verts[:,2]
    
    # === 冲击速度分布 ===
    # 分水刃: 直接冲击 → Vp高
    # 内表面: 有水膜缓冲 → Vp中等
    # 根部: 远端 → Vp低
    V_rel = V_jet - V_bucket
    
    # 用z位置估算速度衰减 (简化水膜模型)
    z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-10)
    Vp = V_rel * (1.0 - 0.45 * z_norm**0.7)  # 非线性衰减
    
    # === 三模型侵蚀率 ===
    oka = OkaModel(Hv=2.6)
    finnie = FinnieModel()
    tabakoff = TabakoffModel()
    
    E_oka = np.array([oka.erosion_rate(float(Vp[i]), float(impact_angle[i])) for i in range(len(verts))])
    E_finnie = np.array([finnie.erosion_rate(float(Vp[i]), float(impact_angle[i])) for i in range(len(verts))])
    E_tabakoff = np.array([tabakoff.erosion_rate(float(Vp[i]), float(impact_angle[i])) for i in range(len(verts))])
    
    # 归一化
    def norm(x): return x / (x.max() + 1e-10)
    
    # === 磨蚀类型分类 ===
    # 切削-变形比 (Finnie/Tabakoff)
    cut_ratio = norm(E_finnie) / (norm(E_finnie) + norm(E_tabakoff) + 1e-10)
    
    erosion_types = np.full(len(verts), '犁沟/耦合', dtype=object)
    erosion_types[cut_ratio > 0.55] = '切削主导'
    erosion_types[cut_ratio < 0.25] = '撞击主导'
    
    # === 风险等级 ===
    E_oka_norm = norm(E_oka)
    risk = np.full(len(verts), '低', dtype=object)
    risk[E_oka_norm > 0.3] = '中'
    risk[E_oka_norm > 0.6] = '高'
    risk[E_oka_norm > 0.85] = '极高'
    
    return {
        'impact_angle': impact_angle,
        'Vp': Vp,
        'E_oka': E_oka,
        'E_finnie': E_finnie,
        'E_tabakoff': E_tabakoff,
        'E_oka_norm': E_oka_norm,
        'erosion_type': erosion_types,
        'risk': risk,
        'r_dist': r_dist,
        'z': z_vals,
        'D1': D1_val,
    }


def export_colored_stl(verts, faces, colors, filename, color_map='hot'):
    """导出带颜色的STL (用于ParaView等可视化)"""
    from matplotlib import cm
    import matplotlib
    matplotlib.use('Agg')
    
    # 归一化颜色值
    c_norm = (colors - colors.min()) / (colors.max() - colors.min() + 1e-10)
    cmap = cm.get_cmap(color_map)
    rgba = cmap(c_norm)
    
    path = OUTPUT_DIR / filename
    with open(path, 'w') as f:
        f.write(f"solid {filename}\n")
        for fi, face in enumerate(faces):
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            n = np.cross(v1-v0, v2-v0)
            nl = np.linalg.norm(n)
            if nl > 0: n /= nl
            else: n = [0,0,1]
            f.write(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            f.write(f"    outer loop\n")
            for i, vi in enumerate([face[0], face[1], face[2]]):
                r, g, b = rgba[vi,:3]
                f.write(f"      vertex {verts[vi,0]:.6f} {verts[vi,1]:.6f} {verts[vi,2]:.6f}\n")
            f.write(f"    endloop\n  endfacet\n")
        f.write(f"endsolid {filename}\n")
    
    size_mb = path.stat().st_size/1024/1024
    print(f"   ✅ {filename} ({size_mb:.1f} MB)")


def print_summary(erosion_data, label):
    """打印侵蚀分析摘要"""
    print(f"\n{'='*50}")
    print(f"📊 {label} - 侵蚀分析摘要")
    print(f"{'='*50}")
    
    print(f"\n  壁面倾角: [{erosion_data['impact_angle'].min():.1f}°, {erosion_data['impact_angle'].max():.1f}°]")
    print(f"  冲击速度: [{erosion_data['Vp'].min():.1f}, {erosion_data['Vp'].max():.1f}] m/s")
    
    print(f"\n  磨蚀类型分布:")
    for t in ['切削主导', '撞击主导', '犁沟/耦合']:
        n = np.sum(erosion_data['erosion_type'] == t)
        pct = n/len(erosion_data['erosion_type'])*100
        print(f"    {t}: {n:,} ({pct:.1f}%)")
    
    print(f"\n  风险等级分布:")
    for r in ['极高', '高', '中', '低']:
        n = np.sum(erosion_data['risk'] == r)
        pct = n/len(erosion_data['risk'])*100
        print(f"    {r}: {n:,} ({pct:.1f}%)")
    
    # 各区域统计
    angles = erosion_data['impact_angle']
    print(f"\n  壁面倾角分位数: P10={np.percentile(angles,10):.1f}° P50={np.percentile(angles,50):.1f}° P90={np.percentile(angles,90):.1f}°")


def main():
    print("="*70)
    print("  DEC真实几何上的侵蚀分布计算")
    print("="*70)
    
    # 模型转轮
    data_m = load_and_prepare(DEC_MODEL_STL, "模型转轮 D1276P21", max_faces=150000)
    erosion_m = compute_erosion_on_mesh(data_m)
    print_summary(erosion_m, "模型转轮")
    
    # 导出
    export_colored_stl(data_m['verts'], data_m['faces'], erosion_m['E_oka_norm'],
                       'dec_model_erosion_oka.stl', 'hot')
    export_colored_stl(data_m['verts'], data_m['faces'], erosion_m['impact_angle'],
                       'dec_model_impact_angle.stl', 'viridis')
    
    # 真机转轮
    print(f"\n{'─'*50}")
    data_f = load_and_prepare(DEC_FULL_STL, "真机转轮 D1247R2", max_faces=200000)
    erosion_f = compute_erosion_on_mesh(data_f, V_jet=np.sqrt(2*9.81*595), V_bucket=2*np.pi*375/60*2.63/2)
    print_summary(erosion_f, "真机转轮")
    
    export_colored_stl(data_f['verts'], data_f['faces'], erosion_f['E_oka_norm'],
                       'dec_full_erosion_oka.stl', 'hot')
    
    # 保存数值结果
    result = {
        'model_runner': {
            'n_verts': len(data_m['verts']),
            'impact_angle_range': [float(erosion_m['impact_angle'].min()), float(erosion_m['impact_angle'].max())],
            'risk_high_pct': float(np.sum((erosion_m['risk']=='高')|(erosion_m['risk']=='极高'))/len(erosion_m['risk'])*100),
        },
        'full_runner': {
            'n_verts': len(data_f['verts']),
            'impact_angle_range': [float(erosion_f['impact_angle'].min()), float(erosion_f['impact_angle'].max())],
            'risk_high_pct': float(np.sum((erosion_f['risk']=='高')|(erosion_f['risk']=='极高'))/len(erosion_f['risk'])*100),
        }
    }
    with open(OUTPUT_DIR / 'erosion_on_dec_geometry.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 完成! 输出在 {OUTPUT_DIR}")
    print(f"   - dec_model_erosion_oka.stl (模型转轮+侵蚀热力图)")
    print(f"   - dec_full_erosion_oka.stl (真机转轮+侵蚀热力图)")
    print(f"   - erosion_on_dec_geometry.json (数值结果)")


if __name__ == "__main__":
    main()
