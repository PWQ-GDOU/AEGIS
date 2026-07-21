"""
Step 1: 将 Parasolid .x_t 文件转换为 STL 面网格
尝试多种读取方式：gmsh → FreeCAD → 在线转换
"""
import os
import sys
from pathlib import Path

DATA_DIR = Path(r"B:\undertaking\学业\非专业相关\比赛\挑战杯\水轮机\DecPelton2024\DecPelton2024\DEC PELTON 2024-150MW级冲击式国产化转轮研发共享数据包")
OUTPUT_DIR = Path(r"B:\undertaking\学业\非专业相关\比赛\挑战杯\水轮机\workspace\output")

MODEL_RUNNER_XT = DATA_DIR / "1-转轮水力开发数据" / "1.1-金窝模型转轮D1276P21_.x_t"
FULL_RUNNER_XT = DATA_DIR / "2-真机转轮三维实体数据" / "2.1-150MW级冲击式国产化真机转轮三维实体数据.x_t"


def method_gmsh(xt_path: Path, stl_path: Path):
    """方法1: 使用 gmsh (内置 OpenCASCADE) 读取 Parasolid"""
    import gmsh
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 3)
    
    print(f"\n  [gmsh] 尝试读取 Parasolid: {xt_path.name}")
    
    try:
        # gmsh 使用 OpenCASCADE kernel 读取 CAD 格式
        # open() 会自动检测格式
        gmsh.open(str(xt_path))
        print(f"  [gmsh] ✅ 文件打开成功!")
        
        # 获取实体信息
        entities = gmsh.model.getEntities()
        print(f"  [gmsh] 实体数量: {len(entities)}")
        
        for dim, tag in entities[:5]:
            name = gmsh.model.getEntityName(dim, tag) if dim >= 0 else "N/A"
            print(f"    dim={dim}, tag={tag}, name={name}")
        if len(entities) > 5:
            print(f"    ... 还有 {len(entities)-5} 个实体")
        
        # 生成 2D 网格 (表面网格)
        print(f"  [gmsh] 生成表面网格...")
        gmsh.model.mesh.generate(2)  # 2 = 表面网格
        
        # 获取网格统计
        node_types = gmsh.model.mesh.getNodeTypes()
        elem_types = gmsh.model.mesh.getElementTypes()
        print(f"  [gmsh] 节点类型: {node_types}")
        print(f"  [gmsh] 单元类型: {elem_types}")
        
        nodes = gmsh.model.mesh.getNodes()
        print(f"  [gmsh] 节点数: {len(nodes[0])}")
        
        # 导出 STL
        print(f"  [gmsh] 导出 STL...")
        gmsh.write(str(stl_path))
        print(f"  [gmsh] ✅ STL 导出完成: {stl_path.name}")
        
        gmsh.finalize()
        return True
        
    except Exception as e:
        print(f"  [gmsh] ❌ 失败: {e}")
        try:
            gmsh.finalize()
        except:
            pass
        return False


def method_freecad(xt_path: Path, stl_path: Path):
    """方法2: 使用 FreeCAD (如果可用)"""
    try:
        import FreeCAD
        import Mesh
        import Part
        
        print(f"\n  [FreeCAD] 尝试读取 Parasolid...")
        
        # FreeCAD 可以导入 Parasolid
        shape = Part.Shape()
        shape.importX_t(str(xt_path))  # 可能有不同的 API
        
        # 或者使用:
        # Part.read(str(xt_path))
        
        print(f"  [FreeCAD] ✅ 读取成功!")
        
        # 生成网格
        mesh = Mesh.Mesh()
        mesh.addFacets(shape.tessellate(0.5))
        Mesh.export([mesh], str(stl_path))
        
        print(f"  [FreeCAD] ✅ STL 导出完成")
        return True
        
    except ImportError:
        print(f"  [FreeCAD] FreeCAD 未安装")
        return False
    except Exception as e:
        print(f"  [FreeCAD] ❌ 失败: {e}")
        return False


def method_manual_parse(xt_path: Path):
    """方法3: 手动解析 Parasolid 文本格式获取基本信息"""
    print(f"\n  [手动解析] 分析 {xt_path.name} 的头部信息...")
    
    with open(xt_path, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.read(5000)
    
    info = {}
    for line in header.split('\n'):
        line = line.strip()
        if line.startswith('**PART1;'):
            continue
        if line.startswith('**'):
            continue
        if '=' in line and not line.startswith('T51'):
            # 尝试提取键值对
            parts = line.split('=', 1)
            if len(parts) == 2:
                info[parts[0].strip()] = parts[1].strip().rstrip(';')
    
    print(f"  [手动解析] 文件头信息:")
    for key, val in info.items():
        print(f"    {key}: {val}")
    
    # 查找实体数量信息
    with open(xt_path, 'r', errors='ignore') as f:
        content = f.read()
    
    # 查找数字模式来估算实体数量
    import re
    
    # 找 GID 或类似模式
    surface_count = len(re.findall(r' surface ', content[:100000], re.IGNORECASE))
    body_count = len(re.findall(r' body ', content[:100000], re.IGNORECASE))
    
    print(f"  [手动解析] 估计:")
    print(f"    'surface' 关键词出现次数(前100K): {surface_count}")
    print(f"    'body' 关键词出现次数(前100K): {body_count}")
    
    return info


def analyze_stl(stl_path: Path):
    """分析 STL 文件"""
    import trimesh
    import numpy as np
    
    print(f"\n--- STL 分析: {stl_path.name} ---")
    mesh = trimesh.load(str(stl_path))
    
    if isinstance(mesh, trimesh.Scene):
        print(f"场景包含 {len(mesh.geometry)} 个几何体")
        meshes = []
        for key, geom in mesh.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            print("❌ 未找到网格!")
            return None
    
    print(f"顶点: {len(mesh.vertices):,}")
    print(f"面片: {len(mesh.faces):,}")
    print(f"水密: {mesh.is_watertight}")
    print(f"包围盒: {mesh.extents}")
    print(f"体积: {mesh.volume:.0f} mm³" if mesh.is_watertight else "体积: N/A")
    
    return mesh


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    xt_path = MODEL_RUNNER_XT
    stl_path = OUTPUT_DIR / "model_runner_D1276P21.stl"
    
    if not xt_path.exists():
        print(f"❌ 文件不存在: {xt_path}")
        return
    
    # 先手动解析获取信息
    method_manual_parse(xt_path)
    
    # 尝试 gmsh
    success = method_gmsh(xt_path, stl_path)
    
    if success:
        mesh = analyze_stl(stl_path)
    else:
        print("\n⚠️ gmsh 转换失败，尝试 FreeCAD...")
        # 不尝试 FreeCAD (可能未安装)
        print("请手动使用 CAD Assistant 转换: https://www.opencascade.com/products/cad-assistant/")
        print("或使用 Siemens NX 导出 STL")


if __name__ == "__main__":
    main()
