"""
Step 1b: 解析 Parasolid .x_t 头信息 + 提取关键统计数据
同时从公开文献重建金窝水电站的关键几何参数
"""
import re
from pathlib import Path
from collections import Counter

DATA_DIR = Path(r"B:\undertaking\学业\非专业相关\比赛\挑战杯\水轮机\DecPelton2024\DecPelton2024\DEC PELTON 2024-150MW级冲击式国产化转轮研发共享数据包")
MODEL_XT = DATA_DIR / "1-转轮水力开发数据" / "1.1-金窝模型转轮D1276P21_.x_t"
FULL_XT = DATA_DIR / "2-真机转轮三维实体数据" / "2.1-150MW级冲击式国产化真机转轮三维实体数据.x_t"

def parse_xt_file(xt_path: Path, label: str):
    """解析 Parasolid 文本文件的基本结构"""
    print(f"\n{'='*60}")
    print(f"📦 {label}: {xt_path.name}")
    print(f"   大小: {xt_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    with open(xt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 提取头部
    header_end = content.find('**END_OF_HEADER**')
    header = content[:header_end]
    
    # 解析头部键值对
    print(f"\n   📋 文件头:")
    for line in header.split('\n'):
        line = line.strip()
        if '=' in line and not line.startswith('**') and not line.startswith('T51'):
            key, _, val = line.partition('=')
            key = key.strip()
            val = val.strip().rstrip(';')
            if key and val:
                print(f"      {key}: {val}")
    
    # 提取实体部分
    body = content[header_end + len('**END_OF_HEADER**'):]
    
    # 统计实体类型
    # Parasolid 文本格式中，实体由特定模式标识
    # 查找常见的实体类型标记
    entity_patterns = {
        'BODY': r'\bBODY\b',
        'SURFACE': r'\bSURFACE\b| surface ',
        'boundary_mesh': r'boundary_mesh',
        'schema_embedding': r'schema_embedding',
        'index_map': r'index_map',
        'node_id': r'node_id',
        'SPLIN': r'SPLIN',  # Spline surface
        'NURBS': r'NURBS',
        'POINT': r'\bPOINT\b| point ',
        'CURVE': r'\bCURVE\b| curve ',
        'TRANSFORM': r'TRANSFORM',
        'ASSEMBLY': r'ASSEMBLY| assembly ',
        'INSTANCE': r'INSTANCE| instance ',
    }
    
    print(f"\n   📊 实体统计 (在完整文件中):")
    for entity, pattern in entity_patterns.items():
        count = len(re.findall(pattern, content, re.IGNORECASE))
        if count > 0:
            print(f"      {entity}: {count:,}")
    
    # 检查数据中的数字模式（可能是几何坐标）
    # 找浮点数的密度和范围
    numbers = re.findall(r'[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?', body[:200000])
    floats = []
    for n in numbers[:50000]:
        try:
            f = float(n)
            if abs(f) < 1e10:
                floats.append(f)
        except:
            pass
    
    if floats:
        print(f"\n   📏 数值统计 (前50K个数值):")
        print(f"      总数: {len(floats):,}")
        print(f"      范围: [{min(floats):.3f}, {max(floats):.3f}]")
        print(f"      均值: {sum(floats)/len(floats):.3f}")
        
        # 推测模型尺度
        abs_max = max(abs(f) for f in floats)
        print(f"      最大绝对值: {abs_max:.1f} → 推测模型尺寸约 {abs_max*2:.0f} mm")
    
    # 找 Z 值后的数字 (Z 通常标记实体计数)
    z_numbers = re.findall(r'Z(\d+)\s+(\d+)', content[:50000])
    if z_numbers:
        print(f"\n   🔢 Z标记计数:")
        for z in z_numbers[:5]:
            print(f"      Z{z[0]} = {int(z[1]):,}")


def literature_reference():
    """从公开文献中提取金窝水电站已知参数"""
    print(f"\n{'='*60}")
    print(f"📚 金窝水电站 - 公开文献参数")
    print(f"{'='*60}")
    
    params = {
        "电站名称": "田湾河梯级金窝水电站",
        "额定功率": "150 MW",
        "设计水头": "约580 m (据田湾河流域)",
        "额定转速": "约500 rpm",
        "转轮类型": "冲击式 (Pelton) 立轴",
        "喷嘴数": "6个",
        "水斗材料": "ZG0Cr13Ni4Mo (低碳马氏体不锈钢)",
        "转轮设计编号": "D1247R2 (真机) / D1276P21 (模型)",
        "模型转轮来源": "DEC PELTON 2024 共享数据",
        "真机转轮来源": "DEC 东方电机设计",
        "含沙量": "汛期平均 1.31 kg/m³ (文献)",
        "粒径范围": "0.001-0.23 mm (文献)",
    }
    
    for key, val in params.items():
        print(f"   {key}: {val}")
    
    print(f"\n   📖 关键参考文献:")
    refs = [
        "肖业祥 等 (2024) 水力发电学报 - 冲击式水轮机转轮水斗汛期泥沙磨损特性分析",
        "梁权伟, 肖业祥 等 (2021) 水电与抽水蓄能 - 含沙三相流与磨损特性",
        "Rai et al. (2020) Renewable Energy - IEC 62364 Pelton 解释",
        "Guo et al. (2020) Renewable Energy - 喷射机构含沙流与侵蚀",
        "Ge et al. (2023) CJME - 冲击式水轮机泥沙侵蚀综述",
    ]
    for i, ref in enumerate(refs, 1):
        print(f"   [{i}] {ref}")


def main():
    parse_xt_file(MODEL_XT, "模型转轮")
    parse_xt_file(FULL_XT, "真机转轮")
    literature_reference()
    
    print(f"\n{'='*60}")
    print(f"💡 下一步:")
    print(f"   1. 使用已知参数构建参数化水斗模型")
    print(f"   2. 实现 Oka/Finnie/Tabakoff 物理侵蚀模块")
    print(f"   3. 在参数化模型上验证侵蚀分布")
    print(f"   4. 等待 CAD Assistant 或手动转换获取真实几何")

if __name__ == "__main__":
    main()
