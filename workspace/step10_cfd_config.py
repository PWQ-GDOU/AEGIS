"""
Step 10: DEC文献CFD仿真配置 — 直接复用梁权伟/肖业祥团队的设置

来源:
[1] 肖业祥,郭豹,张瑾,梁权伟,刘洁. 冲击式水轮机含沙三相流与磨损特性数值研究. 水电与抽水蓄能. 2021(02).
[2] 李彦浩,朱逸林,肖业祥,刘洁,梁权伟,李海军. 冲击式水轮机转轮水斗汛期泥沙磨损特性分析. 水力发电学报. 2024,43(2).

本文件可直接用作 OpenFOAM case 的配置参考
"""
import json
from pathlib import Path

OUTPUT_DIR = Path(r"workspace/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DEC团队的完整CFD设置 (直接复用)
# ============================================================

CFD_CONFIG = {
    "文献来源": {
        "论文1": "肖业祥,郭豹,张瑾,梁权伟,刘洁 (2021) 水电与抽水蓄能",
        "论文2": "李彦浩,朱逸林,肖业祥,刘洁,梁权伟,李海军 (2024) 水力发电学报",
        "几何来源": "DEC PELTON 2024 — D1276P21/D1247R2 (金窝水电站)",
        "试验验证": "金窝水电站汛期实测泥沙数据"
    },
    
    # ========================================================
    # 多相流模型
    # ========================================================
    "多相流": {
        "模型": "VOF (Volume of Fluid)",
        "相": ["水 (主相)", "空气 (次相)", "沙粒 (离散相)"],
        "界面追踪": "Geo-Reconstruct 格式 (Fluent) / isoAdvector (OpenFOAM)",
        "表面张力": "0.072 N/m (水-气)",
        "接触角": "壁面接触角 90°"
    },
    
    # ========================================================
    # 湍流模型
    # ========================================================
    "湍流": {
        "模型": "SST-CC (Shear Stress Transport with Curvature Correction)",
        "OpenFOAM等效": "kOmegaSST",
        "入口湍流强度": "5%",
        "入口湍流粘度比": "10",
        "壁面处理": "自动壁面函数 (y+ < 1 需要低Re处理)"
    },
    
    # ========================================================
    # 颗粒追踪 (DPM = Discrete Phase Model)
    # ========================================================
    "颗粒追踪": {
        "模型": "DPM (离散相模型, 拉格朗日追踪)",
        "耦合方式": "单向耦合 (颗粒体积分数 << 1%)",
        "颗粒类型": "惰性颗粒 (inert)",
        "粒径分布": "Rosin-Rammler分布",
        "粒径范围": {"min": 0.001, "max": 0.23, "unit": "mm"},
        "中位粒径_d50": 0.05,  # mm (论文中的典型值)
        "颗粒密度": 2650.0,  # kg/m³ (石英砂)
        "颗粒形状": "球形假设",
        "浓度": 1.31,  # kg/m³ (金窝电站汛期实测)
        "质量流量": "根据入口流量 × 浓度计算",
        "DPM边界条件": {
            "入口": "escape (颗粒随流体进入)",
            "出口": "escape (颗粒流出)",
            "壁面": "reflect (反弹)",
            "反弹系数": "0.8 (法向) / 0.9 (切向) — 经验值"
        }
    },
    
    # ========================================================
    # 边界条件
    # ========================================================
    "边界条件": {
        "入口": {
            "类型": "总压入流 (Total Pressure Inlet)",
            "总压": "根据水头 H=595m → P=ρgH (真机) / 按相似准则缩放到模型",
            "水体积分数": 1.0,
            "泥沙注入": "从入口面均匀注入"
        },
        "出口": {
            "类型": "静压出口 (Static Pressure Outlet)",
            "相对压力": 0.0,
            "回流": "允许, 水体积分数=0 (空气)"
        },
        "壁面": {
            "类型": "无滑移壁面 (No-Slip Wall)",
            "粗糙度": "光滑壁面假设",
            "材料": "ZG0Cr13Ni4Mo 不锈钢"
        },
        "动静交界面": {
            "类型": "滑移网格 (Sliding Mesh) / MRF",
            "转速": "模型: 根据n11=40rpm换算 / 真机: 375 rpm"
        },
        "周期面": {
            "类型": "旋转周期边界 (Rotational Periodic)",
            "周期数": 21  # 水斗数
        }
    },
    
    # ========================================================
    # 求解器设置
    # ========================================================
    "求解器": {
        "算法": "Coupled (压力-速度耦合求解)",
        "OpenFOAM等效": "PIMPLE (PISO+SIMPLE混合)",
        "离散格式": "二阶迎风格式 (2nd Order Upwind)",
        "压力插值": "PRESTO! (Fluent) / 线性+限制器 (OpenFOAM)",
        "体积分数": "CICSAM (Fluent) / MULES (OpenFOAM)",
        "梯度": "Least Squares Cell Based / Gauss linear",
        "瞬态格式": "一阶隐式 / Euler",
        "时间步长": 7.89e-5,  # s (论文值)
        "库朗数": {"最大": 1.0, "平均": 0.5},
        "每时间步迭代": 20
    },
    
    # ========================================================
    # 计算域 & 网格
    # ========================================================
    "计算域": {
        "分区": ["喷射机构区 (喷嘴+喷针)", "射流区 (自由射流)", "静止域", "转轮域 (含水斗)"],
        "总网格量": "~1500万节点",
        "网格类型": "非结构化为主, 射流段为结构化",
        "局部加密": "水气交界面 (VOF界面) + 水斗表面",
        "壁面y+": "< 5 (近壁面加密)",
        "OpenFOAM网格工具": "snappyHexMesh (从STL生成) 或 blockMesh + refineMesh",
        "STL输入": "dec_model_runner.stl (模型转轮) / dec_full_runner.stl (真机转轮)"
    },
    
    # ========================================================
    # 侵蚀模型 (后处理)
    # ========================================================
    "侵蚀模型": {
        "DEC使用": "Mansouri模型 + Oka冲击角度函数",
        "等效实现": "自定义 Field Function / Python后处理",
        "所需壁面数据": ["颗粒冲击速度 Vp", "冲击角度 α", "颗粒质量通量 ṁp"],
        "后处理": "提取壁面碰撞数据 → Python计算Oka侵蚀率 → 映射回网格渲染"
    },
    
    # ========================================================
    # 工况参数 (金窝电站)
    # ========================================================
    "金窝电站参数": {
        "额定水头": "595.0 m",
        "额定功率": "143.6 MW (真机) / 对应模型工况",
        "额定转速": "375.0 rpm (真机)",
        "转轮直径": "2.63 m (真机) / 312.5 mm (模型)",
        "水斗数": 21,
        "喷嘴数": 6,
        "射流直径": "0.24 m (真机)",
        "含沙量": "1.31 kg/m³ (汛期平均)",
        "泥沙中值粒径": "0.05 mm",
        "泥沙成分": "石英砂为主 (>70%) + 长石 + 云母"
    },
    
    # ========================================================
    # OpenFOAM 具体配置映射
    # ========================================================
    "OpenFOAM配置映射": {
        "system/controlDict": {
            "application": "interFoam",
            "startFrom": "startTime",
            "startTime": 0,
            "stopAt": "endTime",
            "endTime": 0.1,  # s, 根据需要的物理时间调整
            "deltaT": 7.89e-5,
            "writeInterval": 0.001,
            "maxCo": 1.0,
            "maxAlphaCo": 1.0
        },
        "constant/turbulenceProperties": {
            "simulationType": "RAS",
            "RASModel": "kOmegaSST"
        },
        "constant/transportProperties": {
            "phases": ["water", "air"],
            "water": {"rho": 998.2, "nu": 1.004e-6},
            "air": {"rho": 1.225, "nu": 1.48e-5},
            "sigma": 0.072
        },
        "constant/g": {
            "value": [0, 0, -9.81]
        },
        "颗粒注入": {
            "使用": "OpenFOAM Lagrangian库 (kinematicparcel)",
            "/constant/kinematicCloudProperties": {
                "solution": {"active": True, "coupled": False},
                "particleForces": {"gravity": True, "drag": {"type": "sphereDrag"}},
                "injectionModels": {
                    "patchInjection": {
                        "patchName": "inlet",
                        "massFlowRate": "根据浓度计算",
                        "parcelBasisType": "mass",
                        "sizeDistribution": {"type": "RosinRammler", "minValue": 1e-6, "maxValue": 2.3e-4, "d": 5e-5, "n": 1.5}
                    }
                }
            }
        }
    }
}


def main():
    print("="*70)
    print("  DEC文献CFD仿真配置文档")
    print("="*70)
    
    # 保存JSON
    config_path = OUTPUT_DIR / "cfd_config_from_dec_literature.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(CFD_CONFIG, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ CFD配置文档: {config_path}")
    
    # 打印关键设置摘要
    print(f"\n{'='*50}")
    print(f"  关键设置速查")
    print(f"{'='*50}")
    print(f"""
  求解器:      interFoam (VOF气液两相流)
  颗粒追踪:     OpenFOAM Lagrangian (kinematicparcel)
  湍流模型:     kOmegaSST (SST-CC等效)
  转速:         375 rpm (真机)
  时间步长:     7.89×10⁻⁵ s
  含沙量:       1.31 kg/m³ (汛期实测)
  粒径D50:      0.05 mm (Rosin-Rammler分布)
  侵蚀模型:     Oka (后处理, Python)
  几何:         dec_model_runner.stl / dec_full_runner.stl
  
  📖 下一步:
     1. 安装 WSL2 + Ubuntu
     2. 安装 OpenFOAM v2306
     3. 复制本配置搭建 case
     4. 生成网格 (snappyHexMesh)
     5. 运行仿真
     6. 后处理提取壁面数据 → Oka侵蚀计算
""")


if __name__ == "__main__":
    main()
