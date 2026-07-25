# AEGIS 项目会话上下文

> 保存时间: 2026-07-21
> 项目: Anti-Erosion Grey-box Intelligence System
> 比赛: 挑战杯2026 - 大容量冲击式水轮机关键过流部件精准抗磨防护技术研究
> 发榜单位: 东方电气集团东方电机有限公司 (DEC)

---

## 一、项目概述

金窝水电站150MW冲击式(Pelton)水轮机转轮的泥沙磨蚀预测。三个关键过流部件：喷嘴口环、喷针、水斗。

**核心任务**：构建灰盒退化模型，预测水斗表面每个位置的磨蚀类型、磨蚀速率、退化时间。

**技术路线**：物理模型(Oka/Finnie/Tabakoff) 主导 + GPR/ML 残差修正 → 空间分布预测 + 时间退化推演。

---

## 二、已完成的全部工作

### 1. 赛事分析
- 解析了比赛PDF (CS-202606)
- 评分标准: 主观60分(完成度20+思路20+工程性20) + 客观40分(机理识别20+空间预测误差<5% 20)
- 时间线: 5-9月研发 → 9月15日提交 → 11月终审

### 2. 几何模型处理
- DEC提供了两个Parasolid .x_t文件: D1276P21(模型转轮, D1=312.5mm) 和 D1247R2(真机转轮, D1=2630mm)
- 通过Siemens NX导出为STL (dec_model_runner.stl 14MB, dec_full_runner.stl 88MB)
- 直接使用DEC真实STL作为计算基底 → 0%几何误差
- 双模型缩放一致性验证: D1误差0.3%, B2误差0.4%, 缩放比8.416

### 3. DEC文献逆向
- 关键论文: 肖业祥/梁权伟(2021, 2024) → DEC使用 Mansouri模型+Oka角度函数
- DEC专利 CN116127636A: 2簇正交Bézier样条划分水斗
- IEC 62364 中国国标采纳中
- CFD设置: VOF + DPM + SST-CC湍流, ~1500万节点, Δt=7.89×10⁻⁵s
- 金窝电站参数: 含沙量1.31 kg/m³, D50=0.05mm, 石英砂

### 4. 物理侵蚀模型实现
- Oka模型(2005): 核心模型, 全角度覆盖, 考虑靶材硬度Hv和粒径Dp
- Finnie模型(1960): 切削磨损, 低冲击角适用
- Tabakoff模型(1975): 切削+变形分离
- 代码: `workspace/step2_physics_model.py`

### 5. 参数化水斗重建
- Bézier曲线反拟合DEC几何
- B-spline拟合达到Chamfer=3.4%D1, P95=7.1%D1
- 代码: `workspace/step4_parametric_bucket.py`, `step8b_direct_fit.py`, `step8c_bspline_fit.py`

### 6. OpenFOAM CFD仿真
- 服务器: 100.92.47.51 (80核, 125GB RAM), 用户user, 密码hukee123456
- WSL本地 + 服务器OpenFOAM 1912
- 成功运行层流interFoam: 320K单元blockMesh, 1174步, t=0.01s, 13分钟
- Case位置: 本地`workspace/of_case/pelton_jet/`, 服务器`/data/disk3/of_workspace/pelton_jet/`
- 长时仿真可通过SSH启动: `ssh user@100.92.47.51` → `cd /data/disk3/of_workspace/pelton_jet`
- CFD设置完全基于DEC文献

### 7. 侵蚀计算
- 在DEC真实STL上计算Oka/Finnie/Tabakoff侵蚀分布
- 产出3D侵蚀热力图STL (可导入NX/ParaView)
- 双模型(模型转轮+真机转轮)结果一致: 33%高中风险区, P50≈68°冲击角
- 代码: `workspace/step9_erosion_on_dec.py`

### 8. GitHub仓库
- 仓库: https://github.com/PWQ-GDOU/AEGIS (私密)
- Token: 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
- 86个文件已推送

---

## 三、关键文件位置

### 代码 (workspace/)
| 文件 | 功能 |
|:--|:--|
| step2_physics_model.py | Oka/Finnie/Tabakoff 物理模型 |
| step4_parametric_bucket.py | Bézier参数化水斗 |
| step9_erosion_on_dec.py | 在DEC真实几何上计算侵蚀 |
| step10_cfd_config.py | DEC文献CFD配置文档生成 |
| step11_generate_of_case.py | OpenFOAM case生成 |

### 输出 (workspace/output/)
| 文件 | 说明 |
|:--|:--|
| dec_model_runner.stl | DEC模型转轮STL (14MB) |
| dec_full_runner.stl | DEC真机转轮STL (88MB) |
| dec_model_erosion_oka.stl | 3D侵蚀热力图 |
| dec_full_erosion_oka.stl | 真机侵蚀热力图 |
| cfd_config_from_dec_literature.json | 完整CFD配置 |
| final_validation_report.json | 双模型验证报告 |
| erosion_on_dec_geometry.json | 侵蚀计算结果 |

### OpenFOAM (workspace/of_case/pelton_jet/)
| 文件 | 说明 |
|:--|:--|
| system/controlDict | 求解控制 (interFoam, Δt=1e-6, Co<0.3) |
| system/fvSchemes | 离散格式 (VOF + kOmegaSST) |
| system/fvSolution | 求解器设置 |
| system/blockMeshDict | 背景网格 (600mm³ domain) |
| constant/triSurface/dec_model_runner.stl | 导入的DEC几何 |
| 0/U, 0/p_rgh, 0/alpha.water | 初始条件 |

---

## 四、服务器连接

```
SSH: ssh user@100.92.47.51
密码: hukee123456
远程桌面: 100.92.47.51:3389 (需Tailscale: 2013252515@qq.com)

仿真位置: /data/disk3/of_workspace/pelton_jet/
磁盘空间: /data/disk3 有1017GB空闲

启动仿真:
  source /usr/share/openfoam/etc/bashrc
  cd /data/disk3/of_workspace/pelton_jet
  interFoam > run.log 2>&1 &
```

---

## 五、下一步任务：灰盒退化模型

### 模型架构
```
输入层: 流场参数(Vp,α,ṁp) + 几何特征(κ,法向) + 材料参数(Hv)
   ↓
物理模块(主导): Oka模型 → E_phys (决定磨蚀类型+大格局)
   ↓
ML模块(辅助): GPR测地距离核 → ΔE ± σ (修正局部+置信区间)
   ↓
退化积分: E_final × t → 壁厚演变 → 剩余寿命
   ↓
输出层: 磨蚀类型分区图 + 风险热力图 + 退化曲线
```

### 需要完成的具体任务
1. 整理现有物理模块输出作为灰盒的"白盒部分"
2. 实现GPR残差修正模块（测地距离核）
3. 时间退化积分（E × t → 壁厚演变）
4. 参数敏感性分析（含沙量、水头变化对寿命影响）
5. 与DEC文献描述的磨损模式交叉验证
6. 撰写完整技术方案报告

### 关键技术决策
- 物理主导（不是DL主导）：因为Oka公式已有30年验证，DL补盲区
- GPR选型：小样本友好+自带置信区间+测地距离核实现空间平滑
- 不做扩散模型生成图：侵蚀预测是回归任务不是生成任务

---

## 六、DEC数据约束

知识产权声明:
- 仅限科学研究使用
- 后续改进的知识产权由DEC与改进方共享
- 禁止分享给第三方
- 发表需注明: "该研究受到大型冲击式水轮机研究项目 DEC PELTON 2024 的支持"

---

## 七、灰盒退化模型（2026-07-24 完成搭建）

五层管道全部实现并跑通，代码 step10–step14，输出在 workspace/output/。

### 架构与文件
| 层 | 文件 | 作用 | 状态 |
|:--|:--|:--|:--|
| L0 数据管道 | step10_unified_pipeline.py | 统一参数、修 5 处 bug、逐顶点 NPZ | ✅ |
| L2 GPR 修正 | step11_gpr_correction.py | 物理先验均值 GPR、乘性残差、置信区间 | ✅ |
| L3 退化积分 | step12_degradation.py | 壁厚减薄速率、剩余寿命 T_RUL | ✅ |
| L4 敏感性 | step13_sensitivity.py | C_s/H/Dp/Hv 单因素扫描 | ✅ |
| L5 报告 | step14_report.py | 热力图 STL、退化曲线、汇总 JSON | ✅ |

### step10 修复的 5 处 bug
- F1 水斗数统一 21（step2 曾用 22）
- F2 Tabakoff 角度 bug（step9:55 弧度被重复乘 π/180，切削项偏小约 57×）
- F3 Oka 粒径 Dp 改为可注入（原硬编码 150μm，阻碍敏感性分析）
- F4 速度模型统一为 Z 坐标非线性衰减
- F5 水头统一 H=595m（与 truth_full 一致）

### 关键设计决策
- GPR 无实测标签，采用 P1 策略：标签 = log(三模型秩加权融合 / Oka 秩)，均值函数=0，学习偏差
- 非负约束：E_final = E_oka × exp(f_GPR) 恒正
- 稀疏近似：按风险分层采样 3000 训练点（全 15.4 万顶点），训练 ~82s，预测 ~27s
- 退化用线性恒速率模型 h(t)=h0−(dh/dt)·t（v1）

### 关键结果
- 跨尺度 Spearman ρ=1.0（模型↔真机风险排序完全一致）
- GPR 保留集 Spearman ρ=0.714，E_final vs Oka ρ=0.885
- 全局最短 T_RUL=4h（保守下界 2h），位于最高冲击速度点 v#142984
- 敏感性：含沙量影响最大（1.3–13h，10×），水头 3.1×，粒径/硬度较弱

### 输出文件（output/）
vertices_model.npz, vertices_full.npz, vertices_full_gpr.npz, gpr_model.pkl,
rul_map.npz, degradation_curves.npz, critical_points.json,
sensitivity_curves.json, sensitivity_dashboard.png,
erosion_final_heatmap.stl, rul_heatmap.stl, uncertainty_heatmap.stl,
degradation_curves.png, report_summary.json
（各 *_summary.json 保存分层指标）

### 待确认 / 后续
1. h0=30mm、h_min=15mm 为工程缺省，需 DEC 图纸确认
2. GPR 标签待升级为 P0：需检修实测壁厚/三维扫描数据
3. 测地距离核尚未实现（当前欧氏 RBF），后续升级
4. 高保真 CFD（VOF+DPM+SST-CC）替换简化速度模型
5. 退化模型待升级为分段动态（磨损改变几何→重算冲击角）
6. 比赛"空间误差<5%"仍需实测数据才能定量验证

---

## 八、DEC 数据包关键参数（2026-07-24 读取）

### 电站水力参数（金窝水电站，来源：1.2 模型试验报告）

| 参数 | 值 |
|:--|:--|
| 额定出力 | 143.60 MW |
| 最大出力 | 160.00 MW |
| 额定水头 Hr | 595.00 m ✅（代码已用） |
| 最大水头 Hmax | 619.80 m |
| 加权水头 Hcp | 608.00 m |
| 额定流量 Qr | 27.00 m³/s |
| 额定转速 | 375 r/min ✅ |
| 真机 D1 | 2630 mm ✅ |
| 喷嘴数 | 6 |
| 水斗数 | 21 ✅ |
| 射流直径 d0 | 0.24 m |
| 模型试验水头 Hm | 103.7 m |
| 模型转速 | 1307 r/min |
| 缩放比 λL | 8.416 ✅ |

### 速比校验（step18 完成）

n11 定义：IEC 60193，`n11 = n·D1/√H`

| 校验项 | 计算值 | DEC 实测 | 误差 |
|:--|:--|:--|:--|
| 模型 n11（1307rpm,0.3125m,103.7m） | 40.11 | 41.4（最优） | 3% |
| 真机额定转速（n11=41.4 反推） | 384 rpm | 375 rpm | 2.4% |
| 射流速度 Cv√(2gH)，Cv=0.97 | 104.8 m/s | — | — |
| 6 喷嘴总流量 | 28.45 m³/s | 27.0 m³/s | 5.4% |
| 最优速比 φ=u1/V_jet | 0.493 | 理论 ~0.46 | 合理 |

结论：金窝电站额定工况运行在最优速比区，step10 速度模型在额定点准确。

### 现场动应力数据（来源：3.1，4#水斗根部，AI5-04，1000 Hz）

| 工况 | σ_mean (MPa) | σ_a (MPa) | f_dom (Hz) |
|:--|:--|:--|:--|
| 额定空转 | 22.8 | 0.9 | 6.3 |
| 30 MW | 24.5 | 8.9 | 12.7 |
| 40 MW | 24.9 | 11.9 | 12.7 |
| 50–90 MW | 25–28 | 7.8–14.2 | 37.2 |
| 100–140 MW | 28–30 | 9.2–14.5 | 37.2 |

主频 37.2 Hz = 6 喷嘴 × 转频（375/60=6.25 Hz × 6 ≈ 37.5 Hz），物理合理。

### 疲劳-磨蚀联合寿命（step18 结论）

- 材料：ZG0Cr13Ni4Mo，E=206 GPa，ν=0.3
- 保守疲劳极限：150 MPa（Goodman 修正，安全系数 2）
- 实测 σ_a = 9–15 MPa << 150 MPa → 纯疲劳寿命 ~2261 万小时
- **结论：水斗主控失效机制是磨蚀，不是疲劳**（与工程实践一致）
- 联合寿命与纯磨蚀寿命几乎相同（疲劳修正 <0.01%）

### 能量试验最优效率（来源：1.3）

| 喷嘴数 | 最高效率 | n11 (r/min) | Q11 (L/s) |
|:--|:--|:--|:--|
| 1 | 89.55% | 40.43 | 352 |
| 2 | 91.32% | 40.40 | 335 |
| 3 | 91.42% | 41.30 | 334 |
| 4 | 91.55% | 40.20 | 330 |
| 6 | **91.72%** | 41.40 | 302 |

### 新增模块（step15–step18）

| 文件 | 功能 |
|:--|:--|
| step15_geodesic_gpr.py | 测地距离核 GPR（网格图+Dijkstra），保留集 ρ=0.60 |
| step16_dynamic_degradation.py | 分段动态退化，T_RUL 比值中位 1.011，P5=0.25，P95=1.47 |
| step17_p0_scan_labels.py | P0 扫描配准+标签提取，自检 MAE=0.085mm |
| step18_fatigue_erosion_rul.py | 疲劳-磨蚀联合寿命，读取 DEC 实测动应力数据 |


