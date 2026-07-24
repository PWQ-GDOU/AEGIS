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
