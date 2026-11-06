# Step30 水轮机磨蚀预测项目交接文档

> 用途：在新对话中恢复当前工程上下文。
> 阅读后须先运行 `python -m pytest -q` 和各模块 `--selftest`，以实际结果为准，不要依赖本文档中的历史计数。
> 生成日期：2026-08-03
> 工作区：`workspace`
> Git 分支：`fix/strict-collision-open-cfd`

---

## 1. 当前状态

### 1.1 生产顶层状态（不变）

```text
official_reference_gate  = false
official_accuracy_gate   = false
production status        = blocked
```

当前已保存 DEC PELTON 2024 官方资料包，包含模型/真机 Parasolid CAD、模型水力试验报告与水力数据、现场动应力实测数据及说明；但当前包没有可核验的官方 FEM/CFD solver case/result 或侵蚀空间分布参考，且尚未有 trusted out-of-band SHA-256 pin。`official_reference_gate` 因此仍保持 `false`，不代表“没有官方资料”。

在官方参考对象、完整身份和 trusted pin 未完成绑定前，任何正式 official acceptance 必须保持 `blocked`。

### 1.2 已通过的验证模块（截至 2026-08-03）

| 模块 | 自测状态 | 关键数值 |
|---|---|---|
| `step30_urans_validation.py` | PASS | maxCo=0.499，无 fatal/nonfinite |
| `step30_hydraulic_validation.py` | PASS | 6 嘴效率误差 2.24%（<5%），5/5 工况 |
| `step30_dec_stress_validation.py` | PASS | 见 §1.3 |
| `step30_erosion_spatial_map.py` | PASS | Spearman=0.692，430 hits，0.57% 覆盖 |
| `step30_duty_cycle_erosion.py` | PASS | 热区 IoU（额定 vs 全工况）=1.000 |
| `step30_fem_stress_field.py` | PASS | 140 MW 预测误差 1.2% |
| `step30_official_reference_simulation.py` | PASS | 软件自测；DEC 官方资料包已存在，但当前无可复现官方 FEM/CFD 结果包或侵蚀空间参考 |

### 1.3 动应力验证结果（严格竞赛规则 vs 工程容差）

**竞赛严格规则（相对误差 ≤ 5%，唯一计分依据）：**

- 静应力均值：13/13 通过，最大相对误差 1.32%
- 动幅：**12/13 通过**；130 MW 相对误差 ≈5.69%，**严格失败**
- 整体严格结论：**12/13 通过，130 MW 为严格失败**

**可选工程容差（不纳入赛事评分）：**

- 定义：相对误差 ≤ 5%，或绝对幅值误差 ≤ 1.2 MPa
- 1.2 MPa 是项目自定义工程阈值，**不是 DEC 仪器噪声下限、±1–2% FS 精度规范或测量不确定度**
- DEC 3.2/3.3 不支持任何此类噪声下限或满量程精度说法
- 工程容差结论：13/13 通过（含 130 MW，绝对误差 0.70 MPa <1.2 MPa）

**重要：工程容差通过不等于赛事评分通过。130 MW 在赛事严格规则下是失败。**

额定空转幅值：标记为 `not_evaluated`（离心标定点，不评估幅值）。

### 1.4 侵蚀空间场状态

- 首选数据源：`particle_hits_cfd.npz`（simpleFoam，430 hits，Spearman=0.692）
- 旧 URANS 5000-parcel 诊断：181 碰撞，覆盖约 0.074%，但为 v1、`n_injected=2000` 且未绑定 UMean solver semantics，不进入正式收敛矩阵
- 新 immutable 5k/seed=42 pilot（strict_v2、UMean、dt=1e-5）：177 碰撞、78 个首碰 parcel、159 个命中父面、覆盖 0.0726%，耗时 28m34s
- 新 pilot 的侵蚀贡献 ESS 仅 16.5，最大 parcel 占 12.26%，Top-5 占 46.2%，Top-10 占 66.7%；普通 number-sampling 未收敛，因此转入无偏 `mass_stratified_v1` PSD 抽样
- 已完成 5k/10k/20k × 5 seeds（42/141421/161803/271828/314159）共 15 个 immutable strict_v2/UMean 质量分层运行；所有 NPZ/summary/validation SHA-256 与各自 particle manifest 一致，fallback=0
- 最终 v3 收敛报告：`output/step30_particle_convergence_5k_10k_20k_mass_stratified_v3.json`，内容 SHA-256=`af46b6427c64a550cea3c0cdb0e6641e446ee696241ff59b4a35f12939484b1c`，状态为数值 `failed`（证据合同有效，不是 blocked）
- v3 额外重算上级 artifact-set manifest，绑定 evaluator SHA-256=`6dda83ef5a00ebe312f5cafbc9eaa9b32c7a34dc1dc93d3827a07b57b1f2fc12` 和合同 SHA-256=`94e2b7b46d9cb16cb8b7c4c17edcad188822a9fafc1a8d6203260e4583a2b58c`；v1/v2 为历史中间报告，不覆盖、不作为最终结论
- 报告和每个 run manifest 中的 tracker implementation hash 均绑定运行时源码字节；后续源码格式整理或修订不得倒写、冒充或覆盖这 15 个历史 immutable run
- 20k 已通过 pooled ESS=740.34、pooled 最大 parcel 份额=0.595%、首碰质量/速度/角度经验 seed 区间半宽、热点 IoU 中位数=1.0；未通过目标侵蚀质量经验 seed 区间半宽=12.23%、Spearman 中位/P10=0.516/0.403、10k→20k 最大质量变化=32.74%、JS=0.0601
- 20k 是预注册最大尺度，停止扩到 40k；下一步应调查碰撞/空间估计不稳定性，而不是机械增加粒子。经验 seed 区间是描述性 2.5/97.5 分位范围，不是总体均值置信区间
- 动应力精度和粒子统计收敛均不证明侵蚀空间精度；这些是独立证据问题

### 1.5 独立侵蚀空间真值链（代码已就绪，真实数据仍缺失）

已新增并通过端到端回归的合同链：

```text
protocol (冻结 baseline + worn 源)
  -> prediction commitment (揭盲前模型承诺 + topology identity)
  -> truth (扫描/稀疏测厚 parent-face 真值)
  -> reveal receipt
  -> replay_independent_erosion_acceptance
```

- `step30_independent_erosion_protocol.py`：只允许 measured 源；split 只由预注册身份和固定 seed 决定，标签不能影响分区；同一 asset/service episode 不得跨组；阈值和 90% hotspot 定义冻结。
- `step30_independent_erosion_truth.py`：支持前后表面扫描和稀疏测厚；单位统一为 mm；拒绝 synthetic/DEC/CFD、未知单位、负磨损、空有效集和源哈希不一致；强制 topology-v3、artifact_set、transform 和 parent-face identity。
- `step30_measurement_acceptance.py`：`replay_independent_erosion_acceptance` 必须校验 protocol/commitment/truth/receipt、外部 trusted commitment SHA-256，并从原始 baseline/worn 源重建 truth；合法数值超阈值为 `failed`，合同/哈希/来源问题为 `blocked`。
- 独立实测数值通过只产生 `independent_accuracy_gate=true`；`official_accuracy_gate=false`、`official_reference_gate=false`，且 `production_acceptance_status=blocked`。DEC 官方资料包本身不能替代未提供的官方侵蚀空间参考或可复现 FEM/CFD solver result；在参考对象身份和 trusted organizer pin 到位前，不得宣称正式官方 acceptance 通过。
- 当前仓库没有真实扫描、测厚或独立材料试验文件，因此真实执行仍必须输出 `blocked`；现有测试是 synthetic fixture，只验证软件合同和 fail-closed 行为。

---

## 2. 条件评分估计

以下估计基于当前已完成的技术证据。评委裁量权无法量化，实际得分可能偏离。

| 评分项（各 20 分） | 当前估计 | 主要依据 | 主要缺口 |
|---|---|---|---|
| 设计方案完成程度 | 14–16 | CFD+粒子+侵蚀+应力链路，以及扫描/材料/CFD/FEM/防护工具合同已就绪 | 无真实水斗扫描、同材料试验、求解结果和完整防护收益证据 |
| 研发思路合理性 | 15–17 | 水力解析误差 2.24%，静应力误差 1.32%，物理量和证据状态已分层 | 解析效率不是直接 CFD 功率验证；侵蚀材料参数尚无真实标定 |
| 工程可实现性 | 14–16 | native-SI、拓扑 v3、严格碰撞、监视器解析、CalculiX case 和 Pareto 合同 | 当前 case 是静止单斗；无整机旋转 VOF、无实际 FEM 求解、粒子覆盖低 |
| 主要磨蚀类型和机理 | 12–14 | 切削/变形分类、压力映射和防护分区工具已实现 | 机理仍是模型规则；无同材料形貌/现场磨损验证，热点定位弱 |
| 空间分布预测误差 <5% | 3–5 | 已有 identity-only split、parent-face 真值、盲测重放和外部 pin 验证器 | 当前无真实扫描/测厚；Spearman 0.692、覆盖 0.57% 不是真实误差 <5% |
| **合计** | **58–68** | | |

**当前不满足稳定 90+ 的条件。** 证据缺口（见 §3）未消除之前不应做出此类宣称。

---

## 3. 已知局限与未解证据缺口

以下缺口直接影响评分上限，技术工作可改进但不能完全消除：

### 3.1 侵蚀空间精度
- 无组委会侵蚀参考分布，第5项评分依据不足
- top10_iou=0.105，Spearman=0.692，覆盖 0.57%——空间精度可辩护但弱
- RBF 无 holdout 验证集
- 旧 simpleFoam 首选源只有 438 hits / 218988 父面 = 0.20%；新 URANS 质量分层 20k 单 run 有 1017–1146 impacts，但 5-seed 空间 Spearman 和 10k→20k 稳定性仍未通过预注册门槛

### 3.2 动应力
- 130 MW 动幅相对误差 5.69%，超过竞赛严格阈值 5%
- 工程容差下通过，但赛事评分下失败
- 物理模型为参数化解析模型，非真实 FEM 解算

### 3.3 结构应力场与 FEM
- `step30_fem_stress_field.py` 仍是标定约束参数化模型，非真实有限元求解
- 新增 `step30_cfd_pressure_mapping.py`、`step30_calculix_case.py`、`step30_fem_results.py` 和 `step30_fem_validation.py` 只提供压力映射、case 生成和 fail-closed 验收
- 本机已安装 CalculiX 2.19，并通过 bundled `freq_test` smoke case；provenance 见 `fem_tool_provenance_ccx219.json`。这只证明 solver 可用，不是项目 FEM 结果
- `step30_calculix_case.py` 已新增 `centrifugal_only_s1` 纯离心 375 rpm 合同，避免把 S1 错写成带压力载荷的 case
- 从 GPU/COMSOL 历史工作目录拉回了单水斗 STEP/日志/脚本：`output/step30_fem_remote_export_v1/`。本机 Gmsh 能将 `single_bucket_target_authoritative.step` 读成 1 个 volume 和 30 个 surfaces，说明可网格化实体几何已存在
- 已新增 `step30_fem_surface_catalog.py` 并冻结 `output/step30_fem_remote_export_v1/step_surface_catalog.json`，用于人工确认 STEP 的 30 个 boundary surfaces；当前仍不能把 surface tag 直接当 ROOT/WETTED/AI5 证据
- 当前没有 CalculiX 收敛结果、三网格收敛或独立 AI5-04 盲验证，因此 `fem_gate` 不能通过
- 当前剩余阻塞点已冻结在 `output/step30_fem_official_evidence_v1.json`：ROOT/WETTED/AI5 的物理选择、AI5 数值 CAD patch 与 gauge direction、STEP 坐标单位独立确认、材料密度证书；在这些证据到位前，`FEM_S0_S1_READINESS.md` 仍保持 `blocked`
- 已新增 `step30_fem_evidence_intake.py`、`step30_fem_selection_intake.py`、`FEM_EVIDENCE_ACQUISITION.md` 和 `templates/step30-fem-selection-material-intake-v1.json`；前者冻结 16 个官方 DEC 源文件哈希，后者对完整 CAD/材料提交包 fail-closed 验证

### 3.4 直接 CFD 水力性能
- 新增 `step31_hydraulic_case_contract.py`、`step31_openfoam_monitor_io.py`、`step31_cfd_hydraulic_performance.py` 和 DEC 对齐模块
- 当前 case 是 `stationary_single_bucket`，无旋转 runner/VOF/净水头测量面；其力矩不能称整机轴功率，性能 gate 保持 blocked
- 现有 2.24% 是解析效率与 DEC 能量数据比较，不是直接 CFD 扭矩/功率验证

### 3.5 材料与工况加权
- 新增材料 trial protocol/replay，冻结 heat/batch、证书和 raw replay；当前无真实同材料试验，`independent_material_gate=false`
- 工况分配比例（额定 25%）仍是项目假设，非实际运行历史

### 3.6 防护设计与权衡
- 新增 `step30_protection_design.py` 和 `step30_multiobjective_tradeoff.py`
- 工具可生成分区/Pareto 诊断，但缺真实侵蚀真值、收敛 FEM 和每候选 CFD/FEM/侵蚀重跑，`protection_gate/tradeoff_gate` 必须 blocked

### 3.7 正式证据
- `official_reference_gate = false`、`official_accuracy_gate = false`；已有 DEC 官方资料包，但当前缺少可复现的官方 FEM/CFD solver case/result、官方侵蚀空间参考和 trusted SHA-256 pin
- 合成数据、DEC 测量值、生成的 CalculiX case 和参数化 FEM 不能代替官方或真实物理证据

---

## 4. 核心技术模块（简表）

| 模块 | 用途 |
|---|---|
| `step30_lagrangian_particle_tracking.py` | 粒子追踪、Mansouri 侵蚀、严格碰撞 |
| `step30_cfd_topology.py` | 拓扑 v3、mesh signature、父面几何 |
| `step30_artifact_identity.py` | artifact set/particle run 身份 |
| `step30_si_manifest.py` | native SI manifest v2、solver 语义 |
| `step30_urans_validation.py` | URANS 门禁 |
| `step30_hydraulic_validation.py` | 水力效率验证（η 误差 2.24%） |
| `step30_dec_stress_validation.py` | 动应力验证（严格 12/13，工程 13/13） |
| `step30_erosion_spatial_map.py` | 线性 RBF 侵蚀空间场（稳定替代破损 GPR） |
| `step30_duty_cycle_erosion.py` | 工况加权侵蚀（已加固空值/除零/NPZ 句柄） |
| `step30_particle_seed_convergence.py` | 多 seed/多规模统计收敛、ESS/CI/空间稳定性门禁 |
| `step30_lagrangian_particle_tracking.py` | 新增显式 seed 与 `mass_stratified_v1` 无偏质量分层 PSD；默认旧行为不变 |
| `step30_fem_stress_field.py` | 参数化应力场（非真实 FEM） |
| `step30_official_reference_simulation.py` | 官方参考仿真框架（当前有 DEC 资料包，但无可复现官方 FEM/CFD result package） |
| `step30_independent_erosion_protocol.py` | 独立真值协议、identity-only split、预测承诺 |
| `step30_independent_erosion_truth.py` | 扫描/测厚差分、topology-v3 parent-face 真值 |
| `step30_measurement_acceptance.py` | 四工件原始证据 replay（独立 gate，不是官方 gate） |
| `step30_independent_erosion_cli.py` | 真值 v2 manifest/CLI/coverage delivery |
| `step30_external_commitment_pin.py` | 外部 commitment pin 离线验签 |
| `step30_material_trial_protocol.py` / `step30_material_trial_replay.py` | 同材料试验冻结与原始重放 |
| `step31_hydraulic_case_contract.py` / `step31_cfd_hydraulic_performance.py` | 直接 CFD 性能合同（当前 case blocked） |
| `step30_cfd_pressure_mapping.py` / `step30_calculix_case.py` | CFD 压力到 FEM 载荷与 CalculiX 输入 |
| `step30_fem_results.py` / `step30_fem_validation.py` | FEM 结果与三网格/AI5 验收 |
| `step30_fem_surface_catalog.py` | 对 authoritative STEP 的 30 个 boundary surfaces 做不可变 catalog，供 ROOT/WETTED/AI5 人工确认 |
| `FEM_S0_S1_READINESS.md` | 真实 S0 三网格和 S1 离心 FEM 的不可变准备/退出合同（当前 blocked） |
| `step30_protection_design.py` / `step30_multiobjective_tradeoff.py` | 防护分区与 CFD/FEM/侵蚀 Pareto |
| `step30_validation_common.py` | 哈希、单位、evidence taxonomy |

---

## 5. 验证命令

```bash
cd "./workspace"

# 完整回归（以实际输出为准）
python -m pytest -q

# 模块自测
python step30_dec_stress_validation.py --selftest
python step30_hydraulic_validation.py --selftest
python step30_erosion_spatial_map.py --selftest
python step30_duty_cycle_erosion.py --selftest
python step30_fem_stress_field.py --selftest
python step30_official_reference_simulation.py --selftest
python step30_measured_validation.py --selftest
python step30_measurement_acceptance.py --selftest
# 材料 replay 没有无输入 selftest；使用下方聚焦 pytest
# CFD 性能 CLI 需要冻结 contract 和 monitor JSON，当前无真实整机 case

# 新技术链聚焦回归
python -m pytest test_step30_material_trial_replay.py \
  test_step31_cfd_hydraulic_performance.py \
  test_step30_structural_protection.py \
  test_step30_independent_erosion_delivery.py -q

# 独立真值合同回归
python -m pytest test_step30_independent_erosion_protocol.py \
  test_step30_independent_erosion_truth.py \
  test_step30_independent_erosion_acceptance.py -q
```

---

## 6. 不要做的事情

- 不要把 AI5-04 动应力数据当作侵蚀空间分布参考
- 不要把 1.2 MPa 工程阈值描述为 DEC 仪器噪声下限或 ±1–2% FS 精度规范
- 不要把动应力验证通过（12/13 严格）等同于侵蚀空间误差 <5%
- 不要把 `step30_fem_stress_field.py` 描述为真实 FEM 求解
- 不要宣称严格 13/13——130 MW 在竞赛规则下失败
- 不要因为已有 DEC 官方资料包就直接声称 `official_reference_gate=true`；必须有明确的官方参考对象、身份绑定和 trusted pin
- 不要把合成 fixture 或 DEC 测量值冒充官方参考数据
- 不要把组委会仿真写成 `measured`
- 不要用 URANS 5000-parcel 结果替代 simpleFoam 首选侵蚀源（覆盖 0.074% vs 0.57%）
- 不要用 thin-plate-spline RBF（smoothing<0.1）做侵蚀空间插值

---

## 7. DEC PELTON 2024 数据包说明

```
DecPelton_extracted/
```

| 目录/文件 | 内容 |
|---|---|
| `3.1/30MW.txt … 140MW.txt` | AI5-04 动应力时序（1000 Hz，MPa） |
| `3.1/额定工况.txt` | 额定空转（mean=22.77 MPa，std=0.32） |
| `3.2 PDF` | 动应力试验报告（E=206 GPa，ν=0.3） |
| `3.3 PDF` | 数据说明（半桥，AI5-04=水斗根部最大应力测点） |
| `1.3/_E*.txt` | 能量试验效率曲线（n₁₁, Q₁₁, η） |
| `1.1 .x_t` | Parasolid CAD 水斗几何 |

**竞赛未提供侵蚀空间分布参考数据。**

使用声明（DEC PELTON 数据要求）：
*The research is supported by the Research project of large Pelton turbine DEC PELTON 2024.*

---

### 5.1 本轮回归结果（2026-08-03）

```text
新技术链聚焦套件：90 passed
粒子 PSD/immutable/收敛聚焦套件：24 passed
完整 workspace pytest：239 passed, 1 skipped
py_compile：通过
```

新增测试主要使用临时 synthetic/解析 fixture，仅证明合同、哈希、单位、身份、重建、监视器解析、压力守恒、FEM 输入门禁和 fail-closed 逻辑；不构成真实侵蚀空间精度、真实 CFD 效率、真实 FEM 或材料标定证据。

---

交接文档结束。

## 8. 本次接手阶段 A 验证（2026-08-10）

本次会话实际访问的快照路径为：

``

交接说明中的 `D:` 路径在当前会话中未挂载；未创建替代证据，也未重置或清理原工作区改动。Git 分支仍为 `fix/strict-collision-open-cfd`。

实际验证结果：

- `uv run --with meshio --with vtk --with gmsh --with pytest python -m pytest -q`：`249 passed, 1 skipped`
- Step30/Step31 相关模块 `py_compile`：通过
- `python step30_fem_evidence_intake.py --validate output/step30_fem_official_evidence_v1.json`：`blocked`；冻结 manifest 仍引用不可访问的旧 `B:` 官方资料根目录
- `python step30_fem_selection_intake.py --package templates/step30-fem-selection-material-intake-v1.json`：`blocked`；模板仍未填写

本次仅修复了两个可复现的软件问题：旧 NumPy 对 `np.trapezoid` 的兼容回退，以及 evidence CLI 对缺失来源的 `blocked` 分类。真实 FEM、S0/S1 和 official gate 状态仍保持 `blocked`。
## 9. Literature Prior and Geometry Audit (2026-08-10)

The current exploratory package is frozen in
`output/step30_literature_fem_provisional_manifest_v1.json`.

- `step30_literature_prior_degradation.py` compares Oka, Finnie, Tabakoff, and
  Mansouri on the fixed `particle_hits_cfd.npz` artifact.
- The prior ranges are density 7700/7800/7900 kg/m3, hardness 3/4/5 GPa,
  concentration scale 0.5/1/2, and 32 deterministic draws with seed 20260810.
- `output/step30_literature_prior_degradation_v2.json` and its NPZ maps are
  `provisional`; the optional stress field is analytical/parametric and is not
  a real FEM solver result.
- The report explicitly keeps `fem_gate=blocked`,
  `real_fem_evidence=false`, and `ai5_used_for_calibration=false`.
- `step30_fem_geometry_audit.py` binds the authoritative STEP, COMSOL export
  Java source, export log, and surface catalog. It records 1 Gmsh volume and
  30 Gmsh surfaces.
- The STEP contains an embedded `SI_UNIT(...METRE...)` declaration and the
  export Java requests `m`; both are recorded as `declared`, not as a formal
  unit pass, because the Gmsh native scale consistency check is `blocked`.
- `ROOT_INTERFACE`, `WETTED_SURFACE`, and `AI5_04` remain `blocked`. Surface
  tags, bounding boxes, and visual inspection are not accepted as physical
  selection evidence.

## 10. Literature reference and independent scan status (2026-08-10)

The conservative public-literature regional package is frozen in
`output/step30_literature_reference_distribution_v1.json`.
Its v2 provisional artifact manifest is
`output/step30_literature_fem_provisional_manifest_v2.json`.

- It is explicitly typed as `literature_reference` and `provisional`.
- Splitter-like and cutout-like regions have the strongest cross-source support.
- Source-specific geometry anchors are retained as reported: approximately
  `0.03` splitter-height reduction and `0.05` cutout abrasion, both expressed
  as fractions of bucket width in the cited field study. These values are not
  transferred to the DEC geometry.
- The package stores an ordinal literature-support share, not a physical
  probability and not a DEC face-wise truth field.
- Model-to-region comparison is `blocked` because no externally confirmed
  face-region mapping is available. Gmsh surface tags, bounding boxes, and
  visual guesses remain inadmissible.

The independent measurement delivery chain is documented in
`INDEPENDENT_BASELINE_WORN_ACQUISITION.md` and the intake template
`templates/step30-independent-baseline-worn-acquisition-v1.json`.

- No real baseline/worn scan pair is present in this repository.
- The existing scan-pair protocol remains the only admissible path to spatial
  truth and requires at least three independent groups, with five recommended
  for a formal candidate.
- Synthetic fixtures validate software contracts only; they do not open
  `independent_accuracy_gate`.

## 11. Conservative current score estimate

This estimate preserves the five 20-point dimensions already recorded in this
handoff. It is a technical evidence estimate, not an official judge decision.

| Criterion | Score | Current strengths | Current limits |
|---|---:|---|---|
| Design completeness | 15/20 | CFD, particle, four-model prior, uncertainty propagation, exploratory FEM, protection analysis, and scan intake chain are implemented | No real worn scan, same-material trial, solver result, or protection benefit evidence |
| R&D rationale | 16/20 | Public literature is frozen with DOI/source roles; hydraulic and static-stress checks are documented; evidence states are fail-closed | Literature regions are qualitative; material parameters and spatial truth are not independently calibrated |
| Engineering feasibility | 15/20 | Native-SI artifacts, topology-v3, strict collision semantics, convergence contracts, and CalculiX preparation exist | Current structural case is not a completed real FEM; hydraulic case is not a full rotating runner/VOF acceptance case |
| Wear mechanism and type | 13/20 | Multiple erosion models and impact/mechanism classification are available; splitter/cutout/base-side regions are now literature-priorized | No same-material morphology or field-worn validation; exact DEC hotspot mapping is blocked |
| Spatial prediction error below 5% | 3/20 | Identity-only split, parent-face truth builder, preregistered metrics, and independent replay verifier exist | No real baseline/worn truth; current CFD coverage/correlation cannot establish a measured error below 5% |
| **Total** | **62/100** |  | The score is capped by blocked independent spatial truth and real FEM evidence |

The score should not be increased merely because the literature package exists:
it improves the rationale and prioritization evidence, but it does not replace
measured baseline/worn scans or DEC official erosion reference data.

For the stress component, the current source-of-truth replay is
`output/dec_stress_validation_report_current_v1.json`: strict competition
result `12/13` with 130 MW amplitude error `5.685%` (`failed`), while the
separate project engineering tolerance is `13/13` (`passed`). The historical
`output/dec_stress_validation_report.json` is retained unchanged and is not
the current scoring source.
