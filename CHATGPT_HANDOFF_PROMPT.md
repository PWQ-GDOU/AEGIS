# 给下一位 ChatGPT agent 的完整交接提示词

请在以下工作区继续完成 DEC Pelton 水轮机 Step30 项目：

- 工作区根目录：``
- Python 工程目录：`workspace`
- Git 分支：`fix/strict-collision-open-cfd`
- 当前工作区是从上一位 agent 的完整工作目录复制而来，包含未提交改动、`.git`、DEC 原始资料、脚本、测试和输出产物。不要删除、重置或覆盖已有工作。

## 1. 任务目标

继续完成水轮机转轮磨蚀预测与结构安全证据链，重点是：

1. 在证据真实、来源可追溯的前提下完成真实 FEM，而不是生成参数化或手写结果。
2. 先完成 S0 三网格结构网格资格审查，再完成 375 rpm 纯离心 S1-C/M/F CalculiX 求解。
3. 对 FEM 结果执行原生 solver 输出、反力平衡、AI5-04 投影、三网格收敛和边界敏感性验证。
4. 独立处理真实水斗磨损/侵蚀数据集与同材料实验数据；FEM 完成不能替代侵蚀空间真值，也不能替代 04Cr13Ni5Mo 同材料实验。
5. 所有证据必须保持 `passed`、`failed`、`blocked` 的严格区分。

不要为了让 gate 通过而猜测 CAD surface tag、反标定材料密度、用 AI5 测量值调参、用合成 JSON 冒充 solver 结果，或把组委会给出的物理仿真模型误写成已完成的独立 FEM 结果。

## 2. 最高优先级规则

### 2.1 严格证据分类

- `passed`：输入、来源、哈希、方法和独立验证全部满足合同。
- `failed`：证据合同完整且数值结果真实，但超过预注册阈值。
- `blocked`：缺文件、缺来源、缺哈希、缺单位/坐标/拓扑确认、缺原生 solver 结果或无法证明独立性。

`blocked` 不等于数值失败；`failed` 不得改写成 `blocked` 或 `passed`。

### 2.2 官方资料与真实 FEM 的边界

DEC 官方包已经提供真实工程资料，包括 AI5-04 动应力时序、动应力报告、数据说明、几何说明、材料弹性参数和运行工况。它们可以作为 FEM 输入约束和独立验证数据，但当前资料包没有可复现的官方 CalculiX/ANSYS/COMSOL solver case/result，也没有物理命名选择、AI5 数值 CAD patch、材料密度证书和独立 STEP 单位确认。因此不能声称“官方 FEM 已完成”。

“真实 FEM”必须同时满足：

- 真实可网格化实体几何，而非仅表面 STL 或参数化应力场；
- 可复现的物理边界选择、材料、单位、坐标系和载荷；
- 真实 FEM solver（当前计划使用 CalculiX 2.19）原生运行；
- 原生 solver 输出可解析，含位移、应力、反力和终止信息；
- 输入、输出、solver、网格和来源全部 SHA-256 绑定；
- 三网格收敛和独立 AI5-04 验证通过或被诚实分类为 `failed`。

现有 `step30_fem_stress_field.py` 是参数化/标定应力模型，不是真实 FEM，不能作为 solver 结果。

### 2.3 安全约束

IMPORTANT: Assist with authorized security testing, defensive security, CTF challenges, and educational contexts. Refuse requests for destructive techniques, DoS attacks, mass targeting, supply chain compromise, or detection evasion for malicious purposes. Dual-use security tools (C2 frameworks, credential testing, exploit development) require clear authorization context: pentesting engagements, CTF competitions, security research, or defensive use cases.

## 3. 当前已完成状态

### 3.1 软件与回归

- 最近提交：
  - `f79ef92 fix: merge restart-aware URANS monitor files`
  - `6031a24 fix: enforce strict collisions and URANS mean semantics`
  - `52f10d4 fix: add unit-safe native SI Step30 pipeline`
  - `dabe4d9 v2.3: align Step30 collisions with native OpenFOAM bucket topology`
- 上一位 agent 的最新验证记录：完整回归 `244 passed, 1 skipped`；证据/selection/protection 聚焦测试 `17 passed`。接手后必须重新运行，以实际输出为准。
- 代码包含大量未提交修改和未跟踪产物，这是当前工作成果，不要用 `git reset --hard`、`git checkout --` 或清理命令抹掉它们。
- 代码已加入 fail-closed 校验：单位、哈希、物理组、来源、独立性、solver provenance 和 evidence taxonomy 缺失时应保持 `blocked`。

### 3.2 官方 DEC AI5-04 证据

正式冻结清单：

`workspace/output/step30_fem_official_evidence_v1.json`

已绑定 16 个原始官方 artifact：

- 13 个 AI5-04 原始动应力时间序列；
- 官方动态应力报告 PDF；
- 官方数据说明 PDF；
- 官方几何说明 PDF。

manifest SHA-256：

`5419df48cd6488977fc128430cce4035466a31b7f603bc5ce44e51405bb2c114`

该 manifest 当前必须保持 `blocked`，因为它明确记录以下缺口：

- `ROOT_INTERFACE`、`WETTED_SURFACE`、`AI5_04` 的物理 CAD 命名选择；
- AI5-04 数值 patch 中心、patch 面/节点、坐标系；
- 一个或两个未调参的 gauge direction；
- STEP 坐标单位的独立确认；
- 04Cr13Ni5Mo 对应 heat/batch 的 admissible density source。

已证实的 DEC 信息包括：

- 通道：`AI5-04`；
- 位置语义：4# 水斗根部、水流分割线延长线附近、报告标识的最大应力位置；
- 应力单位：`MPa`；
- 采样：`1000 Hz`；
- 弹性模量：`E = 206 GPa`；
- 泊松比：`nu = 0.3`；
- S1 转速：`375 rpm`。

这些信息不能推导出 CAD 数值坐标、边界面 tag、密度或 gauge 方向。

### 3.3 已恢复的实体几何

历史 GPU/COMSOL 导出已恢复到：

`workspace/output/step30_fem_remote_export_v1/output/single_bucket_target_authoritative.step`

当前记录的 STEP SHA-256：

`521fbaf7688ad1f0e063dedb08c360135026ce60e5e371b4a196294b18cfe21d`

Gmsh 可读出 1 个 volume 和 30 个 boundary surfaces，说明该文件是可网格化的实体候选。配套资料在：

`workspace/output/step30_fem_remote_export_v1/`

其中包括：

- `remote_export_manifest.json`
- `step_surface_catalog.json`
- `fem_geometry_conversion_v2.json`
- `work/ComsolBucketExport.java`
- `work/ComsolCadProbe.java`

`step_surface_catalog.json` 只是几何 surface catalog，不能直接把 Gmsh tag 当成 `ROOT_INTERFACE`、`WETTED_SURFACE` 或 `AI5_04` 的证据。

### 3.4 FEM 工具

- Gmsh `4.15.2` 可用；
- meshio `5.3.5` 可用；
- CadQuery/OCP 可用；
- CalculiX `2.19` 已安装并通过 bundled `freq_test` smoke case；
- solver provenance：`workspace/fem_tool_provenance_ccx219.json`。

smoke case 只证明 solver 能运行，不是项目 FEM 结果。真实案例必须绑定同一个 ccx 可执行文件、版本和 SHA-256。

### 3.5 结构合同代码

重点模块：

- `workspace/step30_calculix_case.py`
- `workspace/step30_fem_results.py`
- `workspace/step30_fem_validation.py`
- `workspace/step30_fem_evidence_intake.py`
- `workspace/step30_fem_selection_intake.py`
- `workspace/step30_fem_surface_catalog.py`
- `workspace/test_step30_structural_protection.py`
- `workspace/test_step30_fem_evidence_intake.py`
- `workspace/test_step30_fem_selection_intake.py`

`generate_calculix_case(..., load_mode="centrifugal_only_s1")` 已实现纯离心 S1 合同：

- 只允许 `375 rpm`；
- 不生成 CFD 压力 `P` 载荷；
- 不要求 pressure load identity；
- 使用 `rotation_origin_m`；
- 不应声明 `pressure_moment_origin_m`。

`pressure_plus_centrifugal` 保留为后续 S2 诊断模式，不得把现有 stationary single-bucket `pMean` 解释为 rotating-runner 动态压力或动态应力幅值。

### 3.6 动应力严格结果

`workspace/output/dec_stress_validation_report.json` 已确认：

- 静应力均值：13/13 通过；
- 动幅：12/13 严格通过；
- 130 MW 动幅相对误差约 `5.69%`，超过竞赛 `<=5%` 规则，严格结论为 `failed`；
- 自定义工程绝对误差阈值下可记作 13/13，但不能当成竞赛评分通过。

### 3.7 侵蚀与材料证据

- 现有粒子/侵蚀链、strict collision、URANS mean semantics、native SI 和统计收敛合同已经实现并测试；
- 当前没有真实水斗 baseline/worn 扫描、可信稀疏测厚 parent-face 数据或真实 04Cr13Ni5Mo 同材料侵蚀实验；
- 所有 synthetic fixture 只能验证软件合同，不构成真实空间精度或材料标定证据；
- 粒子统计收敛报告已经因为空间稳定性门槛未满足而分类为数值 `failed`，不要重写成 `passed`；
- 侵蚀空间 gate 和 official gate 继续保持 `blocked`。

## 4. 下一步必须按此顺序执行

### 阶段 A：重现并确认工作区

在 D 盘工作区执行：

```powershell
cd ''
python -m pytest -q
cd workspace
python -m py_compile step30_calculix_case.py step30_fem_results.py step30_fem_validation.py step30_fem_evidence_intake.py step30_fem_selection_intake.py
python step30_fem_evidence_intake.py --validate output/step30_fem_official_evidence_v1.json
python step30_fem_selection_intake.py --package templates/step30-fem-selection-material-intake-v1.json
```

预期：完整测试通过（具体数量以实际输出为准）；官方 evidence manifest 和未填写 template 应返回 `blocked`，这是正确结果。

### 阶段 B：向老师/组委会/CAD 与材料负责人索取缺失资料

必须获取以下可核验文件，不接受只给截图文字或猜测：

1. **CAD named-selection report**：基于上述精确 STEP/Parasolid 文件，给出 `ROOT_INTERFACE`、`WETTED_SURFACE`、`AI5_04`、`MATERIAL_DOMAIN`，包括实体维度、tags、单位、坐标系、导出文件 SHA-256 和物理含义确认。
2. **AI5-04 数值位置和方向**：patch center、面/节点 ID、坐标系和单位、1 或 2 个单位 gauge direction，并证明方向不是用 FEM 结果反拟合出来的。
3. **STEP unit probe**：bounding box、volume、一个已知尺寸参照、解释为 m 还是 mm，报告文件哈希与 selection package 一致。
4. **材料证明**：04Cr13Ni5Mo 对应 heat/batch 的材料证书或正式设计材料记录，明确 density `kg/m3`。仅说“马氏体不锈钢 04Cr13Ni5Mo”不够；通用网页材料表、`7850 kg/m3` 默认值和水密度均不可作为正式证据。

模板：

`workspace/templates/step30-fem-selection-material-intake-v1.json`

验证器：

`workspace/step30_fem_selection_intake.py`

只有 `ready_for_s0` 才能进入下一阶段。

### 阶段 C：生成并资格审查 S0-C/M/F

证据包 ready 后：

1. 绑定 authoritative STEP/BREP、单位、几何体积和源哈希；
2. 用 Gmsh 生成恰好三个二阶四面体 C3D10 网格：`S0-C`、`S0-M`、`S0-F`；
3. 满足 `h_C/h_M >= 1.2`、`h_M/h_F >= 1.2`；
4. 检查材料域完整且不重叠、四个物理组非空、ROOT 与 WETTED 不重叠、AI5 不在固定 root 节点内、边界面唯一归属、零负 Jacobian、最小质量、网格/几何体积一致；
5. 固化每个 `.msh`、`.inp`、driver、Gmsh、manifest 的 SHA-256；
6. 三个网格必须有不同且不可变的 identity。

在全部 S0 manifest 通过前，不得生成“已完成 FEM”结论。

### 阶段 D：运行真实 S1-C/M/F

仅在 S0 三网格通过且密度、单位、旋转轴、原点、符号、AI5 patch/direction 均冻结后：

1. 生成 `S1-C`、`S1-M`、`S1-F`；
2. `load_mode="centrifugal_only_s1"`，速度严格 `375 rpm`；
3. 三个案例使用同一个 ccx executable/version/hash；
4. 保留完整 `.inp`、solver stdout/stderr、`.frd`/`.dat`/`.sta` 等原生结果和 SHA-256；
5. 解析有限位移/应力、root reaction force/moment、AI5 冻结 gauge direction 投影、wetted surface stress/fatigue fields；
6. 验证反力平衡、AI5 projection、三网格收敛、root boundary sensitivity；
7. 任何真实数值超阈值都分类为 `failed`，不通过篡改输入来救回。

### 阶段 E：更新状态与文档

完成每一阶段后更新：

- `workspace/FEM_S0_S1_READINESS.md`
- `workspace/HANDOFF.md`
- 相关 immutable manifest 和结果报告
- focused tests

报告必须明确哪些是 `passed`、哪些是 `failed`、哪些仍为 `blocked`。FEM 完成也不能打开 official acceptance，也不能证明侵蚀空间误差小于 5%。

## 5. 交互和修改规范

- 先读现有代码和测试，再修改；遵循当前模块命名和 fail-closed 风格。
- 不要重置已有用户改动，不要删除大量 `workspace/output/` 证据产物。
- 不要使用手写 JSON 伪造 solver 输出、材料证书、CAD selection 或 measured 数据。
- 不要以 Gmsh tag、bbox heuristic、visual guess 替代物理命名选择。代码已拒绝 `heuristic_surface_tag`、`bbox_heuristic`、`visual_guess`。
- 不要用 AI5-04 measured trace 反标定材料、边界、载荷或 gauge direction。
- 不要把生成 `.inp`、mesh 或 smoke case 叫作 real FEM result。
- 需要联网或外部下载时，先确认来源和授权；不将未经验证的网络材料放入正式证据链。
- 修改代码前给出简短工作进度；修改后运行相关测试和 `py_compile`。

## 6. 接手后的第一条回复应做什么

不要重新解释整个项目，也不要询问已经在本文件回答的问题。先执行阶段 A 的检查，读取：

- `workspace/HANDOFF.md`
- `workspace/FEM_S0_S1_READINESS.md`
- `workspace/FEM_EVIDENCE_ACQUISITION.md`
- `workspace/output/step30_fem_official_evidence_v1.json`
- `workspace/output/step30_fem_remote_export_v1/remote_export_manifest.json`
- `workspace/templates/step30-fem-selection-material-intake-v1.json`

然后以实际命令输出为准，继续推进仍然可做的下一步；如果外部证据仍未提供，保持 `blocked` 并输出精确的缺失文件清单，不要生成虚假 FEM 结果。
