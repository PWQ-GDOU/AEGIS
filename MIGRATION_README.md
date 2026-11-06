# 工作区迁移说明

## 位置

当前接手入口：

``

Python 工程：

`workspace`

本目录是从原工作区完整复制得到的快照，约 3.6G。复制范围包含：

- `.git` 和当前分支历史；
- `workspace/` 下的全部源码、测试、schema、template；
- `workspace/output/` 下的 CFD、粒子、侵蚀、DEC、FEM 和验证产物；
- `DecPelton_extracted/`、`DecPelton2024/`、`DecPelton2026/` 和参考资料；
- 原工作区当时的未提交修改和未跟踪文件。

原工作区仍保留在：

`./`

迁移没有删除、重置或覆盖原工作区。H 盘副本与原工作区是两个独立目录；后续修改不会自动同步回 B 盘。

由于 H 盘文件系统不记录 Git 所需的所有权信息，首次在副本中执行 Git 命令时可能出现 `detected dubious ownership`。可在确认路径无误后执行一次：

```powershell
git config --global --add safe.directory ''
```

或者只对单次命令使用：

```powershell
git -c safe.directory='' status
```

## 交接入口

把下面文件的全部内容复制给新的 ChatGPT agent：

`CHATGPT_HANDOFF_PROMPT.md`

现有工程交接文档仍在：

`workspace/HANDOFF.md`

FEM 证据获取说明：

`workspace/FEM_EVIDENCE_ACQUISITION.md`

FEM S0/S1 readiness 合同：

`workspace/FEM_S0_S1_READINESS.md`

## 建议启动命令

在 PowerShell 中：

```powershell
cd ''
python -m pytest -q
cd workspace
python -m py_compile step30_calculix_case.py step30_fem_results.py step30_fem_validation.py step30_fem_evidence_intake.py step30_fem_selection_intake.py
python step30_fem_evidence_intake.py --validate output/step30_fem_official_evidence_v1.json
python step30_fem_selection_intake.py --package templates/step30-fem-selection-material-intake-v1.json
```

预期的证据状态是：官方 evidence manifest `blocked`，未填写的 selection/material template `blocked`。这两个 `blocked` 都是正确的 fail-closed 行为。

## 当前不可跳过的外部输入

- `ROOT_INTERFACE`、`WETTED_SURFACE`、`AI5_04`、`MATERIAL_DOMAIN` 的 CAD named-selection report；
- AI5-04 patch center、patch entity、坐标系和 gauge direction；
- 精确 STEP 的独立 m/mm unit probe；
- 04Cr13Ni5Mo 对应 heat/batch 的材料密度证书或正式设计材料记录；
- 后续真实水斗 baseline/worn 数据和同材料侵蚀实验数据。

没有这些证据时，不得开始正式 S0/S1，不得用估计值替代，也不得把历史 smoke case 或参数化应力字段称为真实 FEM。
