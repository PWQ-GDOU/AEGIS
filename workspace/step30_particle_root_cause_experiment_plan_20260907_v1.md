# Step30 粒子不收敛根因实验与数据审计计划

## 当前结论

截至 2026-09-07，根因已经被定位到“稀疏有效命中支持 + CFD/解析参考空间不一致 + 历史版本混用”的组合风险，但尚未修复。2026-09-02 的远程 20k confirmation 已完成但失败，不能因为 v5 粗聚合指标较高而宣布通过。

本计划把问题分成数据完整性、确定性重放、采样、跟踪、碰撞映射、估计器和物理映射七层。所有计算只在 DCU 服务器执行，H 盘只保存控制文件和返回包。

## 先做的审计

`step30_particle_data_audit_v1.py` 对每个 immutable run 检查：

- `particle_hits.npz`、`summary.json`、`validation.png`、`particle_run_manifest.json` 是否齐全；
- manifest 中每个文件的 SHA-256 是否与实际文件一致；
- `parent_face_id`、质量、冲蚀率、速度、角度、粒径的长度和有限性；
- NPZ 计算的 impact mass 与 summary 一致；
- NPZ 计算的 `parcel_mass_kg * ER_mansouri` 与 face erosion 总目标质量一致；
- strict-v2 是否出现 fallback、非 intersection hit 或非 1.0 strict hit ratio；
- seed、artifact_set_id、场时间、碰撞策略、碰撞后端和实现哈希是否混用；
- 命中 parcel 比例、near-miss/impact 比、substep/parcel，识别性能瓶颈。

审计输出分为 `passed`、`warning`、`failed`。任何 error-level 数据问题都阻止后续 confirmation。

## 根因隔离顺序

1. **确定性重放**：同 seed、同输入、同代码哈希，在两个新远程工作区重放。比较 NPZ 字节哈希、summary 哈希、impact 数、终止原因和运行时间。若不一致，先修复并行调度、随机数消费、浮点或文件写入问题。
2. **采样层**：保持跟踪器不变，只比较 PSD/注入采样。主指标是 hit-parcel rate、active-face ESS、target-mass CV 和 active-face JS。
3. **跟踪层**：固定粒子初值，单因素比较 `dt`、`strict_max_substep`、`idw_k8_v1/nearest_cell_v1` 和 integrator。额外记录轨迹位移、碰撞前速度和入射角变化。
4. **碰撞层**：对已记录线段重放交点，核对 triangle ID、parent face ID、法向方向、hit time 和 suppressed-face 逻辑。不得把 near miss 计为 strict impact。
5. **估计器层**：同一组 hits 同时输出 raw、surface kernel 和多个 v5 aggregate 分辨率。粗聚合只能作为 declared coarse screening map，不能替代 parent-face 物理空间准确率。
6. **物理映射层**：单独核对 m/mm、局部到全局变换、wall sample 与 native topology 对应、`UMean` 时间均值语义、碰撞前速度和入射角定义。只有这一层解释了负相关，才重新讨论物理准确率。

## 性能审计重点

历史 20k seed 约需 16,427 至 16,735 秒，而当前 v2 单 seed 默认超时为 7,200 秒。每 parcel 逐步执行 CFD KD-tree 查询和三角形候选求交，near-miss 诊断远高于真实 impact，说明计算资源大量消耗在无效候选筛选上。实验必须先做 smoke benchmark，再为每个远程 job 注册 timeout；不得通过关闭超时或磁盘保护来“解决”性能问题。

## 停止条件

在以下条件全部满足前，不启动新的高粒子数 confirmation：

- 数据审计无 error-level finding；
- 同 seed 重放确定性已证实，或不确定性已明确定位；
- 同一物理输入与实现哈希没有混用；
- CFD 到参考场的注册差异已有解释；
- 运行时间在远程安全预算内；
- raw/active/aggregate 三个层级的指标分开报告。

即使未来数值稳定性通过，`official_accuracy_gate=false`、`official_reference_gate=false`、`fem_gate=blocked` 仍保持原语义。
