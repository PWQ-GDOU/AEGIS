# Step30 粒子收敛根因状态 v3

## 结论

根因已经从“可能不确定”收窄为两条主线：

1. 有效严格命中支持太稀疏，空间场的 Monte Carlo 方差仍是首要风险。
2. 严格 tracker 的主要成本在重复 CFD KD-tree 查询和 swept-segment 候选三角形测试，而不是近失距离诊断。

确定性问题、单 run 数据完整性问题和历史文件损坏问题已经排除。当前保守评分仍为 `62.4471/100`，粒子收敛门仍为 `failed`。

## 已完成证据

- 35 个 immutable run 已完成数据审计；单 run 链通过，但历史根目录混合了四个实现/配置组，不能直接合并比较。
- 两个独立 DCU replica 的 1000-parcel replay 通过字节级确定性检查。
- 近失诊断 A/B：baseline `810.60 s`，optimized `817.47 s`；严格交点、命中 parcel、质量、子步数和所有物理数组一致。唯一 NPZ 差异是 `particle_run_id` provenance 字段。
- 100-parcel collision profile：`221839` 个严格子步、`443678` 次 CFD 查询、`221839` 次 surface-tree 查询、`1623132` 个三角形测试；近失距离只涉及 `4714` 个候选三角形。

## 数据处理风险

- A/B 的注册距离本身较小（median `3.435 mm`、p95 `5.181 mm`），但比较掩膜只有 `5` 个暴露面，不能支撑空间准确率。
- 历史报告中出现 `n_faces_compared=0` 的工况仍然必须保持 blocked，不能用全域零值或空掩膜提高分数。
- `particle_run_id` 会随配置变化，不能用于判断物理结果是否变化；应使用物理数组规范化哈希和 tracking summary。
- 当前 1000-parcel A/B 只有 `6` 个 impact、`2` 个命中 parcel，不能作为独立空间真值或收敛通过证据。

## 下一轮实验

1. 固定 seed 和输入，单因素改变 `strict_max_substep`、`dt`、CFD 插值和积分器，记录墙钟、子步数、命中 parcel/face、速度、角度和终止原因。
2. 对候选面做宽相直方图和 segment-AABB 对照，目标是减少每子步 triangle test，同时保证 hit arrays 完全一致。
3. 单独完成 m/mm、坐标变换、wall sample、法向和 `UMean` 语义审计，扩大有效比较掩膜后再讨论空间评分。
4. 只有上述实验解释清楚后，才重新启动同一实现哈希下的 discovery/confirmation。

所有门禁保持原语义：`official_accuracy_gate=false`、`official_reference_gate=false`、`fem_gate=blocked`，不得把性能 profiling 或自建数据写成正式准确率。
