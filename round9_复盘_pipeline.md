# Round 9 复盘：DoubleBuffer / 流水调优深化

> 我的实现：`my_impl/round9_pipeline/`（DESIGN.md / add_deep_pipeline.cpp）——对 Round 1 的 Add 做流水深化：单队列装载 x|y 拼接块（队列操作 8 次/轮→4 次/轮）+ BUFFER_NUM 可配置
> 对比材料：
> - 官方《DoubleBuffer》《如何使用workspace》文档（`docs_notes/pb_0090/0092.md`）
> - 官方论坛《Ascend C 算子性能优化实用技巧》系列（搜索确认 TQueBind 语义）
> - `reference/samples/.../21_vectoradd_kernellaunch/VectorAddMultiCoreWithTilingBroadcast/`（社区 add 的带广播变体）
> ⚠️ 诚实说明：samples 中 2_features/8_doublebuffer_pipeline、4_best_practices/7/8/9（流水类）**全部是"待补充"空壳**——本专题的社区代码证据比前几轮少，部分结论只能到 API 文档与论坛文章层面。

---

## 一、文档层的收获

1. **DoubleBuffer 的本质是队列并行**：MTE2（GM→UB）/ MTE3（UB→GM）/ V（计算）三类指令队列相互独立，双缓冲只是让这三类队列的指令交替发射。利用率模型：充分重叠后稳态吞吐 = **max(t_MTE2, t_V, t_MTE3)**，瓶颈段决定一切。
2. **文档明确的两个反例**：搬运远快于计算时收益小；数据单块装得下时强行分块适得其反。这与我前几轮积累的"BUFFER_NUM=1 合理场景"（softmax 大块、broadcast 物化、MIX 同步节奏、TBufPool 阶段内）完全一致——**缓冲深度是瓶颈分析的输出，不是输入**。
3. **TQueBind**（官方性能技巧系列）：纯搬运场景把 VECIN 与 VECOUT 绑定，省掉"VECIN→VECOUT"的 UB 内部拷贝与相应同步——本轮新学到的流水优化 API。

## 二、我的设计 vs 社区证据（//?? 销案）

1. **单队列拼接 x|y（我的核心设计）**：未在 samples 中找到先例。社区的同场景写法（21 号 Broadcast add 变体）是**两个独立 TBuf + in-place Add**：`Add(tmpTensor0, tmpTensor0, tmpTensor1, tileLength)`——dst 与第一源 100% 重叠是文档明确允许的形态。我的拼接队列保留为待实测假设（收益假设：队列操作减半 + x/y 单次同步；风险：`buf[tileLength]` 偏移寻址的对齐与编译器支持未验证）。
2. **BUFFER_NUM 配置矩阵**：我的预测（访存瓶颈型 2 足够、加深不减吞吐只加 UB）与文档反例边界自洽，且被 2/5/7/8 四轮的 BUFFER_NUM=1 实例交叉印证。本轮实现用编译宏做成可配置，符合"按实测调"的定位。
3. **in-place Add（Round 8 遗留 //?? #5）**：**跨轮销案**——21 号样例 `Add(tmpTensor0, tmpTensor0, tmpTensor1)` 正是我 Round 8 写的 `Add(xLocal, xLocal, yLocal)` 同款形态，社区确实在用。
4. **head/tail 开销、尾核优化、融合流水**：三个最佳实践全是空壳，本轮**无法销案**——这是九轮以来第一次对比环节证据不足，如实记录并列入后续轮次待办。

## 三、流水调优的决策序列（本轮沉淀的方法论）

按收益/风险排序的调优检查单：

1. **先判定瓶颈段**（MTE2 / V / MTE3 谁最慢）——决定一切后续动作；
2. 缓冲深度 = 让非瓶颈段全程有活干的最小值（访存型 2 起步，计算型可减到 1）；
3. 减少队列操作次数（合并队列/拼接块，前提是寻址合法性已验证）；
4. 纯搬运路径检查能否 TQueBind；
5. 中间量用 in-place（dst=src0 重叠约束内）省缓冲；
6. 头开销/尾开销（首轮发射、尾轮排空）——证据不足，待后续轮次补。

## 四、经验教训

1. **性能优化文档的分布与功能文档不同**：功能语义在官方文档站很全，而性能技巧在 samples 里大量"待补充"，实际散落在官方论坛技巧系列——第九轮起对比源要加"官方论坛/性能技巧系列"这一层。
2. ** DoubleBuffer 不是开关而是瓶颈分析的产物**：先写利用率模型（哪段最慢），再决定缓冲深度和结构，顺序反了就是玄学调参。
3. TQueBind 补全了"API 三件套"的流水维度：队列不只是同步结构，绑定后还是搬运路径优化。
4. 跨轮销案是循环练习的复利：Round 8 的 in-place 疑问在 Round 9 被 21 号样例自然解决——**复盘里的"未验证"清单要跨轮滚动，不能一轮一清**。

## 九轮总览

| 轮 | 主题 | 最大盲区 |
| --- | --- | --- |
| 1 | Add 基础流水 | tiling 泛化性 |
| 2 | Softmax | 没查高阶 API |
| 3 | ReduceSum | repeat/mask/stride |
| 4 | LeakyReLU | 产品支持矩阵 |
| 5 | Broadcast | API 三件套 / UB 布局是 host 决策 |
| 6 | Matmul/Cube | API 封装边界 |
| 7 | MIX 融合 | 伪代码 ≠ 完整实现 |
| 8 | 内存语义 | 池复用 / L2 选择性保护 |
| 9 | 流水调优 | 瓶颈先行；社区性能类 samples 空缺，需换源 |

**下一轮候选**：AtomicAdd 与跨核同步（samples 2_features/5_cross_core_sync、6_aiv_core_sync、7_atomicadd 有真实代码，证据可用）；或 LayerNorm/Gelu 复合算子（cann-ops norm 目录证据充足）。
