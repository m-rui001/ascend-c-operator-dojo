# Round 9: DoubleBuffer/流水调优深化 —— 我的设计（写代码前，未看社区流水样例）

## 0. 前置功课收获

1. **机制本质**：AI Core 上有三类独立指令队列——MTE2（GM→UB）/ MTE3（UB→GM）/ V（矢量计算）/ Cube。DoubleBuffer 的收益来自**队列间并行**，不是"缓冲区多"本身。
2. **利用率模型**：三阶段各耗时 t 时串行利用率 1/3；充分重叠后稳态吞吐 = max(t_MTE2, t_V, t_MTE3)——**瓶颈段决定吞吐，非瓶颈段全部被隐藏**。
3. **反例边界**（文档明确警告）：搬运远快于计算时收益小；数据太小一次能装下时强行分块反而降低利用率。BUFFER_NUM 不是越大越好（第 2/5/8 轮已三次见到 BUFFER_NUM=1 的合理场景）。
4. workspace 文档顺带：系统 vs 用户 workspace 的分配规则（第 6/7 轮已实践）。

## 1. 本轮练习：对 Round 1 的 Add 做流水深化

### 我的流水分析（Add：读 x、读 y、加、写 z，每元素 4 次访存 1 次计算）

- Add 是**访存瓶颈型**：MTE2 要搬 2 份输入、MTE3 搬 1 份输出，V 只做 1 次加法。稳态吞吐 ≈ max(t_MTE2, t_MTE3)，V 几乎总能被喂饱。
- 结论 1：BUFFER_NUM=2 通常已够（MTE 是瓶颈，V 消费快，2 块交替即可让 V 不空转）；BUFFER_NUM=3+ 的收益仅在 V 波动或 MTE3 与 MTE2 争抢时出现——**应做成可配置，按实测调**。
- 结论 2：Round 1 的**双队列结构有同步开销**——x、y 两个队列各自 Alloc/EnQue/DeQue/Free，8 次队列操作/轮，且两次 EnQue 之间隐含两次独立同步。优化：**单队列装载 x|y 拼接块**（一次 Alloc 一块 2×tileLength，DataCopy 两段写入同一 buffer，一次 EnQue/DeQue），队列操作减半、x/y 天然同步到达。
- 结论 3：MTE3 侧输出队列独立双缓冲，与输入侧解耦。

### 结构设计

```
InitBuffer(inQ, BUFFER_NUM, 2*tileLength)   // 单队列，槽内 x|y 拼接
loop: CopyIn(i){alloc→DataCopy x→[tile,2tile) DataCopy y→EnQue}
      Compute(i){DeQue→Add(z, buf[0], buf[tile], tile)→EnQue(out)}
      CopyOut(i){DeQue→DataCopy→Free}
```

### 配置矩阵（留给对比环节验证我的判断）

| 配置 | 我的预测 |
| --- | --- |
| 双队列 BUFFER_NUM=2（Round 1 原版） | 基线，队列同步开销 8 次/轮 |
| 单队列拼接 BUFFER_NUM=2 | 同步开销减半，稳态吞吐相同（瓶颈在 MTE） |
| 单队列 BUFFER_NUM=3/4 | 吞吐几乎不变（MTE 瓶颈不因缓冲加深而变），UB 占用增加 |
| 数据量小到单块装下 | 应退化为 BUFFER_NUM=1 单发 |

## 2. 不确定点（//?? 清单）

1. 单队列里"取拼接块的后半段"的正确写法：`buf[tileLength]` 偏移 LocalTensor 是否合法/高效？
2. DataCopy 目的写 `xLocal[tileLength]` 是否要求新的对齐处理？
3. Add(dst, src0, src1) 的 src0/src1 来自同一块 buffer 的不同偏移——地址重叠约束是否允许（部分重叠禁止，但这里两段不重叠，只是同属一个 Tensor）？
4. 社区的"流水调优"是否还有我没意识到的段间优化（如 CopyOut 提前、尾块单列、MTE3 聚合）？
5. head/tail 开销类最佳实践（7/8 号样例）暗示**首轮发射延迟**和**尾轮排空**也是调优点——社区用什么结构处理（prologue/epilogue 拆分）？
