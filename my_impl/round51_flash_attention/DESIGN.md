# Round 51: FlashAttentionScore —— 我的设计（A 节前置）

> 语义：注意力分数 S=Q·K^T·scale(+mask/pse) → softmax → O=P·V。生产 6 变体（2199 行主变体）。
> 本轮定位：架构研读 + 单核简化骨架（在线 softmax / S2 循环），不复刻 L1 复用与 sparse 全量。

## A. CHECKLIST 设计必答

1. **API 选型**：双 Matmul 高阶 API（bmm1 QK^T / bmm2 P·V）+ SoftMax 高阶 API + 事件对编排。bmm1 有 ND/NZ 双输出格式变体（NZ 直供 softmax 省重排）。
2. **算法核心（Flash 在线 softmax，R45 LSE 的分块流式版）**：跨 S2 块维护每行 (softmaxMax, softmaxSum, accO) 状态；新块 max 更新时 **accO 与 sum 按 exp(oldMax−newMax) 重缩放**；softmaxMax/softmaxSum 落 GM 供反向使用（R24 正向留中间量的实例）。
3. **任务描述符流水**：`SplitExtraInfo extraInfo[3]`——三级任务槽让 Cube(bmm1)/Vec1(softmax)/Cube(bmm2) 三阶段解耦流水；每个任务的所有派生参数（各轴索引/尾块/有效长度）打包进槽。
4. **L1 复用配对**：enableL1Reuse 时 blockIdx%2 配对，偶核载 B 进 L1、奇核复用；不配对场景补空循环（needFakePair）。
5. **sparse/causal**：GetS1LoopRange 按掩码三角计算每核有效 s2 循环范围——**跳过全 mask 块**（计算量省一半）。
6. **事件 ID 管理**：`AllocEventID`（独占号，多并行依赖）vs `FetchEventID`（共享号）——B14 的精细化管理。

## 1. 我的骨架（单核，S2 循环在线 softmax）

```
Init: Q/K/V/O GM 视图；行状态 softmaxMax=-inf, softmaxSum=0, accO=0(UB)
for s2块 in [0..S2):
    S = bmm1.Iterate(Q块, K块)            // [s1, s2块]
    S = scale*S + mask                     // 掩码/pse
    m_new = max(softmaxMax, rowmax(S))
    P = exp(S − m_new)
    c   = ΣP
    accO = accO * exp(softmaxMax−m_new) + P·V块   // bmm2 迭代
    softmaxSum = softmaxSum*exp(...) + c
    softmaxMax = m_new
O = accO / softmaxSum
```
