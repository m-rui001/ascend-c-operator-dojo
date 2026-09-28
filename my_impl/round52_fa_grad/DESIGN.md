# Round 52: FlashAttentionScoreGrad —— 我的设计（A 节前置）

> 语义：dQ/dK/dV = 反向传播。数学：dS = softmaxGrad(P, dO)（用正向 softmaxSum 重缩放）；dQ = dS·K；dK = dS^T·Q；dV = P^T·dO。
> 已读：`flash_attention_score_grad_s1s2_bn2.h`（3114 行）关键段 + post 头。

## A. CHECKLIST 设计必答

1. **状态消费契约（R51 的对偶）**：正向写 [S1, 8] 布局的 softmaxMax/Sum（8=fp32 块对齐），反向 `MTE2_SFT` 用 `DataCopyPad{1, s1Inner*8}` 原布局读回——**跨算子 GM 布局是显式契约**（正向的输出布局=反向的输入布局）。
2. **`SoftmaxGradFront` 高阶指令**：dS = softmaxGrad(P, dO, sum) 一条指令（含重缩放），免手拼 log-softmax 导数链——A1 第一级的高阶 API 实例。
3. **三 Matmul 反向**：dQ=dS·K、dK=dS^T·Q、dV=P^T·dO——`bTypeTranspose` 转置复用；`MatmulCallBackFunc<DataCopyOutLocal>` **回调自定义写出**（dQ 需跨核聚合前先落 workspace）。
4. **确定性两段**：dQ 部分和 → SyncAll ×2 → 聚合（B28 三态在 FA grad 的实例）。
5. **layout 三态**（BNGSD/SBNGD/SBHND）的 grad 读入 stride 处理——非连续输入的 DataCopyPad 通用化（MTE2_STFGrad）。

## 1. 我的实现（单核 mini：dS→dQ/dK/dV 三 matmul 抽象骨架）

消费 (softmaxMax, softmaxSum) → dS → 三路 matmul。
