# Round 52 复盘：FlashAttentionScoreGrad（softmax 状态消费侧）

> 我的实现：`my_impl/round52_fa_grad/`（DESIGN.md / fa_grad_custom.h，消费侧骨架 mini）
> 对比对象：`cann-ops-adv/.../flash_attention_score_grad/`（8 文件，主变体 3114 行）

---

## 一、跨算子状态契约（本轮核心）

正向（R51）写 [S1, 8] 布局的 softmaxMax/Sum（8 = fp32 块对齐，每行状态占一块），反向 `MTE2_SFT` 用 `DataCopyPad{1, s1Inner*8}` **按同布局读回**——**正向输出布局 = 反向输入布局**是跨算子 GM 契约，与 R24"正向留 rstd 给反向"同族但更结构化（布局+尺寸都契约化）。

## 二、生产 grad 的四个可迁移点

1. **`SoftmaxGradFront` 高阶指令**：dS = softmaxGrad(P, dO, sum) 一条指令（含 softmaxSum 重缩放），免手拼导数链——A1 第一级的直接实例（"先查高阶 API"在导数域同样适用）。
2. **三 matmul 转置复用**：dQ=dS·K、dK=dS^T·Q、dV=P^T·dO——`bTypeTranspose` 一个 MatmulType 标志翻转 B 矩阵，同对象覆盖三路。
3. **`MatmulCallBackFunc<DataCopyOutLocal>` 回调写出**：dQ 需要"先落 workspace 再聚合"——用回调定制 Matmul 的输出路径而非另写循环。
4. **确定性双 SyncAll**：dQ 部分和 → SyncAll → 聚合 → SyncAll——B28 三态的 FA 版（跨核两段，两次屏障分段保护）。
5. **layout 三态 stride 处理**：`MTE2_STFGrad` 按 BNGSD/SBNGD/SBHND 推导 DataCopyPad 的 srcStride——非连续 grad 读入的通用化封装。

## 三、我的差距

mini 只做消费侧状态读取与骨架；三 matmul 内部、dropout/pse 适配器、DyncReal/DyncLoop 动态循环结构（动态 shape 的循环参数结构体）均未复刻。**FA grad 3114 行 = 正向 2199 行 × 1.4**——反向的边界处理更多（dQ 聚合+三输出+布局）。

## 四、CHECKLIST 增量

- **B46（新）**：FA grad = 消费 [S1,8] softmax 状态（跨算子布局契约）+ SoftmaxGradFront 高阶指令 + 三 matmul（bTypeTranspose 复用 + MatmulCallBackFunc 定制写出）+ dQ 确定性双 SyncAll 聚合。

## 下一轮候选

flash_attention_score_grad_post.h（553 行后处理——确定性聚合细节）或 BN2GS1S2_B 变体多核切分对比。
