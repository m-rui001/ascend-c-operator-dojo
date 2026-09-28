# Round 53 复盘：FA Grad Post（独立后处理阶段 + NZ→ND）

> 我的实现：`my_impl/round53_fa_post/`（DESIGN.md / fa_post_custom.h，post mini）
> 对比对象：`cann-ops-adv/.../flash_attention_score_grad_post.h`（553 行）

---

## 一、Post 独立阶段模式（本轮核心结构）

主 grad 核（并行、fp32 部分和落 workspace 三段 dq|dk|dv）之后，**rescale+cast+布局转换拆成独立 post 阶段**：

```
主核: fp32 部分和 → workspace[dq|dk|dv 三段偏移]
post: 分块读 workspace → Muls(rescale 合并系数) → Cast(ROUND) → 按 dtype 对齐写出
```

判据：**收尾操作（cast/scale/布局）与主计算耦合会破坏流水**——post 把"聚合+格式化"从热循环剥离。与 R25"单核串行聚合"对照：post 可多核并行（按输出分块），聚合与格式化解耦后两全。

## 二、NZ→ND 布局转换（Copy 三层循环）

`NZ2ND` 用 **`Copy`（UB→UB）+ CopyRepeatParams 四步长**（srcStride/dstStride/srcRepeatSize/dstRepeatSize，块单位）做 c0_repeat × c1_repeat × n_repeat 三层循环重排——**NZ 是 Cube 的母语，ND 是 Vector 的母语，转换放 post**。与 R44 的 GatherMask 抽行、R22 的 DoTranspose 同属"布局转换工具箱"，NZ2ND 用 Copy 因源是 Matmul 的规则分形。

## 三、cast 写出粒度随 dtype

fp32 输出对齐 8 元素、fp16 对齐 16 元素（均 32B）——`(n+7)/8*8` vs `(n+15)/16*16`。B2 对齐规则的 dtype 维补全。

## 四、CHECKLIST 增量

- **B47（新）**：确定性聚合+格式化拆独立 post 阶段（workspace 三段偏移 dq|dk|dv；post 按输出分块多核并行）；NZ→ND 用 Copy+CopyRepeatParams 四步长三层循环；cast 写出对齐随 dtype（fp32=8 元素、fp16=16 元素）。
- **B47 关联**：R25（聚合三态）→ R52（双 SyncAll）→ R53（post 阶段）——确定性聚合的三级演进：同核内 → 跨核屏障 → 独立阶段。

## 下一轮候选

BN2GS1S2_B 变体对比（多核切分策略差异）或 attention 系收尾小元轮（FA 三轮入 DECISION_FLOW）。
