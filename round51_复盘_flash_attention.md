# Round 51 复盘：FlashAttentionScore（架构研读轮，cann-ops-adv）

> 我的实现：`my_impl/round51_flash_attention/`（DESIGN.md / flash_attention_custom.h，在线 softmax 骨架 mini）
> 对比对象：`cann-ops-adv/src/transformer/flash_attention_score/`（6 变体，主变体 2199 行；本轮成功克隆 cann-ops-adv）
> 定位：FlashAttention 级融合算子的**架构研读轮**——不复刻 L1 复用/sparse 全量，骨架级实现在线 softmax。

---

## 一、FA 的五层架构（生产主变体结构）

1. **Cube×Vector×Cube 三明治**：bmm1（Q·K^T，ND/NZ 双输出格式变体，NZ 直供 softmax 免重排）→ ProcessVec1（scale/pse/mask/在线 softmax，全 UB）→ bmm2（P·V，按 layout 条件选 ND/NZ cType）。
2. **任务描述符流水**：`SplitExtraInfo extraInfo[3]`——三级任务槽，每个任务的全部派生参数（各轴索引/尾块/有效长度/L1 配对信息）打包入槽，Cube 与 Vec 阶段经槽解耦流水。**"任务描述符流水"是新结构**：比事件对更高层的阶段解耦。
3. **在线 softmax（flash 核心，R45 LSE 的分块流式版）**：跨 S2 块维护每行 (softmaxMax, softmaxSum, accO)，新块 max 更新时旧状态按 exp(mOld−mNew) 重缩放；softmaxMax/softmaxSum 落 GM **供反向复用**（R24 正向留中间量的实例）。
4. **L1 复用配对**：enableL1Reuse → blockIdx%2 配对，偶核载 B 进 L1 奇核复用；不配对补空循环（needFakePair）。**核间协作在编译期旗标+运行时配对**。
5. **sparse/causal 循环范围**：GetS1LoopRange 按掩码三角算每核有效 s2 范围——**跳过全 mask 块省一半计算**。

## 二、事件 ID 管理的精细层级（B14 深化）

生产在 ProcessVec1 单函数内 **AllocEventID × 5**（MTE2_V/V_MTE2×3/MTE3_V）+ FetchEventID × 2——`Alloc`（独占号，函数内多并行依赖各领一号）vs `Fetch`（共享号，串行依赖复用）。R14 的"每笔依赖领一个号"在此升级为**显式分配/获取两级 API**。

## 三、我的骨架与生产的差距（诚实清单）

1. 生产 extraInfo 三级流水 + L1 复用 + sparse 范围——mini 全部未复刻（单核串行）；
2. mini 的 S/P 逐元素标量读（GetValue(pRow[j]) Axpy）应 tensor 化为 bmm2/矩阵运算——示意级；
3. pse/drop/padding 前缀等适配器（drop_mask_adapter 独立头）未涉及；
4. 2199 行主变体中大约 60% 是尾块/布局/配对特判——FA 的复杂度在**组合爆炸的边界处理**而非核心算法。

## 四、CHECKLIST 增量

- **B45（新）**：FlashAttention 级 = 双 Matmul 夹 Vector 三明治 + 在线 softmax（跨块状态 (m,sum,accO) + rescale）+ softmaxMax/Sum 落 GM 供反向；任务描述符流水（extraInfo 槽）解耦三阶段；L1 复用核配对；sparse 循环范围跳全 mask 块。
- **B14 深化**：事件 ID 两级——AllocEventID（独占）/FetchEventID（共享）。

## 下一轮候选

FA grad（dQ/dK/dV 反向，softmax 状态复用）或 flash_attention_score 的 BN2GS1S2_B 变体对比（多核切分策略差异）。
