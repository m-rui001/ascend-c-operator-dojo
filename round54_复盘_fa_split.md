# Round 54 复盘：FA 变体多核切分对比（轴序即切分策略）

> 我的实现：`my_impl/round54_fa_split/`（DESIGN.md / axis_decomposer.h，AxisDecomposer 可配置轴序分解）
> 对比对象：`cann-ops-adv/.../flash_attention_score_s1s2_bn2gs1.h`（R51 主变体）vs `flash_attention_score_bn2gs1s2_b.h`（1570 行）

---

## 一、两变体切分策略对比

| 维度 | s1s2_bn2gs1（R51 主变体） | bn2gs1s2_b（本轮） |
| --- | --- | --- |
| multiCoreInnerIdx 语义 | S1×S2 展开优先（sparse 裁剪 GetS1LoopRange） | **boIdx = multiCoreInnerIdx 直映射**（B 外层） |
| 数据局部性 | 相邻任务共享 Q 行（S1 方向） | 相邻任务共享 **B/N2（KV cache 局部性）** |
| 尾部排空 | extraInfo[3] 空任务 | `multiCoreInnerLimit += 2` 同型（+流水深度-1） |
| sparse 支持 | GetS1LoopRange 跳全 mask 块 | GetS1LoopRange 同名共存（两变体各自实现） |

**轴序即切分策略**：同一组轴（B,N2,G,S1,S2），线性任务号的分解序不同 → 相邻任务的数据局部性完全不同（B 优先共享 KV，S1 优先共享 Q）——切分策略选择 = **哪个张量该被相邻任务复用**。

## 二、尾部空任务的普遍化

`multiCoreInnerLimit += 2`（B 变体）与 R51 的 needFakePair 空循环同源：**任务数 = 有效任务 + 流水深度 − 1**——三级任务槽（extraInfo[3]）的排空代价被显式编入任务循环。B36 的"排空成本"在 FA 里是具体数字。

## 三、我的实现与生产的映射

`AxisDecomposer` 用 div/mod 链把线性任务号分解为 5 维轴索引（R17 CalcOffset 的 5 轴泛化），AxisPriority 可配置——生产两个变体即两个轴序配置 + 各自的 sparse 裁剪。**变体差异被压缩为"轴序 + 裁剪策略"两个参数**（R40 变体分类学的延续：变体维度=自由度）。

## 四、CHECKLIST 增量

- **B48（新）**：多核切分 = 线性任务号按轴序分解（div/mod 链）；**轴序选择 = 相邻任务该共享哪个张量的局部性**（KV 局部→B/N2 优先，Q 局部/sparse 跳块→S1 优先）；任务数 = 有效任务 + 流水深度 − 1（尾部空任务排空）。

## 下一轮候选

FA 小元轮（R51-R54 四轮入 DECISION_FLOW）或 asc-devkit examples（GitCode，basic_api 系列的现代版对照）。
