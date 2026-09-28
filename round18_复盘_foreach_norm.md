# Round 18 复盘：foreach_norm（列表 × 归约 × 跨核同步）

> 我的实现：`my_impl/round18_foreach_norm/`（DESIGN.md / foreach_norm_custom.h，预测式两段归约）
> 对比对象：`cann-ops/src/common/inc/foreach/op_kernel_v2/kernel_foreach_reduce.h`（归约工厂）+ `src/foreach/foreach_norm/op_kernel/foreach_norm.h`（449 行独立实现）
> 新结构类：每张量归约输出一个标量——列表结构与 R3 归约、R10 两段归约的交汇点。

---

## 一、结构确认与偏差

**两段归约工厂模型（读后预测）基本正确**：

- Stage1：每核扫自己的张量份额，每张量份额 → 一个 partial（fp32）→ 写 workspace（按 [核][核内张量序] 布局，`coreMiddleOffset + i - tensorStart`）；
- 同步：`CrossCoreSetFlag<0, PIPE_MTE3>(1)` + `WaitFlag(1)` 一对旗，PIPE_MTE3 保证 workspace 写可见；
- Stage2：核 b 跨步认领输出张量 `i = b, b+needCoreNum, ...`，按 `tensorMiddleStartList[i]/tensorMiddleCountList[i]` 读回该张量的全部 partial，二次归约 + Sqrt（sqrt 只在最后一次）；
- 零长度张量 → OutputZero 兜底。

**重要发现：foreach_norm 生产实现是 449 行独立类，不是 reduce 工厂派生**——工厂（KernelForeadhReduce）与独立类并存（foreach_norm 更早/更特化）。"算子工厂"是方向而非全部：**特化程度高的算子仍会脱框手写，但两段归约的骨架模式被完整复刻**。

## 二、我的 //?? 销案 + 反模式（本轮 P1 清单）

1. **`workGM.SetValue(slot, accVal)` 直写 GM 是反模式**：生产走 **标量→S_V 事件对→写 UB tempLocal→DataCopyPad（`DataCopyExtParams{1, sizeof(P)}` 1 元素字节粒度）→GM**。我违反了 R3"标量通路禁令"的 GM 变体——GM 标量写也要 tensor 化搬运。进 CHECKLIST B18。
2. **漏了 S_V 事件对**：生产每处"标量写回 UB 后再进 V 管"都挂 `SetFlag<HardEvent::S_V>`/`WaitFlag`（与 V_S 成对出现）。我只有 V_S/MTE2_V/V_MTE3。CHECKLIST B10 扩展：**V_S 与 S_V 是一对往返**。
3. **跨段累加生产全 tensor 化**：每段 ReduceSum 后 partial 存进 UB 的 tempLocal（按段下标 SetValue），段循环结束后**对 tempLocal 整体再 ReduceSum 一次**——标量通路只剩 GetValue/SetValue 各一次。我的 `accVal += acc.GetValue(0)` 标量累加属 O(段数) 级（CHECKLIST 允许），但生产连这个都消掉了。CHECKLIST B3 精化。
4. **OutputZero**：生产用 SetValueAdapter/Duplicate 写 UB → DataCopyPad 出去；我猜的 `Adds(z,z,0)` 不对（没有任何源数据）。
5. **ReduceSum 惯用形态**：生产用基础 API `ReduceSum<float>(dst, src, work, count)`（src/work 同 tensor 可重叠），不是我写的 WholeReduceSum 模板化调用——列表归约层的社区默认。
6. **fp16 分段 cast**：`dataLocal[index*maxCastDataCount]` 偏移切片 + cast 到 float32Tensor，段间用 ReduceSum 累加——确认 Round 3/12 的偏移切片 idiom 在列表归约层的用法。

## 三、超出模型的两个设计点

1. **ord 参数 → modelCode 模板特化**：`NormAdapter<P, NORM_MODEL_CODE>` 按 ord=1（Abs+无平方无开方）与 ord=2（MulSelf+Sqrt）特化——**数学参数成为编译期模板旗标**（R13 B11"模板特性旗标"的又一实例）。
2. **归约的"幂次可分解性"决定两段结构**：范数可写成一族（ord 次幂和的 1/ord 次方），stage1 算 Σx^ord、stage2 收尾开方——**partial 的定义是数学问题**：只有满足"分段可结合"的量才能两段归约。mean/variance 同理（partial = Σx 与 Σx² 两个 workspace 通道）。

## 四、经验教训（进 CHECKLIST）

1. **GM 标量读写的 tensor 化**（B18）：任何"标量直写 GM"都要改为"标量→S_V→UB→DataCopyPad(1 元素字节粒度)"。
2. **V_S/S_V 成对往返**（B10 扩展）：读归约结果（V_S）后若标量再喂回 V 管线，必须 S_V 回程。
3. **归约流水线的 partial 设计是数学决策**（A 节新增）：分段可结合性 + 收尾算子只做一次（Sqrt/1/ord）。
4. **工厂与独立类并存**：特化算子可脱框手写，但骨架模式（两段+同步+workspace 布局）保持一致——读生产代码先判"工厂派生 or 独立复刻"。
5. **ReduceSum 基础 API 三元形态**（dst, src, work 同 tensor）是列表归约层默认，WholeReduceSum 留给整块 repeat 场景（R3）。

## 下一轮候选

- `10_communicate_compute_fused` 的官方文档（MC2 样例空壳，但 programug 章节在 10033 抓过——读文档写预测式实现）；
- cann-ops `foreach_addcdiv_scalar`（除法+倒数 idiom 的列表版）或 `foreach_lerp_scalar`（插值，三运算符组合）。
