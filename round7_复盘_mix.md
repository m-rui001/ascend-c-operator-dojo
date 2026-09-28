# Round 7 复盘：MIX 模式融合算子（设计 + 实现 全流程对比）

> 我的实现：`my_impl/round7_mix/`（DESIGN.md / matmul_leakyrelu_custom.cpp / op_host/），Matmul+LeakyRelu，范式路径
> 对比对象：
> - `reference/samples/.../12_matmulleakyrelu_frameworklaunch/MatmulLeakyReluCustom/`（完整工程）
> - `reference/samples/.../22_baremix_kernellaunch/BareMixInvocation/`（底层手动同步）
> 前置功课：官方《融合算子-基础知识/算子实现》（`docs_notes/mix_0049/0050.md`）

---

## 一、对比结果：融合算子存在三代写法（本轮最重要的结构性发现）

| 代际 | 特征 | 证据 |
| --- | --- | --- |
| ① 耦合架构老写法 | 单一 `__global__ __aicore__` 入口，GET_TILING_DATA，Matmul API + 矢量计算顺序写在一个 Process 里 | samples 12（2024）：`Iterate<true>` 循环内 GetTensorC→LeakyRelu→CopyOut |
| ② KFC 范式（9.0 新） | `__global__ __mix__(1,2)` 入口 + `__kfc_workspace__` workspace + TCubeTiling 值传；**AIC/AIV 隔离与同步由框架完成** | 官方文档 0049/0050 伪代码 |
| ③ 底层 BareMix | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` + `if ASCEND_IS_AIC / if ASCEND_IS_AIV` 编译期分支 + `CrossCoreSetFlag<0x2, PIPE_FIX>(3)` / `CrossCoreWaitFlag(3)` 手动同步 | samples 22 |

我的实现按文档选了代际②，方向正确；但三代并存意味着**写融合算子前先确认目标 CANN 版本支持哪套**，samples 12 还停留在①。

### 数据路径的关键差别（融合收益的物理来源）

- 范式②：`GetTensorC<true>` 走 CO2→UB，Cube 结果**不落 GM** 直接给 AIV 激活——这是融合省 GM 往返的机制本体；
- 底层③：AIC `IterateAll` 写 GM → SetFlag → AIV WaitFlag 后**从 GM 抄回 UB** 再激活——显式简单，但把融合最想省的那段 GM 往返又付了出去（还把 C 按核数对半劈给两个 AIV）。
- 结论：手写同步省心，但**数据路径要自己设计**，否则融合的性能收益被同步方式吃掉。

## 二、//?? 清单销案结果

1. **`Iterate<true>` 语义**：样例注释"sync is set true here"——每次 Iterate 内部完成跨核同步（通知 AIC 并等待结果可用）。`GetTensorC<true>(local, false, true)` 首个模板参数 = 结果写 LocalMemory；后两个实参样例未注释，语义仍未完全确认（诚实记入遗留）。
2. **count→偏移映射**：我的公式与样例一致，但样例 `roundM = Ceiling(singleCoreM, baseM)`——**我写成了整除**，singleCoreM 非 baseM 整数倍时我的输出位置会错位，真实 P1。iterateOrder=M 先行确认。
3. **SetDim/SetBlockDim**：样例 host `SetDim(GetCoreNumAiv())`=48、`SetBlockDim(GetCoreNumAic())`=24（分离架构）。我猜的"SetDim = blockDim*2"数值上撞对、语义上撞错——正确理解是 **SetDim=AIV 数，SetBlockDim=AIC 数**，1:2 配比只是结果。
4. **TCubeTiling 值传**：KFC 代际②专用；代际①③都用 GET_TILING_DATA/CopyTiling。另外样例把 **alpha 放进 tiling 下发**（host 从 attr 读），我写成了 constexpr——激活参数应当可配置。
5. **尾块**：`SetTail(mUse, nUse)` + CopyOut 里 `curCopyM/curCopyN/stride` 全部与 singleCore 尺寸取 min 收口；LeakyRelu 的 count 仍用 baseM*baseN，但 UB 缓冲本来就按 baseM*baseN 分配，**多算不越界、CopyOut 不搬运**——尾块安全模式确认。
6. **AIC/AIV 隔离如何实现**：代际②由框架完成（所以我的线性 Process 代码成立）；代际③靠 `if ASCEND_IS_AIC/AIV` 编译期裁剪。我的疑问源于把③的机制套在②上，实际②里我根本不用管。

## 三、我的实现缺陷清单（诚实记录）

1. **P0（第 6 轮教训重犯）**：官方文档的范式伪代码里**没有** CalcOffset/SetTail/usedCoreNum 早退，我照着伪代码写就又把它们丢了——所有核会算同一批 base 块互相覆盖。真实工程（samples 12）里这三样一个不少。**教训升级：文档伪代码是教学骨架，完整工程才包含全部核间脚手架；上轮的教训没有转化为"写 kernel 前的检查清单"是我的流程漏洞。**
2. roundM 整除 bug（见上）。
3. alpha 写死 constexpr，应进 tiling。
4. 我的 Init 没有 `SetOrgShape`（样例在 kernel Init 里设 M,N,Ka,Kb）；queue 用了 BUFFER_NUM=2，样例用 1（与 Cube 逐块同步的节奏匹配，双缓冲无意义——第三次见到 BUFFER_NUM=1）。
5. 样例 MatmulType 的 C 位置用 `TPosition::VECIN`，9.0 文档用 `LCM`——版本演进，写法要跟配套版本走。

## 四、经验教训

1. **三代写法并存**：选代际 = 选 CANN 版本 + 同步复杂度 + 数据路径需求的综合决策。
2. **融合收益 = 省一次 GM 往返**：范式②用 GetTensorC<true> 实现"片上交接"；底层③手写同步容易把这段收益还回去。判断融合算子好坏先看中间结果有没有落 GM。
3. **文档伪代码 ≠ 完整实现**：核间脚手架（CalcOffset/SetTail/早退）从不出现在教学代码里，但缺了就是 P0。行动项：把它列进我每轮 kernel 的必检清单（第 6 轮已犯，第 7 轮重犯，必须流程化）。
4. 尾块安全模式 = SetTail 裁剪 + CopyOut 收口 + 计算允许多算（UB 够大）三件套。
5. alpha 等激活系数走 attr→tiling，不进 constexpr。

## 七轮总览

| 轮 | 算子 | 新问题类 | 最大盲区 |
| --- | --- | --- | --- |
| 1 | Add | 基础流水 | tiling 泛化性 |
| 2 | Softmax | 跨片归约 | 没查高阶 API |
| 3 | ReduceSum | 硬件归约 | repeat/mask/stride |
| 4 | LeakyReLU | attr/选型 | 产品支持矩阵 |
| 5 | Broadcast | 多维语义 | API 三件套/host 决定 UB 布局 |
| 6 | Matmul | Cube 通路 | API 封装边界（核间仍是开发者的） |
| 7 | MIX 融合 | 跨核协同 | 文档伪代码 ≠ 完整实现（核间脚手架缺席） |

**模式识别**：第 6、7 轮的 P0 是同一个根因——"高阶抽象替我做了多少"被高估。收敛方向已不是知识而是**流程**：写 kernel 前先列"核间/尾块/早退/workspace 四件检查项"，与 //?? 清单并列。
