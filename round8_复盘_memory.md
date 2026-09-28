# Round 8 复盘：内存语义专题（TBufPool / 内存复用 / L2 cache）

> 我的实现：`my_impl/round8_memory/`（DESIGN.md / pool_reuse_custom.cpp / op_host/）
> 两阶段练习算子：前半 `z1=x+y`（in-place Add），后半 `z2=x-y`；两阶段 UB 池复用 + 流过数据 L2 bypass
> 对比对象：
> - `reference/samples/.../2_features/2_tbufpool/`（唯一有完整实现的样例；1_l2cache 与 3_memory_reuse 的 README 均为"待补充"）
> - `reference/samples/.../4_best_practices/12_l2_cache_bypass/`（L2 CacheMode 的实际应用样例，README 完整）
> 注：本轮是"技术专题"而非"算子专题"，算子只是三项内存技术的载体。

---

## 一、//?? 清单销案结果

| //?? | 我的猜测 | 实际 API/语义（样例证据） |
| --- | --- | --- |
| 1. 池创建 API | `pipe.InitBufPool(pool, size)` | **形态猜对**：`pipe->InitBufPool(tbufPool0, BUFFER_LENGTH)`。但漏了**池可以嵌套划池**：`tbufPool0.InitBufPool(tbufPool1, size)` |
| 1b. 复用声明 | 发明了 `pool2.SetReuse(pool1, size)` | **真实形态是 InitBufPool 的第三个参数**：`tbufPool0.InitBufPool(tbufPool2, size, tbufPool1)`——创建 pool2 时即声明与 pool1 共享起始地址与长度 |
| 2. 池内分配 | 只想到 TBuf + Get 语义 | 池内 **InitBuffer 直接放 TQue**（`tbufPool1.InitBuffer(srcQue1, BUFFER_NUM_T1, len)`）——队列也能住进池，阶段内流水不受影响 |
| 3. 阶段切换释放 | 猜"直接覆盖即可" | 显式 `tbufPool1.Reset()` / `tbufPool2.Reset()` / `tbufPool0.Reset()` 逐级归还 |
| 4. L2 枚举与粒度 | `CACHE_MODE_DISABLE`（枚举对）；但我"三个 tensor 全 DISABLE" | 枚举正确；**粒度错了**：L2 hint 是选择性保护，不是全局开关（见下） |
| 5. in-place Add | 依据文档重叠约束设计 | 样例未采用（仍用独立 dstLocal）。in-place 属于"文档承诺、样例未示范"，保留为待上板验证项 |
| 6. 串行阶段是否可省队列 | "阶段内串行，TBuf 够了" | **不值得省**：样例把队列放进池里，阶段内的 CopyIn/Compute/CopyOut 流水照常保留。我为了省内存牺牲了阶段内 double buffer，方向错误——池复用已经省了大头 |

## 二、L2 Cache 的正确打开方式（best practice 核心收获）

AddCustom v2 的做法：**只对 y、z 设 `CACHE_MODE_DISABLE`，x 保持 NORMAL**。理由："避免替换已进入 Cache 的 x 数据"。

这说明 L2 bypass 的决策单位是**单个 tensor 的复用特征**：

- 读一次就扔（流过）→ DISABLE，别占 L2；
- 会被再次读（多核共享读、tile 间反复读）→ 保持 NORMAL，甚至希望它常驻；
- **混合场景的精髓：对"流过"的设 DISABLE，恰恰是为了保护"复用"的数据留在 L2 里**。

我"全都 DISABLE"的直觉方案在小算子里无感，但在 L2 紧张的大算子里会把可复用数据一起驱逐，属于典型的"方向对、粒度错"。

## 三、TBufPool 的设计价值（为什么值得学）

对比"每阶段独立分配"与"池复用"两种 UB 预算：

- 无池：UB 需求 = Σ(各阶段 buffer) —— 阶段数一多立即爆 UB；
- 有池：UB 需求 = max(各阶段) —— 多阶段算子（分块归约、多轮迭代、编译器式多 pass）的 UB 规划从"加法"变"最大值"。

配套三个动作：`InitBufPool` 划池（可嵌套）、第三参数声明复用、`Reset()` 归还。队列可入池意味着**阶段间共享内存与阶段内流水并行可以兼得**——这是本轮最大的认知增量。

## 四、经验教训

1. **UB 预算的单位是"阶段"，不是"算子"**：多阶段算子先划分阶段，再做 max 预算，池复用是执行手段。
2. **L2 hint 是保护性工具**：绕开流过数据是为了让复用数据留在缓存里；逐 tensor 决策，别全局开关。
3. **猜 API 时优先考虑"既有机制的位置参数扩展"**（InitBufPool 的复用用第三参数表达），而不是发明新方法名——第 5 轮的"API 三件套"教训的延续：华为风格偏好把语义塞进现有调用的参数表。
4. 阶段内流水不值得为省内存放弃；省内存的正确层次是池复用（跨阶段）。
5. samples 的文档覆盖不均匀（三个样例俩"待补充"）——README 可当文档读，但遇空缺要主动换源（本轮换到 4_best_practices 的 L2 样例补齐）。

## 八轮总览（递进曲线）

| 轮 | 主题 | 最大盲区 |
| --- | --- | --- |
| 1 | Add 基础流水 | tiling 泛化性 |
| 2 | Softmax | 没查高阶 API |
| 3 | ReduceSum | repeat/mask/stride |
| 4 | LeakyReLU | 产品支持矩阵 |
| 5 | Broadcast | API 三件套 / UB 布局是 host 决策 |
| 6 | Matmul/Cube | API 封装边界 |
| 7 | MIX 融合 | 伪代码 ≠ 完整实现（核间脚手架） |
| 8 | 内存语义 | 池复用/L2 选择性保护；"阶段=max 预算" |

**流程修正落实情况**：本轮按第 7 轮行动项预列了"四件检查项"（单核演示不涉及核间偏移/SetTail/早退，已显式在 DESIGN 说明单核边界；workspace 常规处理）——未再犯第 6/7 轮的 P0。下一轮候选：DoubleBuffer 深化与流水调优（0090 页已抓取）、或 AtomicAdd/跨核同步（2_features 5/6）、或回到算子维度练 Gelu/LayerNorm 等复合激活。
