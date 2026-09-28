# Round 6 复盘：Matmul/Cube 通路（设计 + 实现 全流程对比）

> 我的实现：`my_impl/round6_matmul/`（DESIGN.md / matmul_custom.cpp / op_host/）
> 对比对象：
> - `reference/samples/operator/ascendc/0_introduction/10_matmul_frameworklaunch/MatmulCustomMultiCore/`
> - `reference/cann-ops/src/matmul/mat_mul_v3/`（生产级，概览）
> 前置功课：官方 Matmul 章节《基础知识》《算子实现》《Batch Matmul》（`docs_notes/p_0037/0038/0041.md`）

---

## 一、前置功课与设计回顾

文档概览已把 Cube 通路的关键模型讲清：GM → A1/B1(L1) → A2/B2(L0) → CO1 → CO2 → GM 的多级数据流、三级切分（singleCore → base → K 累加）、NZ 分形格式、Matmul 高阶 API 的 host/kernel 两侧调用序列。API 选型结论（唯一现实选择：Matmul 高阶 API）在对比后仍成立。

## 二、对比结果：//?? 清单销案

| //?? | 我的设计假设 | 实际答案（样例证据） |
| --- | --- | --- |
| 1. TCubeTiling 如何进 tiling | "原始 buffer 透传" | **内嵌结构体**：`struct MatmulCustomTilingData { uint64_t localMemSize; AscendC::tiling::TCubeTiling cubeTilingData; }` + `REGISTER_TILING_DEFAULT` + `GET_TILING_DATA`；host 用 `context->GetTilingData<T>()` 类型化指针直接填字段（我那套 SaveToBuffer 手工序列化完全没必要） |
| 2. B 的 ND→NZ 谁来转 | 存疑 | **API 内部转**，但需要 kernel 提供一片 UB 转换工作区：TilingKey=2 路径里 `InitBuffer(tmpMMFormatUb, localMemSize)` + `matmulObj.SetLocalWorkspace(...)`，localMemSize=UB 大小由 host 下发 |
| 3. IterateAll vs Iterate | 简单场景 IterateAll | 确认。GetTensorC/循环留给"迭代控制"场景 |
| 4. AIC 入口声明 | 猜"纯 Cube 默认跑 AIC" | 样例无 KERNEL_TASK_TYPE 宏，但暴露了更深的坑：**分离架构上 SetDim 与 SetBlockDim 指向不同核种**——910B 上 `SetDim(GetCoreNumAiv())`=48（矢量核数），`SetBlockDim(24)`（Cube 核数，AIC:AIV=1:2）。我写的 `SetDim(GetCoreNumAic())` 恰好反了 |
| 5. 多核切 K | 未设计 | 基础样例未启用；生产库有专门的 splitk 变体（见下） |

## 三、设计层最大的误判：高阶 API ≠ 全自动

我的 kernel 直接 `IterateAll(cGlobal)`，默认"核间分配由 API 内部完成"。**错**。样例证明：

1. **核间映射是开发者的活**：`CalcOffset()` 用 blockIdx 反解出 (mIdx, nIdx)，手动算 A/B/C/Bias 四个 GM 偏移；
2. **尾块也是开发者的活**：每核算 `tailM = M - mIdx*singleCoreM`，`SetTail(mUse, nUse)` 裁剪本核实际形状；
3. IterateAll/Iterate 只负责**单核内部**的三级流水与搬运——API 的封装边界是"单核"，不是"整个算子"。

后果：我的版本若真跑起来，**所有核都会算全量 C 互相覆盖**——P0 级错误。这是六轮以来最大的单点误判，根源又是"替 API 想象了它没承诺的语义"。Round 2 的教训（查 API 边界）在 Cube 语境下换了个形态再次出现。

其他值得记录的设计层差异：

- **GlobalTensor 偏移链**：`aGlobal = aGlobal[offsetA]`——GlobalTensor 支持直接赋值偏移出新视图，比反复手算指针偏移优雅，我不知道这个 idiom；
- host 与 kernel **共享同一个 tiling 头文件**（`op_kernel/matmul_custom_tiling.h` 被 op_host include），保证两侧字段不漂移；我按前几轮习惯写成了两份；
- `usedCoreNum` 早退模式第三次出现（第三次验证它是标准模式）。

## 四、生产级概览（cann-ops mat_mul_v3）

生产 Matmul 的形态是**变体组合 + host 策略选择**：

- op_kernel 下 14 个变体头文件：base / l1_full_load / deterministic_splitk / sc_splitk / nd2nz / unaligned / fixpipe 优化算法……
- op_host 下 10+ 文件：base_tiling、l0 专项、l2_cache 专项（**L2 缓存感知切分**）、deterministic 开关……
- 即：工业 Matmul 不是"一个更好的 kernel"，而是 host 依据 shape/SoC/内存布局从变体族里路由（TilingKey），每条路针对一类瓶颈（K 太长→splitK、B 不分形→nd2nz、确定性需求→deterministic）。

这印证了 Round 1 学到的"Tiling 是 host 侧运行时函数"，但量级完全不同：矢量算子的 tiling 输出的是切分参数，Matmul 的 tiling 输出的是**执行策略**。

## 五、经验教训

1. **先问 API 的封装边界**：Matmul 高阶 API 封装"单核内"的一切，核间映射/偏移/尾块仍归开发者。任何"自动化"都要问清边界在哪里。
2. TCubeTiling 是可内嵌的结构体，配 `GetTilingData<T>()` 类型化接口；host/kernel 共享 tiling 头文件。
3. 分离架构（AIC/AIV）上 SetDim 和 SetBlockDim 语义不同，文档小字决定正确性。
4. 给 API 传 ND 输入要预留 UB 转换工作区（SetLocalWorkspace），host 侧 TilingKey 路由有无 workspace 两条路径。
5. GlobalTensor 可偏移赋值成子视图。
6. 生产 Matmul = 变体族 + host 路由；学 Matmul 的顺序应是"单核数据流 → 高阶 API → 变体族何时被触发"。

## 六、下轮方向（如继续循环）

正交空间里剩下的候选：**FixPipe/原子操作（AtomicAdd 累加 C）**、**融合算子 MIX 模式**（AIC+AIV 协同，文档提到 ASCEND_IS_AIC/AIV 宏）、或回头补内存语义（TBufPool/内存复用/L2 bypass）。MIX 模式是 Cube 与矢量两轮知识的交汇点，优先。
