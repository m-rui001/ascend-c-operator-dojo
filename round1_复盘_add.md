# Round 1 复盘：Add 算子（设计 + 实现 全流程对比）

> 规则回顾：先读官方文档概览（不含社区代码）→ 自己做算子设计并实现 → 再对比社区实现 → 总结。
> 我的实现：`my_impl/round1_add/`（DESIGN.md / add_custom.cpp / main.cpp / CMakeLists.txt）
> 对比对象：
> - 教学样例：`reference/samples/operator/ascendc/0_introduction/1_add_frameworklaunch/AddCustom/`
> - 生产级实现：`reference/cann-ops/src/math/add_custom/`（CANN 官方开源算子库，2025 开放共建，200+ 算子）

---

## 一、设计层对比

| 设计维度 | 我的实现 | samples 教学样例 | cann-ops 生产实现 |
| --- | --- | --- | --- |
| shape 泛化 | 只适配 (8,2048) 整除场景，totalLength 假设能被核数×tileNum 整除 | 同样是教学简化（BLOCK_DIM=8、TILE_NUM=8 写死） | **任意 shape**：按 32B 对齐后的字节长度参与所有切分计算 |
| dtype 泛化 | 硬编码 `float` | 用 `DTYPE_X/Y/Z` 宏（host 侧注册的 dtype 自动生成） | **模板参数** `TYPE_X/Y/Z` + 4 种 dtype（fp32/fp16/int32/int8），int8 有专用计算路径 |
| 核间切分 | 均分：`blockLength = totalLength / GetBlockNum()` | 同左（教学版） | **大小核非均分**：32B 块数除核数有余数时，前 `tailBlockNum` 个核多搬一块（bigCore），其余 smallCore |
| 核间切分上限 | 没考虑 | 没考虑 | **核数上限 = 32B 块数**：数据极小时宁可少开核，保证每核至少 32B（DataCopy 的最小搬运粒度）；数据小于一块 UB 时直接 `coreNum=1` 单核兜底 |
| UB 容量意识 | tileLength 从 shape 推导，完全没算 UB 能装多少 | 同左 | **host 侧查询硬件 UB 大小**（`PlatformAscendC::GetCoreMemSize`），`ubPartDataNum = UB/份数/BUFFER_NUM/32B×32B/dtypeLen`，从硬件容量反推每片大小 |
| 尾块处理 | **完全没有** | **没有**（教学版故意回避） | 双层尾块：核间尾块（大小核）+ 核内尾循环（`tailDataNum`，最后一次循环用 `processDataNum=tailDataNum`） |
| tiling 如何传给核函数 | 值传结构体（QuickStart 教程的"Kernel 直调"风格） | `GM_ADDR tiling` + `GET_TILING_DATA` 宏，host 侧 `BEGIN_TILING_DATA_DEF` 宏注册、框架自动序列化 | 同 samples，另加 **TilingKey**（0/1 区分有无大核，kernel 侧 `TILING_KEY_IS` 分支） |
| 参数校验 | 无 | 无 | `ASSERT(GetBlockNum() != 0)`、`coreNum==0` 返回 GRAPH_FAILED |
| Host 工程形态 | 手写 aclrtMalloc + `<<<>>>` 直调 | 完整 op_host 工程：OpDef 注册（输入输出/dtype/format/UnknownShapeFormat）、InferShape、InferDataType、TilingFunc、SetBlockDim、workspace | 同左，dtype/format 列表更全，支持动态 shape |

### 设计层结论

我的实现本质上是"**把教程的固定尺寸样例默写了一遍**"，而生产实现回答的是另一层问题：

1. **算子设计的第一问不是"怎么算"，而是"输入长什么样"**。任意 shape、任意 dtype、不整除、不对齐、数据只有几个字节——这些边界才是 tiling 函数存在的意义。教程为了讲清流水线，把最难的部分（tiling）简化掉了，我照着教程写就继承了它的所有盲区。
2. **Tiling 是 host 侧的一个运行时函数，不是编译期常量**。硬件参数（UB 容量、核数）要向平台查询，shape 要从 context 读，32B 对齐贯穿始终。我写成了"launch 时传个结构体"，只对直调场景成立。
3. **数据流设计里 32B 是魔法数字**：DataCopy 的最小对齐粒度。它同时决定了"每核最少分多少数据"和"ubPartDataNum 怎么取整"，生产代码里每个切分量都围绕它取整。

---

## 二、代码层对比（对照我的 `add_custom.cpp`）

### P0 —— 正确性问题（我的实现会算错/崩）

1. **没有尾块处理**。`totalLength=8×2048` 恰好整除才正确。只要 shape 是 (7,2048) 或 (1000,)，我的 `blockLength = totalLength/GetBlockNum()` 向下取整后**直接丢数据**。cann-ops 用大小核 + 尾循环两套机制兜住。
2. **没有核间偏移的对齐保障**。我按元素数均分，奇数 shape 下第二个核的起始地址 `blockLength*GetBlockIdx()*4` 可能不是 32B 倍数，DataCopy 行为未定义。

### P1 —— 性能/资源问题

3. **UB 预算错误**：我 `tileLength = blockLength / tileNum`，但 `InitBuffer` 又乘 `BUFFER_NUM=2`，每队列占用 = 官方语义的 2 倍。官方样例的语义是 `tileLength = blockLength / tileNum / BUFFER_NUM`、`loopCount = tileNum * BUFFER_NUM`——"每核 tileNum 个大块，每块劈两半做 double buffer"。数据放大后我的版本会先撑爆 UB。
4. **长度用 `uint32_t`**：cann-ops 全部用 `uint64_t`，大数据或字节计算中间量有溢出风险。

### P2 —— 工程规范差距

5. **没有 dtype 抽象**：官方用 `DTYPE_X` 宏（框架按 OpDef 注册的 dtype 注入），cann-ops 用模板。我硬编码 float，一个文件只支持一种 dtype。
6. **没有 `add_custom_do` 封装和 `#ifndef ASCENDC_CPU_DEBUG` 保护**——这个封装让 host 调用代码不接触 `<<<>>>`，同时保留 CPU 孪生调试能力（CPU 上跑同一个 kernel 验证逻辑）。
7. **int8 等低精度路径缺失**：cann-ops 对 int8 先 `Cast` 到 half 相加，再用 `CAST_RINT + ShiftLeft/ShiftRight` 做饱和截断回 int8。矢量核没有 int8 加法指令，**高精度中间类型 + 饱和回转**是通用套路，我完全不知道这个 idiom。
8. API 细节：TQue 的 position 参数官方样例用 `TPosition::VECIN`，cann-ops 用 `QuePosition::VECIN`（版本演进，两者都存在）；cann-ops 额外的临时空间用 `TBuf<QuePosition::VECCALC>` 而不是 TQue（不需要队列同步的暂存用 TBuf）。
9. `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`：9.0 QuickStart 教程里的宏，声明核跑在 AIV（矢量核）上，我的实现漏了。

### 我写对的部分

- SPMD 多核偏移（`SetGlobalBuffer` + `GetBlockIdx()` 偏移）、三级流水 CopyIn/Compute/CopyOut 骨架、Alloc/EnQue/DeQue/Free 的配对顺序、double buffer 的 BUFFER_NUM=2、`extern "C" __global__ __aicore__` 核函数签名——与两份社区实现一致，说明文档概览足够支撑正确骨架。

---

## 三、经验教训

1. **教程样例 ≠ 设计模板**。入门教程固定 shape、固定 dtype、回避尾块，是为了讲流水线。真实算子的复杂度 80% 在 tiling 和边界，不在计算本身。
2. **tiling 的输入是"硬件容量 + shape + dtype"三元组**，输出是"每核数据量 + 循环次数 + 尾块量 + workspace + TilingKey"。写算子先写这个函数。
3. **32B 对齐是整个数据切分体系的锚点**（DataCopy 最小粒度），从核数上限到 ubPartDataNum 都围着它转。
4. **泛化用模板 + TilingKey**：dtype 走模板参数和 `DTYPE_X` 宏，结构性分支（有无大核）走 TilingKey + `if constexpr`，避免运行时开销。
5. **低精度类型没有对应矢量指令时**，Cast 到高精度算，再饱和回转。
6. **防御式 tiling**：除零 ASSERT、coreNum=0 直接返回失败、数据太小降为单核。
7. 生产代码的浮点之外细节全是 `uint64_t`。

## 四、带入 Round 2 的行动点

- 选 softmax：它有 Add 没有的两个新问题——**跨 tile 的依赖**（max/sum 要在整个行上归约，tile 之间不能独立）和**行方向 shape/stride 处理**。正好检验我有没有真正吸收"尾块 + 泛化"的教训。
- 写之前先列 UB 预算表（几个 tensor × 大小 × BUFFER_NUM）。
- host 侧按 op_host 工程结构写（TilingFunc 动态算 tiling），不再写直调版。
