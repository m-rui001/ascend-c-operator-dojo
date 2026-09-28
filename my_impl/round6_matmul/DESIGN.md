# Round 6: Matmul 算子 —— 我的设计（写代码前，未看社区实现）

## 0. 前置功课（Cube 通路文档概览）

与矢量通路完全不同的体系，本轮读到的关键模型：

1. **多级存储数据流**：GM → A1/B1(L1 缓存整块) → A2/B2(L0A/L0B 切分小块) → CO1(L0C 单块结果) → CO2 → GM。A1/B1 是"预载缓存"，A2/B2 才参与 Cube 计算。
2. **三级切分**：多核（singleCoreM/N/K，A 沿 M 切、B 沿 N 切、C 分块对应）→ 核内（baseM/baseN/baseK 沿 K 累加）→ Iterate 自动偏移（iterateOrder 决定 M 优先还是 N 优先）。
3. **NZ 格式**：Cube 单元要分形对齐（N 字形分形外序 + z 字形分形内序），B 矩阵转 NZ 是高性能路径。
4. **Matmul 高阶 API**（确认存在，选型结论：用它）：
   - host：`matmul_tiling::MultiCoreMatmulTiling`——SetDim(GetCoreNumAic())、SetAType/BType/CType/BiasType(TPosition+CubeFormat+DataType)、SetShape/SetOrgShape、SetBufferSpace(-1,-1,-1)、EnableBias、GetTiling(tilingData)
   - kernel：`#define ASCENDC_CUBE_ONLY` + `#include "lib/matmul_intf.h"`；`MatmulType<TPosition, CubeFormat, T>` 四件套 typedef；`Matmul<aType,bType,cType,biasType> mm`；`REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling)`；SetTensorA/SetTensorB/SetBias；`while (mm.Iterate()) { mm.GetTensorC(gm_c); }` 或 `mm.IterateAll(gm_c)`；`mm.End()`
5. **系统 workspace**：Matmul 内部实现要用系统 workspace，host 在 workspaceSizes 里加 `GetLibApiWorkSpaceSize()`；非框架工程 kernel 侧要 `SetSysWorkspace(workspace)` 并判空。
6. Batch Matmul：`IterateBatch` 一次算多个小 Matmul，四种 Layout；小 shape 批量场景专用。

## 1. API 选型

| 候选 | 结论 |
| --- | --- |
| Matmul 高阶 API（Matmul 对象 + Iterate） | ✅ 唯一现实选择。手写 Cube 指令级（mda 指令/fixpipe）超出"单轮练习"合理范围，且高阶 API 封装的正是本轮文档讲的三级切分 |
| 矢量通路手搓矩阵乘 | 只作认知对照：向量单元做 GEMM 意味着放弃 Cube 吞吐，生产不用 |

## 2. 设计

- **算子**：C = A×B + bias，A [M,K] fp16 ND，B [K,N] fp16 ND，C [M,N] fp32 ND（Cube 累加天然 fp32）
- **多核**：host 用 MultiCoreMatmulTiling 自动切 singleCoreM/N（我不手写切分——这正是高阶 API 存在的意义），SetDim = AIC 核数
- **kernel**：纯 Cube 模式（ASCENDC_CUBE_ONLY），IterateAll 简单场景一步到位；bias 按需
- **尾块**：文档说 SetTail 是 kernel 运行时处理尾块的手段——但 singleCore 切分由 tiling 库完成，我假设库内部已处理非整除（//?? 待验证）
- **workspace**：host 计算 sys+user；kernel 判空后 SetSysWorkspace（非框架工程路径）

## 3. 不确定点（//?? 清单）

1. TCubeTiling 在自定义算子工程里怎么进 TilingData？直接塞原始 buffer 还是 TILING_DATA_FIELD_DEF 结构化字段？
2. B 的 ND 输入要不要转 NZ？谁转（host 预处理 / kernel 内 Cast）？ND 输入时 API 内部是否自动处理分形？
3. IterateAll 和 Iterate 循环的真实取舍边界；GetTensorC 直接写 GM 的同步/搬出机制（fixpipe）是否需要我操心
4. kernel 跑在 AIC 上：纯 Cube 模式核函数入口还需要什么声明（KERNEL_TASK_TYPE / AIC 宏）？
5. 多核切 K（文档提到"使能多核切K"）什么时候值得开？
