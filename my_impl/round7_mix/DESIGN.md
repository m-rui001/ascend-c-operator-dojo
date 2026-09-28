# Round 7: MIX 模式融合算子 —— 我的设计（写代码前，未看社区实现）

## 0. 前置功课（融合算子文档概览）

新学到的模型：

1. **CV 融合**：Cube 输出喂 Vector（CO2→VECIN），Vector 输出喂 Cube（VECOUT→A1/B1）。融合收益 = 省掉 GM 往返 + 双单元流水并行 + 省算子调度。
2. **范式路径**（框架管 AIC/AIV 隔离与同步）：核函数入口 `__global__ __mix__(1, 2)`（AIC:AIV=1:2），workspace 参数带 `__kfc_workspace__` 限定符，tiling 以**值传 TCubeTiling**；循环体 `while (matmulObj.template Iterate<true>())` → `GetTensorC<true>(ubLocal, false, true)` 把 Cube 结果搬上 UB → LeakyRelu → EnQue/DeQue → DataCopy 出 GM。
3. **底层路径**（BareMix，自己管同步）：`ASCEND_IS_AIC / ASCEND_IS_AIV` 编译期隔离 + `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` + `CrossCoreSetFlag/CrossCoreWaitFlag` 手动同步 + ASCENDC_CUBE_ONLY。
4. **host 侧 MIX 规则**（分离模式）：Matmul API 从 AIV 侧发起，Iterate 只是"通知"AIC；**SetBlockDim = AI Core 组数，SetDim = AIV 数**（如 20 组 / 40 AIV）。
5. **C 留在片上**：SetCType 用 TPosition::LCM（=VECCALC/UB），不再写回 GM。

## 1. API 选型

| 候选 | 结论 |
| --- | --- |
| 范式路径（KFC，框架同步） | ✅ 本轮选择：文档明确"隔离与同步由框架完成"，抽象层级最高 |
| 底层路径（BareMix 手动同步） | 留作对比环节研究对象：灵活但要自己管 CrossCore flag |
| 拆成 Matmul + LeakyRelu 两个算子 | 性能次优（GM 往返），只作基线 |

## 2. 算子与设计

- 数学：`c = LeakyRelu(a[M,K]×b[K,N] + bias, alpha)`，a/b half、bias/c float、ND
- 数据流五步：GM→AIC → Matmul → CO2→UB(VECIN) → LeakyRelu(AIV) → UB→GM
- tiling：host MultiCoreMatmulTiling，**SetCType(LCM)** 让 C 的 baseM×baseN 片直接落在 UB（约束：baseM*baseN ≤ UB）；SetBlockDim=AI Core 组数、SetDim=AIV 数
- kernel：`__mix__(1,2)` 入口；while(Iterate<true>) 循环内三段（MatmulCompute/LeakyReluCompute/CopyOut）；CopyOut 用 DataCopyParams 二维块（baseM 行 × baseN 字节 + 行距 stride）
- 系统 workspace：`__kfc_workspace__` 参数 + GetSysWorkSpacePtr

## 3. 不确定点（//?? 清单）

1. `Iterate<true>` 和 `GetTensorC<true>(local, false, true)` 的模板/实参各自什么含义？（猜：与 enPartialOut / enAtomic / enMixedPrec 有关，但没依据）
2. CopyOut 的 `startOffset = count % roundM * baseM * N + count / roundM * baseN`——iterateOrder 与 count→(m,n) 的映射关系我推的是"M 先行"，对吗？
3. `__mix__(1, 2)` 里 AIV 数=2，那 SetDim 应该是 blockDim*2？SetBlockDim 与 SetDim 的换算实例？
4. TCubeTiling 值传进核函数（不是 GET_TILING_DATA）——这是 KFC 场景特例还是通用可行？
5. LeakyRelu 的 count 用 `tiling.baseM * tiling.baseN`：最后一轮 Iterate 的尾块（M/N 非整数倍）会不会多算越界 UB？尾块谁负责（SetTail？Iterate 自动缩？）
6. 范式模式下同一个 Process 代码同时跑在 AIC 和 AIV 上——那 LeakyRelu/DataCopy 等矢量调用在 AIC 上怎么"不存在"？（框架重写？还是宏隐藏？）
