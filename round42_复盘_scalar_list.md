# Round 42 复盘：foreach_add_scalar_list（标量列表钩子验证）

> 我的实现：`my_impl/round42_scalar_list/`（DESIGN.md / foreach_add_scalar_list_custom.h）
> 对比对象：`common/inc/foreach/op_kernel/foreach_one_scalar_list_binary.h` 钩子实现 + `foreach_add_scalar_list.cpp`

---

## 一、ProcessPlusInLoop 钩子验证（本轮主题，销案 R29 遗留）

1. **调用点**：基类主循环进入**每张量处理前**（`ProcessPlusInLoop(index, cursorStart)`），cursorStart 供偏移计算——张量粒度钩子，非列块粒度。
2. **实现极简**：`scalarVal = inScalarGM.GetValue(index)`——**标量列表直接 GM GetValue 读**，无 DataCopy/无事件对。R16/R40 我为单标量写的"DataCopy 32B + MTE2_S 事件对"在此场景是过度设计（N 小时 GetValue 更简）。**B10/B18 修正：标量读取的成本决策 = 读取次数 × 是否需要 tensor 通路；逐张量低频读用 GetValue 直读**。
3. **bf16 scalarVal 保 float**：`conditional_t<bf16, float, T>`（R17 E 同款），Adds 在原精度（paramsCount=1 无 cast 槽）。

## 二、实例化参数确认

`ForeachOneScalarListBinary<half, half, Adds, 1, 1>`——bufferNum=1（标量列表版最简流水）+ paramsCount=1。**同一工厂类（list_binary）被 add_scalar_list 用 bufferNum=1 实例化**——模板参数就是"流水配置"，工厂类的复用粒度到配置级。

## 三、我的差距

1. 我的基类模拟把 Compute/CopyOut 合并了（生产分离）——验证钩子时序已足够，但暴露出"模拟基类"与真实基类的行为差异需逐钩子对照（诚实记录）；
2. `MAX_SCALARS` 常量与 scalarGM 长度推导是占位（生产由 tiling 的 scalarCount 决定）；
3. fp16 隐式输出 Adds 原地 + V_MTE3 直出——与 R30 纯搬运的退化流水同型。

## 四、CHECKLIST 增量

- **B10/B18 修正**：标量读取成本决策 = 次数 × 是否 tensor 通路；**逐张量低频读用 GM GetValue 直读**（免 DataCopy+事件对），高频/大块才走 UB+事件对。
- **E10（新）**：工厂类的模板参数即"流水配置"（bufferNum/paramsCount/needCopyOut/needTempBuf）——同工厂类可按配置级复用，不必为配置差异另写类。

## 下一轮候选

embedding_bag（bag 维度聚合）或 CHECKLIST E 节与 DECISION_FLOW 对表（小元轮）。
