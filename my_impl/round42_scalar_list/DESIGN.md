# Round 32-R42: foreach_add_scalar_list —— 我的设计（A 节前置）

> 语义：y_i = x_i + w_i，w 是**标量列表**（每张量一个标量）。工厂类 `ForeachOneScalarListBinary<T, P, op, 1, 1>`。
> 已读：公共头钩子实现 + op 实例化。

## A. CHECKLIST 设计必答

1. **钩子验证（本轮主题）**：`ProcessPlusInLoop(index, cursorStart)` 在基类主循环进入每张量前被调用，实现为 `scalarVal = inScalarGM.GetValue(index)`——**逐张量刷新标量**。要点：
   - 标量列表直接 **GM GetValue 读取**（不走 DataCopy+MTE2_S！我 R16/R40 单标量场景反而搬了 32B——N 小时 GetValue 更简；单标量也适用此简化）；
   - bf16 用 `conditional_t` 让 scalarVal 保持 float（R17 E 同款）；
   - 钩子调用点 = 张量粒度（cursorStart 传入供偏移），不是列块粒度。
2. **bufferNum=1, paramsCount=1**：标量列表版实例化参数最简（无 cast 槽需求时 Adds 直接原精度）。
3. **op = Adds 函数指针**（v1 形态，R29 E7 对照）。

## 1. 我的实现

复刻 `MyForeachOneScalarList`：基类模拟（主循环+钩子调用点）+ Compute=Adds + ProcessPlusInLoop 读标量。重点验证钩子时序与 GetValue 简化。
