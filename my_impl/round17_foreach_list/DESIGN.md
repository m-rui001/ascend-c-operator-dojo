# Round 17: foreach_add_list —— 我的设计（读公共层后、写实现前）

> 本轮定位：低成本验证 R16 的"算子工厂"模型。预测要实现的 foreach_add_list（out[i] = x1[i] + alpha·x2[i]）在模板族里的形态，然后对比真实 Ternary 层。

## 0. 读公共层后的模型修正（相对 R16）

1. **类命名按"额外标量元数"而非"是否列表"**：Binary/Ternary/Quaternary 指 op 的参数个数；名字里的 "List" 指**标量本身是列表**（每张量一个标量，`ProcessPlusInLoop` 钩子里 `GetValue(index)`）。
2. **基类是模板方法模式**：流水骨架（CopyIn/Compute/CopyOut 主循环）完全在 `KernelForeachUnary`，子类通过五个钩子注入差异：`CopyInPlus / Compute / CopyOut / BeforeProcess / AfterProcess / ProcessPlusInLoop(index, cursorStart)`。
3. **第二个张量列表**：子类自备第二队列 `InQueue_2`，在 `CopyInPlus` 钩子里搬入；`ProcessPlusInLoop` 里 `SetGlobalBuffer(GetTensorAddr(index, inTensorsPtr_2) + cursorStart)` 逐张量重绑地址。
4. **分段的均匀性**：每张量按 `maxDataCount` 等长分段（尾段 isRemainder 标志），不是我 R16 猜的贪心变长块——tiling 保证除尾段外全部等长。
5. **尾段搬运**：`DataCopyExtParams{1, dataCount*sizeof(T)}` + `DataCopyPadExtParams{false,0,0,0}` 字节粒度收口。
6. **bf16 路径**：`std::conditional_t<bf16, float, T>` + InnerComputer 特化（Cast→op→Cast）。

## 1. 我的实现设计（预测式）

按上述模型写出 foreach_add_list 的完整小实现：

- `AddListOp` 函数对象：`op(out, in1, in2, alpha, count)` = `Mul(tmp, in2, alpha)` + `Add(out, in1, tmp)`（fp32 路径直接做）
- 类 `MyForeachAddList`：第二队列 + CopyInPlus/ProcessPlusInLoop 钩子形态
- 尾段 pad、alpha 从标量 GM 读
- dtype 分发 TilingKey(1=fp16, 2=fp32) + `#if __CCE_AICORE__ == 220` bf16

## 2. 验证问题（复盘时逐条勾销）

1. 我的函数对象签名与真实 `OneScalarTernaryOp<P>` 一致吗？
2. 钩子职责切分（第二队列的 Alloc/EnQue 放 CopyInPlus）我的理解对吗？
3. alpha 与列表元素的乘加次序（社区 Adapter 是 Muls(alpha)→Add 还是 Mul→Add？fp32 路径有无合并优化）？
4. cursorStart 的语义（张量内游标）我理解对吗？
5. bf16 的 `conditional_t` + scalar 读取时机（每张量读一次 vs 每核一次）？
