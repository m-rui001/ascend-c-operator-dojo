# Round 17 复盘：foreach_add_list（低成本验证"算子工厂"理解）

> 我的实现：`my_impl/round17_foreach_list/`（DESIGN.md / foreach_add_list_custom.cpp）
> 对比对象：`cann-ops/src/common/inc/foreach/op_kernel/foreach_one_scalar_ternary.h` + `foreach/foreach_add_list/op_kernel/foreach_add_list.cpp`
> 诚实说明：本轮按指示"先读公共层再实现"，属于**引导式验证**而非盲测——验证的是我对 R16 模型的表述是否完整，新知识点集中在 Adapter 层。

---

## 一、"算子工厂"模型验证结果

| R16 模型条目 | 验证 | 结果 |
| --- | --- | --- |
| 三层模板架构（Base/Unary/op 类） | Ternary 同构 | ✅ |
| launch 契约地址表（GetTensorAddr 指针走表） | 第二列表同样走表：`inTensorsGM_2.SetGlobalBuffer(GetTensorAddr(index, inTensorsPtr_2) + cursorStart)` | ✅ 两个列表各自走表，互不干扰 |
| 钩子扩展（CopyInPlus/ProcessPlusInLoop 等） | 第二队列的搬入放 `CopyInPlus`、地址重绑放 `ProcessPlusInLoop` | ✅ 钩子粒度与预期一致 |
| 均匀分段 + 尾段 isRemainder | `index * Base::maxDataCount` + `DataCopyExtParams{1, count*sizeof(T)}` | ✅（纠正我 R16 的"贪心变长块"猜测） |
| bf16 conditional_t + InnerComputer 特化 | 原样出现 | ✅ |

**模型确认有效**：给定新模式（list+list），我能按钩子表推出子类骨架。工厂的扩展成本确实只剩"op 语义函数 + 实例化两行"。

## 二、命名taxonomy（本轮新知）

`Binary/Ternary/Quaternary` 指 **op 的参数元数**，名字里的 "List" 指**标量本身是列表**（每张量一个标量，`ProcessPlusInLoop` 里 `GetValue(index)`）。R16 我把 `foreach_one_scalar_list_binary` 误读为"列表二元"，实为"标量列表二元"。选类前先解码命名规则。

## 三、Adapter 层的新 idiom（本轮真正的收获）

生产 `AddListNormalAdapter`（fp16）：

```cpp
Muls(srcLocal2, srcLocal2, scalarVal, uValue);  // in-place：src2 *= alpha
PipeBarrier<PIPE_V>();                           // 跨指令同缓冲依赖，强制同步
Add(dstLocal, srcLocal1, srcLocal2, uValue);
```

生产 `AddListFloatAdapter`（fp32）：

```cpp
Axpy<T, T>(srcLocal1, srcLocal2, scalarVal, uValue);   // 单指令：src1 += alpha*src2
if (dstLocal.GetPhyAddr() != srcLocal1.GetPhyAddr()) {  // 判物理别名
    PipeBarrier<PIPE_V>();
    DataCopy(dstLocal, srcLocal1, roundup32(uValue));   // UB→UB 拷贝，向上取整 32B
}
```

四个新 idiom：

1. **`Axpy` 融合指令**：`y += alpha*x` 一条搞定，替代我的 Muls+Add 两条（API 探测时见过名字，本轮才知道用在这）——乘加场景的选型清单要加它；
2. **`PipeBarrier<PIPE_V>` 出现在生产代码**：R12 CHECKLIST B9 的"跨指令同缓冲依赖"在生产的 Adapter 里逐对遵守，验证条目有效性；
3. **`GetPhyAddr()` 物理别名判断**：隐式输出优化（dst 可能与 src1 同址）——同址跳过拷贝，异址才 UB→UB DataCopy；这是"判断要不要拷"的通用范式；
4. **UB→UB DataCopy 向上取整 32B**：多拷的尾料留在 UB 里无害，规避尾块 pad 复杂度（B12 Adds-0 拷贝 idiom 的等价形式）。

## 四、我的实现差距（对照真实 Ternary + Adapter）

1. 乘加次序：我 `Muls(out, in2) → Add(out, in1, out)` 让 out 提前落笔且 dst 与第二源重叠——正确性靠重叠约束，但比生产 fp32 路径多一条指令、比 fp16 路径少一个显式 barrier（我靠 Add dst=第二源的重叠合法性规避，语义成立但非社区风格）；
2. 函数对象形态：我用 `struct + operator()`，生产用**普通函数 + 函数指针非类型模板参数**（`OneScalarTernaryOp<P>* op`）——语义等价，但生产形态让 Adapter 可以是普通自由函数；
3. 我的主循环自写（按模型复述），真实架构中它全在 `KernelForeachUnary` 基类——这正是工厂的意义：子类永远不写主循环。

## 五、CHECKLIST 增量（B 节新增 15/16/17）

15. **Axpy 优先**：`y += alpha*x` 场景先查 Axpy，禁手拼 Muls+Add（fp32 单指令；fp16 经 Adapter 组合）。（R17）
16. **别名判断跳拷贝**：dst 与 src 可能同址时用 `GetPhyAddr()` 比较，同址零拷贝、异址 UB→UB DataCopy 向上取整 32B。（R17）
17. **算子族命名解码**：foreach 类名 arity（Binary/Ternary/Quaternary）= op 参数元数；"List" = 标量是列表。选基类先解码命名。（R17）

## 下一轮候选

`10_communicate_compute_fused`（通信计算融合，全新领域，samples 有代码）；或 foreach_reduce（`kernel_foreach_reduce.h` 里出现过同步 API，归约+列表的组合）。
