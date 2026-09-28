# Round 28 复盘：AddRmsNormDynamicQuant（动态量化输出对）

> 我的实现：`my_impl/round28_quant/`（DESIGN.md / dynamic_quant_custom.h，量化块 mini 版）
> 对比对象：`add_rms_norm_dynamic_quant/op_kernel/add_rms_norm_dynamic_quant_helper.h`（两个带性能注释的 ReduceMax）+ normal_kernel.h 量化主流程

---

## 一、量化输出对与数值契约链（R27 链的延伸）

5 输出动态量化：`scale1Out = row_max(abs(y))/127`，`y1Out = round(y/scale)`（int8），加上 xOut/rstdOut/y2Out。**量化输出对（int8 值 + fp32 scale）是 LLM 推理的标准出口契约**——反量化方用 `y1×scale` 还原。融合链的截断点清单升级为：xOut(T) → yNorm(T 回环，R27) → **y1(int8)+scale(fp32)**。每多一种输出格式，融合的舍入路径就多一个必须逐位复现的点。

## 二、两个 ReduceMax 实现（生产注释即性能教材）

| 实现 | 机制 | 生产注释 |
| --- | --- | --- |
| `ReduceMaxFP32` | mask-count 模式 + `get_max_min_cnt()` 读累加器 | "**very very slow**" |
| `ReduceMaxInplace` | **strided Max 折叠树**：`Max(v, v[64], v, 64, reps-1, {1,1,1,0,8,0})`——dstRepStride=0 反复写同块、srcRepStride=8 逐 64 元素组推进，把全行 max 折进前 64 元素，再 `WholeReduceMax` 一次收口 | "about **20us faster** in case fp16:(1024,11264) on 910B" |

结构上它是 R26 折半求和的 **max 版**：树形折叠思想相同，但用 Max 指令的 repeat 参数一次折叠一组（比逐次 Add 快）。**R22 的 `Duplicate(1)+Div`、R26 的折半 Add、本轮的折叠 Max——同一族"树形折叠"在大归约上的三种算术变体**，按运算类型选指令。

## 三、量化路径的选型确认

1. **scale = 127/max 逐行标量除法一次**，然后 `Muls(y, 1/scale)` 广播——除法只在标量域做一次（R20 B8：标量除数→乘倒数），行内是乘法；
2. **Abs 用独立载体**（保 y 原值供第二遍除法）——B14 缓冲配对（max 不能原地覆盖 y）；
3. int8 两段 cast（fp32→half→int8）——**无直接 fp32→int8 cast 指令可用时的中转**（或精度模式要求），helper 里同样出现 `SetDeqScale` 反量化标量接口；
4. smooth scale 可选输入 = 每行多乘一个固定缩放（第二路 y2/scale2），本 mini 版未展开。

## 四、CHECKLIST 增量

- **B31（新）**：行 max 快路径 = strided Max 折叠树（dstRepStride=0 折叠 + WholeReduceMax 收口），大行禁用 mask-count ReduceMax（生产注释 very very slow）；量化路径 = Abs 独立载体 → 树形 max → 标量 127/max → Muls(1/scale) 广播 → 两段 cast int8。
- **A5 补充**：量化输出对（int8+fp32 scale）是融合链的独立截断契约点。

## 下一轮候选

foreach_unary_v2 / foreach_copy（工厂/搬运收尾）或 add_layer_norm_quant（双 smooth 路量化）。
