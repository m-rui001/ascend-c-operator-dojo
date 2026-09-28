# Round 39: AddLayerNorm —— 我的设计（A 节前置）

> 语义：x = x1+x2(+bias)；y = (x−mean)·rstd·γ+β；可选 xOut/mean/rstd 输出。
> 已读：`add_layer_norm_kernel.h`（1410 行）主路径段。

## A. CHECKLIST 设计必答

1. **结构（与 R22/R21 对比定位）**：**驻留行 + 阶段循环**——整行驻留 UB（x_local_fp32），四个阶段（add/mean/var/normalize）各自对列块循环。与 R22（多行块驻留、块级两遍）互为补充：行较长时"单行驻留+阶段循环"，行短时"多行块+批归约"。
2. **模板旗标密度**：`IS_X2_NEEDCAST / IS_BIAS_PRESENT / IS_BIAS_BROADCAST / IS_ADDITIONAL_OUTPUT_ENABLE / OUTPUT_MEAN_RSTD`——可选输入/输出/广播模式全部编译期旗标（B11 集大成）。
3. **xOut 流式写出**：additional output 在 Phase0（add 完成即写），**不需要额外扫描**——三输出里唯一直写流式的。
4. **mean/var 标量累加**：`ave_tmp += GetValue(0)` 每列块一次（O(chunks)，B3 允许级；对比 R37 全 tensor 化的边界）。

## 1. 我的实现（三阶段 mini，fp32 输入）

Phase0 add(+bias)+xOut 流式 → Phase1 mean（列块 Mul·ReduceSum 标量累加）→ Phase2 var（Adds(−mean)→Mul→Muls(1/N)→ReduceSum 累加）→ Phase3 normalize（Muls(rstd)→γ Mul→β Add→写出）。
