# Round 39 复盘：AddLayerNorm（驻留行 + 阶段循环）

> 我的实现：`my_impl/round39_add_layernorm/`（DESIGN.md / add_layer_norm_custom.h）
> 对比对象：`add_layer_norm/op_kernel/add_layer_norm_kernel.h`（1410 行）主路径

---

## 一、norm 族三形态对表（R21/R22/R39 归拢）

| 形态 | 驻留单位 | 方差路径 | 适用 |
| --- | --- | --- | --- |
| single-read（R21） | 整矩阵 | 单遍 E[x²]−mean² | 行中等、整矩阵装得下 |
| transpose（R22） | 多行块 | 两遍（原地减 mean） | 行多且短，转置凑归约宽 |
| **add_layer_norm（本轮）** | **单行 + 阶段循环** | 两遍（原地减 mean） | **行长 > UB 单块宽度**，列块循环 |

三形态共享同一数学（mean/var/normalize），差异只在**驻留单位与扫描次数**。生产 base.h（469 行）承载共享逻辑（DataCopyEx/参数推导），三个 kernel 形态各自实现扫描序。

## 二、本轮结构新知

1. **xOut 在 Phase0 流式写出**：additional output 顺带在 add 完成的列块上直接写——三输出中唯一不占额外扫描的（写出时机前移到首次扫描）。
2. **模板旗标密度峰值**：`IS_X2_NEEDCAST / IS_BIAS_PRESENT / IS_BIAS_BROADCAST / IS_ADDITIONAL_OUTPUT_ENABLE / OUTPUT_MEAN_RSTD`——同一 kernel 承载 6+ 种配置组合，编译期特化（B11 集大成；代价是 1410 行）。
3. **bias 双模式**：独立输入（逐元素）或广播向量（列块偏移对齐）——BROADCAST 模式下 bias 常驻（B25）。
4. **mean/var 标量累加 O(chunks)**：生产主路径每列块一次 GetValue 累加——R37 的"全 tensor 化"在 chunks 很少时反而繁琐，**B3 允许级的实证**（标量累加的量级判定：≤ 列块数）。

## 三、我的差距

1. γ/β 我假设可整行常驻（`wBuf` 两个 rowSize 段）——生产对长行 γ/β 也按列块处理（`gamma_local[col_offset]` 偏移切片），我的常驻假设在小 UB 下不成立；
2. `wBuf.Get<float>()[rowSize]` 同 TBuf 偏移取段是示意写法（未验证，//??）；
3. eps 重犯自查第三次（R21/R22/R39）——已进重犯表，下轮实现模板中 eps 必须从 tiling 读。

## 四、CHECKLIST 增量

- **A6 补充（norm 族三形态终表）**：驻留单位选择 = 行长 vs UB——整矩阵（single-read 单遍）/ 多行块（transpose 两遍）/ 单行列块循环（两遍）；扫描次数由驻留单位决定，xOut 类附加输出在首次扫描流式写出。
- **B3 量级判定精化**：标量累加允许级 = O(chunks) 且 chunks 少；chunks 多时改张量化（R37）。

## 下一轮候选

dynamic_quant_update_scatter（量化+散写组合）或 foreach 系收尾（foreach_lerp_list 等）。
