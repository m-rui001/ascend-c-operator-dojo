# Round 11 复盘：RmsNorm 复合算子

> 我的实现：`my_impl/round11_rmsnorm/`（DESIGN.md / rms_norm_custom.cpp / op_host/）
> 对比对象：`cann-ops/src/norm/rms_norm/`（5 变体：NORMAL / SPILT_D / MERGE_N / SINGLE_ROW / WHOLE_REDUCE_SUM × dtype 模板，约 2000 行 kernel）
> 本轮定位：十轮积累的综合检验场（行归约+双输出+dtype 模板+变体路由）

---

## 一、//?? 销案结果

| //?? | 结果 |
| --- | --- |
| 1. RmsNorm 高阶 API | **不存在**（生产全部基础 API 手搓）——"先查高阶 API"检查项仍要跑，查无才手写 |
| 2. Sqrt 形态 | **矢量化 Sqrt 接口**（对 rstd 向量按 repeat 批量操作），不是我写的标量 `sqrtf` 逐行循环 |
| 3. Mul 次序 | 生产也是两步 Mul（x*rstd 再 *gamma_fp32），且 gamma **一次性 Cast 成 fp32 复用** |
| 4. rstd 写出 | 生产用独立 `outQueueRstd`（1 槽，大小 `tiling->rstd_size`）走队列，不是我设计的"攒批裸 DataCopy" |
| 5. whole_reduce_sum 变体的特殊用法 | 见下——row_factor 批行 + fp32 常驻 |

## 二、生产实现的四个高招（对照我的写法）

### 1. rstd 计算矢量化——Round 3 教训的重犯与正解

我逐行 `sqrtf` + `SetValue`（标量通路，Round 3 明知的反模式）。生产把 row_factor **行的 rstd 当作一个向量**处理：`Muls(avgFactor) → Adds(epsilon) → Sqrt` 全部按 repeat 步长批量作用于多行结果——**归约结果的后续计算也应留在 tensor 通路**。这是第 3 轮教训在"归约之后"半场的再次应用。

### 2. row_factor 批行——2D 块哲学第三次出现

我一次一行（Round 2 的写法）；生产一次 row_factor 行：Cast/归约/归一化全部摊薄到多行。"tiling 的原子是 UB 块不是语义单元"（Round 5）在行算子上的再现。

### 3. fp32 常驻单遍 vs 我的两遍扫描

行装得下时（fp32 视角），生产 **Cast 一次进 UB 常驻**，平方和与归一化共用同一份 x_fp32，x 只读 1 遍；行装不下才有 split_d 两遍变体。我的设计把两遍当默认路径是错的——**先算 fp32 常驻能否装下，再决定单/双遍**（UB 预算按 fp32×2 份算，不是按输入 dtype）。

### 4. ReinterpretCast 缓冲覆盖

`sqx = xLocal.ReinterpretCast<float>()`、`gamma_fp32 = gammaLocal.ReinterpretCast<float>()`——同一块 UB 按不同 dtype 视图复用（half 布局上直接盖 fp32 中间量）。比 TBufPool 更细粒度的 UB 压缩手段，我在第 8 轮没学到这个。

### 其他差异

- `avgFactor = 1/N` 由 host 算好下发（能移到 host 的标量都移走）；
- `ReduceSumHalfInterval` 共用 helper（reduce_common.h）处理非对齐区间归约——生产把可复用的小算子抽成共享头文件；
- host 侧按行/列规模分支（4 个 SetBlockDim 分支）+ TilingKey 路由 5 变体——Round 6 "tiling 输出执行策略"的再现；
- in-place `Mul(sqx, x_fp32, x_fp32)`（dst=src0）与第 9 轮销案一致。

## 三、我的实现缺陷清单

1. rstd 标量循环（上述第 1 条，Round 3 教训重犯——"必检清单"要加第 6 项：**归约结果的所有后续计算必须 tensor 化**）；
2. 两遍扫描当默认（应先按 fp32 常驻判断单遍可行）；
3. `work[ubTile]` 偏移寻址 WholeReduceSum 的输出区——写法未验证（社区用独立 reduce_fp32_buf）；
4. `f32Buf` 按 `ubTile*sizeof(float)*2` 一块双用——同 Round 2 的 TBuf 切两块错误，正确做法是两个 TBuf 或 ReinterpretCast 覆盖；
5. DataCopy 尾块 pad（第 10 轮必检项）本轮仍未逐处落实——`LoadGammaRow`/行搬运的 cols 非 32B 场景仍裸用 DataCopy；
6. gamma 滚动片在 split 路径每行重搬（应行间复用）。

## 四、经验教训

1. **复合算子 = 已学模式的组合考**：本轮全部考点（行归约、fp32 中间量、行广播复用、变体路由、双输出）都是旧知识，丢分全在"教训没有变成检查清单条目"——第 3 条教训（标量通路）和第 5 条（2D 块）都是二次重犯。**行动项：把十轮教训正式固化为 CHECKLIST.md，每轮 DESIGN 逐项打钩**（下一轮开头执行）。
2. **UB 预算要按"中间量的 dtype"算**，不是输入 dtype——fp32 常驻是精度的解，也是带宽的解。
3. **ReinterpretCast 覆盖**进入内存工具箱（TBufPool/池复用/in-place/覆盖，四种粒度）。
4. 归一化类算子的通用骨架：Cast→平方和归约→rstd 矢量化→Mul 链→Cast 回，gamma fp32 化复用。

## 十一轮总览

| 轮 | 主题 | 最大盲区 |
| --- | --- | --- |
| 1-10 | （见前几轮） | — |
| 11 | RmsNorm 综合 | 教训二犯（标量通路/逐行处理）；fp32 常驻决定单/双遍；ReinterpretCast 覆盖 |

**下一轮**：先写 `CHECKLIST.md`（十轮教训→逐项检查单，作为后续所有轮次的 DESIGN 模板前置），然后继续第 12 轮（候选：AddLayerNorm 融合变体 / Gelu / 或 2_features 有代码的 10_communicate_compute_fused）。
