# Round 22 复盘：LayerNorm transpose 策略（many-rows 块状处理）

> 我的实现：`my_impl/round22_ln_transpose/`（DESIGN.md / layer_norm_block2pass.h，两遍方差简化版）
> 对比对象：`layer_norm_v4_transpose.h`（914 行）——聚焦 Process/ProcessBasicBlock/CalcGeneralParams。

---

## 一、生产 transpose 策略的解码

场景：**行多而短**（每 ubFormer 行一块循环）。每块流程：

1. `CopyInPad` 2D 载入（rightPadding 对齐到 X_NUM_PER_BLOCK）；
2. **`DoTranspose` + `DoReshape`**：UB 内硬件转置——把"多行×行内连续"重排为"归约维连续"的布局，让逐行归约变成规整的批归约（短行场景每行归约宽度太小，转置后按列凑满归约宽度）；
3. cast fp32 → `Muls(coef)` → `DoReduce` 得**每行 mean** → 写 mean；
4. **`DoSub(xLocalFp32, mean)` 原地减** → `Mul` 平方 → `Muls(coef)` → `DoReduce` 得 E[(x−mean)²] → `Adds(eps)` → `Sqrt` → **`DoDiv(outM, oneTensor, temp)` 求 rstd**（`Duplicate(1)` 全 1 向量作分子，向量除法代倒数）；
5. `DoMulGamma` / `DoAddBeta` → 转置回 → 写出。

## 二、R21 结论的边界修正（本轮最重要的认知）

R21 我记了"single_read 用单遍 E[x²]−mean²"——**transpose 变体用的却是两遍**：`DoSub(x−mean) 原地 → 平方 → Σ`。两者都不重读 GM！真正的决策变量是：

| | single-read（整矩阵驻留） | transpose（块驻留） |
| --- | --- | --- |
| 方差 | 单遍 E[x²]−mean²（省一次 UB 内平方+归约） | 两遍（原地减 mean 后平方，精度更好） |
| 适用 | 行长中等、整矩阵装得下 | 行多且短、需要转置凑归约宽度 |

**CHECKLIST A6 精化：两遍/单遍的判定顺序 = ①块是否驻留 UB（驻留→两遍零 GM 代价，精度优先）→ ②UB 指令预算紧张→才用单遍 E[x²]−mean²。** "单遍 vs 两遍"不是带宽问题（都不重读 GM），是 UB 指令数与精度的权衡。

## 三、新知清单

1. **`DoTranspose`/`DoReshape` UB 内硬件转置**：many-short-rows 场景让归约维连续、凑满归约宽度——这是"数据布局为归约服务"的又一层（B4 的进阶：不仅块是 2D，必要时转置布局）。
2. **倒数 = `Duplicate(1) + Div(ones, x)`**：生产不走 Reciprocal 指令（R20 刚记的"矢量÷矢量用 Div"在此复用——1/x 也是矢量除矢量）。B8 再精化：倒数统一走 Div(ones,·)。
3. **mean/rstd 每块一次 pad 写出**（`CopyOutMeanOrRstd`，`outQueueMean/Rstd` 小队列 + intriParams），与我的攒批一致。
4. former/tail 循环块大小核（R1 模式在"循环块"粒度复现，非行/元素粒度）。

## 四、我的差距与自查

1. **硬件转置缺失**：我的简化版用 rowAlign 布局逐行归约，短行（rowSize 小）时归约宽度不满、效率低——正是生产引入转置的原因；生产证据把"何时该转置"的判据补进 CHECKLIST（**B24**：短行批归约，行宽不足归约向量大小时考虑 UB 内转置凑宽）。
2. **rstd 求 1/· 的写法**：我标量 `1.0f/sqrt()`（GetValue 后标量除法），生产向量 Div——生产保持 tensor 通路（B3 精化：哪怕 1 元素也尽量留在 tensor 通路，避免 S→V 回程）。
3. **eps 重犯自查**（R21 已记录）：标量进常量的惯性仍在，改 tiling 下发是下次实现第一动作。
4. γ/β 藏载我注释了但未实现——生产在块循环开始前一次性载入 γ/β（短行场景 γ/β 很小，一次载入常驻即可，与 R21 长行场景的"循环内预取"不同——**隐藏加载的适用性取决于系数尺寸**，B22 精化）。

## CHECKLIST 增量

- **A6 精化**：两遍方差判定 = 驻留 UB→两遍（精度）；非驻留→单遍或重读。（R22）
- **B8 再精化**：倒数统一 `Duplicate(1)+Div`，tensor 通路优先于标量除法。（R22）
- **B24（新）**：短行批归约行宽不足时，考虑 UB 内转置（DoTranspose 类）凑归约宽度；长行场景不转置。（R22）

## 下一轮候选

foreach_lerp_scalar（低成本）或 `foreach_copy`/`foreach_unary_v2`（纯搬运工厂 v2，验证 TQueBind 在 v2 的形态）。
