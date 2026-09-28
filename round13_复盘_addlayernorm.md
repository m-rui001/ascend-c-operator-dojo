# Round 13 复盘：AddLayerNorm 融合算子

> 我的实现：`my_impl/round13_addlayernorm/`（DESIGN.md / add_layer_norm_custom.cpp）
> 对比对象：`cann-ops/src/norm/add_layer_norm/`（kernel 3743 行：base 归约工具箱 + 4 变体 + 1410 行主 kernel）
> 本轮核心检验：R12 精化的 A1 条目（融合算子→手拼）与 R11 的 fp32 常驻/矢量化教训

---

## 一、//?? 销案结果

| //?? | 结果 |
| --- | --- |
| 1. LayerNorm 高阶 API 适配性 | 生产**未用**任何高阶 API，全基础 API 手拼 + 模板特性旗标——A1 精化条目（融合→手拼）再获实证 |
| 2. 方差单遍 vs 两遍 | **两遍确认**：pass1 求和→ave；pass2 `Adds(-ave)→Mul 平方→ReduceSum`。我选对了（数值稳定优先） |
| 3. bias 广播 | 生产把 bias 并入加法链（add_buf_local），广播细节在变体内部；我的"标量广播"简化仍不合格，但结构方向一致 |
| 4. less_tensor 变体 | 单行 UB 紧张模式：减少常驻张量数（复用缓冲），即"fp32 常驻装不下"的降级路径——与我的双路径思想同构 |

## 二、结构验证：我的骨架与生产一致（CHECKLIST 流程的正反馈）

生产的计算骨架：**Add(x1,x2) 进 fp32 常驻（add_buf_local/x_local_fp32）→ 两遍 mean/var → rstd → 逐列片归一化 → Cast 回 half**——与我的 DESIGN 四阶段完全同构。这是 CHECKLIST 流程第一次在"结构层面"命中生产设计（前 12 轮都是我错、生产对；本轮骨架对、细节错），说明十轮教训的沉淀开始改变设计直觉。

## 三、新知识点（生产代码习得）

1. **V_S 事件对：归约结果读标量前必须握手**。生产的每次 `ReduceSum` 后接：
   ```cpp
   SetFlag<HardEvent::V_S>(EVENT_ID0);
   WaitFlag<HardEvent::V_S>(EVENT_ID0);
   ave_tmp += y_local_fp32.GetValue(0);
   ```
   我全库 13 轮都在裸 `GetValue(0)`——**这是潜在正确性 bug**（V 管结果未落就读）。B3 条目升级：归约输出读标量必须 V_S 事件对。
2. **Adds 的三个 idiom**：`Adds(dst, src, -ave)` 代替减标量（无 Subs 依赖）；`Adds(dst, src, ZERO)` 当 fp32 的 UB 内拷贝（走 V 管不走 MTE）；`Muls(1/N)` 折进每个 tile 的方差累加（省尾部一步）。
3. **模板特性旗标**：`IS_ADDITIONAL_OUTPUT_ENABLE`、`IS_BETAGAMMA_NEEDCAST`、`OUTPUT_MEAN_RSTD` 等编译期开关——把 TilingKey（R4/R6 的变体路由）细化到**特性粒度**，可选功能零运行时开销。我的 `if (writeX)` 运行时分支是反模式。
4. **归约工具箱分层**：base 头文件提供 ReduceSumFP32 / ReduceSumShort / ReduceSumForSmallReduceDim 等 4+ 个 helper，按 N 的规模选用——R11 的"共用 helper"模式的完整形态。

## 四、我的实现缺陷清单

1. **裸 GetValue 无 V_S 握手**（上述，潜在 bug 级）；
2. 两遍路径的 VarRow/NormalizeRow 直接重读常驻缓冲——大行场景（不 fit）时缓冲里根本没有完整行，//? 已标注但结构上就是错的；正确做法是生产行为：大行变体逐片流式重算（special_reduce / less_tensor 就是干这个的）；
3. mean/rstd 攒批用 stride-2 交错布局再 [coreRows] 取半——布局混乱，生产用独立 mean_local/rstd_local；
4. writeX 输出路径只写了注释（设计 drop，R10 教训第 4 条再犯——但本轮 DESIGN/代码的落差在复盘全部勾销登记）；
5. NormalizeRow 里 fp32→half→乘 gamma 的来回 Cast，生产在 fp32 域完成 gamma/beta（IS_BETAGAMMA_NEEDCAST 控制一次 Cast）。

## 五、经验教训

1. ** GetValue 前的 V_S 事件对**是十三轮里第一个"不知即错"的硬规则——补进 CHECKLIST B 区。
2. **模板特性旗标 > 运行时 if**：可选输出/可选精度路径应编译期展开，这与 TilingKey 一起构成"host 决策→编译期实例化"的完整链路。
3. Adds 负标量/零标量两个 idiom 值得记住（减法与 UB 拷贝都有不走 MTE 的实现）。
4. 结构层面的命中说明 CHECKLIST 在收敛；剩余差距集中在"细粒度正确性"（事件对、布局）与"降级路径完整性"（大行变体）——比 12 轮前的"结构都错"是质变。

## 十三轮总览

| 轮 | 主题 | 最大盲区 |
| --- | --- | --- |
| 1-12 | （见前几轮） | — |
| 13 | AddLayerNorm | V_S 事件对（裸 GetValue 隐患）；模板特性旗标；降级路径完整性 |

**下一轮候选**：把 V_S/HardEvent 事件体系专题化（对照 API 文档把 V_MTE3/V_S/S_MTE2 等事件对枚举全，深化 R9/R10/R13 的同步知识）；或继续算子线（Gelu 检验 R12 精化后的独立算子路线）。
