# Round 30 复盘：foreach_copy（纯搬运的退化流水形态）

> 我的实现：`my_impl/round30_copy/`（DESIGN.md / foreach_copy_custom.h）
> 对比对象：`foreach/foreach_copy/op_kernel/foreach_copy.h`（186 行独立类）

---

## 一、生产形态确认（与我的设计一致）

1. **无 VECOUT 队列**：`ComputeAndCopyOut` 从 dataQueue 的 local 直接 DataCopyPad 写 out GM，再 FreeTensor——单 VECIN 队列兼任输入输出缓冲。这正是 R9 学的 **TQueBind 的手工等价**（VECIN/VECOUT 绑定 = 同一缓冲两用），生产在"无计算"场景选择手写而非绑定 API（foreach 系未用 v2 工厂也未用 bind）。
2. **BUFFER_NUM=1**：搬运瓶颈型 + 单队列直出的最简流水（B7 的第十例——纯搬运连双缓冲都省，EnQue/DeQue 只做 MTE2→MTE3 的依赖排序）。
3. **12 个 TilingKey 扇出**：fp16/fp32/bf16/int8/uint8/int32/uint32/uint64/double/bool 全 dtype 覆盖，`INIT_AND_PROCESS` 宏消重——纯搬运算子的 dtype 维度拉满（无计算精度差异，dtype 只影响搬运宽度）。

## 二、工厂适用性边界（R16 结论的补全）

foreach_copy **不走工厂**（独立类）——工厂的 Compute 钩子在纯搬运场景退化为空，"计算形"骨架反而累赘。结合 R18/R23（数值复杂的也算独立类）：

| 算子特征 | 走向 |
| --- | --- |
| 标准"读-算-写"且语义简单 | v2 工厂（Adapter 一两行） |
| 数值分支复杂（lerp）或含归约（norm） | 独立类 |
| 无计算的纯搬运 | 独立类（退化流水） |

**工厂覆盖的是中间带**——两端的退化形态都走独立类。CHECKLIST E 节补充此边界。

## 三、我的差距

1. 我的实现与生产结构一致（单队列直出），差异仅在：生产把 CopyIn 与 ComputeAndCopyOut 拆成两函数（我可合并）；生产 12 dtype 扇出我只写模板（等价）；
2. 生产 `ComputeAndCopyOut` 里 float32Tensor 参数仍被传入（工厂签名惯性）——独立类保留工厂签名是历史痕迹，新写不需要。

## 四、CHECKLIST 增量

- **E8（新）**：工厂适用中间带——纯搬运（无 Compute）与数值特化（复杂分支/归约）走独立类；纯搬运的退化流水 = 单 VECIN 队列直出（TQueBind 手工等价）+ BUFFER_NUM=1。

## 下一轮候选

add_layer_norm_quant（双 smooth 量化收尾 norm 族）或 foreach_log（超越函数列表版，验证 R12 多项式逼近在工厂里的封装）。
